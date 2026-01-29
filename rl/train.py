#!/usr/bin/env python3
"""
Multi-map SAC training script for akmSteerSim.

Features:
- Fixes import path so `envs.*` works when running: `python rl/train.py ...`
- Supports curriculum/map schedule from configs/sac.yaml (map_schedule)
- Evaluates on random held-out maps (maps NOT in schedule) OR user-provided eval list
- Saves checkpoints and best model
- TensorBoard logging under runs/<run_id>/

Run:
  source .venv/bin/activate
  python rl/train.py --vehicle_cfg configs/vehicle.yaml --sac_cfg configs/sac.yaml
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy


import yaml

# --- make repo root importable (fixes: ModuleNotFoundError: envs) ---
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from envs.f1tenth_sb3_env import F1TenthSACEnv


# ----------------------------
# Helpers: map/track resolution
# ----------------------------

def list_tracks(assets_dir: Path) -> List[str]:
    """
    Returns track folder names under assets/f1tenth_racetracks.
    Folder structure assumed:
      assets/f1tenth_racetracks/<Track>/<Track>_map.yaml
      assets/f1tenth_racetracks/<Track>/<Track>_map.png
      assets/f1tenth_racetracks/<Track>/<Track>_centerline.csv
    """
    if not assets_dir.exists():
        return []
    out = []
    for p in assets_dir.iterdir():
        if p.is_dir():
            out.append(p.name)
    return sorted(out)


def normalize_track_name(track: str) -> str:
    """
    In configs you might write "Sakhir" but sim map_name is usually "Sakhir_map".
    Track folder is "Sakhir". Centerline is "Sakhir_centerline.csv".
    """
    return str(track).replace("_map", "").strip()


def map_name_from_track(track: str) -> str:
    """
    Sim map_name should be "<Track>_map".
    """
    t = normalize_track_name(track)
    return f"{t}_map"


def resolve_map_dir(vehicle_cfg: Dict[str, Any], track: str) -> Path:
    """
    If user provided sim.map_dir, respect it.
    Else default to assets/f1tenth_racetracks/<Track>
    """
    sim_cfg = vehicle_cfg.get("sim", {})
    t = normalize_track_name(track)
    default_dir = ROOT / "assets" / "f1tenth_racetracks" / t
    return Path(sim_cfg.get("map_dir", str(default_dir)))


def resolve_centerline_csv(vehicle_cfg: Dict[str, Any], track: str) -> Path:
    t = normalize_track_name(track)
    map_dir = resolve_map_dir(vehicle_cfg, track)
    return Path(map_dir) / f"{t}_centerline.csv"

# rl/train.py (replace your make_env_for_track with this)


ROOT = Path(__file__).resolve().parents[1]

def make_env_for_track(vehicle_cfg: dict, track: str, render_mode=None):
    """
    Create an env for a given track using the repo's track folder structure:
      assets/f1tenth_racetracks/<Track>/<Track>_centerline.csv
      assets/f1tenth_racetracks/<Track>/<Track>_map.png
      assets/f1tenth_racetracks/<Track>/<Track>_map.yaml
    """
    track_dir = ROOT / "assets" / "f1tenth_racetracks" / track
    if not track_dir.exists():
        raise FileNotFoundError(f"Track folder not found: {track_dir}")

    cl = track_dir / f"{track}_centerline.csv"
    if not cl.exists():
        raise FileNotFoundError(f"Centerline CSV not found for track '{track}': {cl}")

    # Make a per-track copy of cfg and patch sim fields
    cfg = deepcopy(vehicle_cfg)
    cfg.setdefault("sim", {})
    cfg["sim"]["map_name"] = f"{track}_map"
    cfg["sim"]["map_dir"] = str(track_dir)

    # If your env uses these keys, it's fine to set them too (harmless otherwise)
    cfg["sim"]["track_name"] = track

    from envs.f1tenth_sb3_env import F1TenthSACEnv
    env = F1tenthSACEnv = F1TenthSACEnv(
        vehicle_cfg=cfg,
        track_centerline_csv=str(cl),
        render_mode=render_mode,
    )
    return env



# ----------------------------
# Evaluation callback (random held-out map)
# ----------------------------

@dataclass
class EvalResult:
    track: str
    mean_reward: float
    mean_len: float


class RandomMapEvalCallback(BaseCallback):
    """
    Every eval_freq steps:
      - pick a random track from eval_tracks
      - create a fresh env for that track
      - run n_eval_episodes with deterministic policy
      - log eval/mean_reward, eval/mean_ep_len, eval/track_name
      - optionally save best model
    """
    def __init__(
        self,
        vehicle_cfg: Dict[str, Any],
        eval_tracks: List[str],
        eval_freq: int,
        n_eval_episodes: int,
        best_model_dir: Path,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.vehicle_cfg = vehicle_cfg
        self.eval_tracks = eval_tracks
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.best_model_dir = best_model_dir
        self.deterministic = deterministic

        self.best_mean_reward = -float("inf")
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

    def _run_eval(self, track: str) -> EvalResult:
        env = make_env_for_track(self.vehicle_cfg, track, render_mode=None)
        rewards = []
        lengths = []

        try:
            for _ in range(self.n_eval_episodes):
                obs, info = env.reset()
                done = False
                ep_r = 0.0
                ep_len = 0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_r += float(reward)
                    ep_len += 1
                    done = bool(terminated or truncated)

                rewards.append(ep_r)
                lengths.append(ep_len)
        finally:
            try:
                env.close()
            except Exception:
                pass

        mean_r = sum(rewards) / max(1, len(rewards))
        mean_l = sum(lengths) / max(1, len(lengths))
        return EvalResult(track=track, mean_reward=float(mean_r), mean_len=float(mean_l))

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True

        if (self.num_timesteps % self.eval_freq) != 0:
            return True

        track = random.choice(self.eval_tracks) if self.eval_tracks else ""
        if not track:
            return True

        res = self._run_eval(track)

        # TensorBoard scalars
        self.logger.record("eval/mean_reward", res.mean_reward)
        self.logger.record("eval/mean_ep_len", res.mean_len)
        # Can't log strings as scalars; log as a numeric hash-ish id:
        self.logger.record("eval/track_idx", float(self.eval_tracks.index(res.track)))
        self.logger.dump(self.num_timesteps)

        # Save best model
        if res.mean_reward > self.best_mean_reward:
            self.best_mean_reward = res.mean_reward
            save_path = self.best_model_dir / "best_model"
            self.model.save(str(save_path))
            if self.verbose:
                print(f"[eval] New best mean_reward={res.mean_reward:.3f} on track={res.track} -> saved {save_path}")

        if self.verbose:
            print(f"[eval] step={self.num_timesteps} track={res.track} mean_reward={res.mean_reward:.3f} mean_len={res.mean_len:.1f}")

        return True


# ----------------------------
# Main training
# ----------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle_cfg", default="configs/vehicle.yaml")
    ap.add_argument("--sac_cfg", default="configs/sac.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n_eval_episodes", type=int, default=3)
    ap.add_argument("--eval_tracks", default="", help="Optional comma-separated tracks for eval (e.g. 'Austin,Monza'). If empty, uses held-out maps.")
    args = ap.parse_args()

    vehicle_cfg_path = ROOT / args.vehicle_cfg
    sac_cfg_path = ROOT / args.sac_cfg

    veh_cfg = load_yaml(vehicle_cfg_path)
    sac_cfg = load_yaml(sac_cfg_path)

    if not veh_cfg:
        raise ValueError(f"Vehicle config is empty/invalid: {vehicle_cfg_path}")
    if not sac_cfg:
        raise ValueError(f"SAC config is empty/invalid: {sac_cfg_path}")

    seed = int(sac_cfg.get("seed", 0))
    set_random_seed(seed)

    assets_tracks_dir = ROOT / "assets" / "f1tenth_racetracks"
    available_tracks = list_tracks(assets_tracks_dir)

    schedule = sac_cfg.get("map_schedule", None)
    if not schedule:
        # fallback: use current map_name to infer a single track
        sim_cfg = veh_cfg.get("sim", {})
        raw_map = sim_cfg.get("map_name", "Sakhir_map")
        default_track = normalize_track_name(raw_map)
        schedule = [{"track": default_track, "steps": int(sac_cfg["train_steps"]), "eval_freq": int(sac_cfg["eval_interval_steps"])}]

    schedule_tracks = [normalize_track_name(item["track"]) for item in schedule]

    # --- eval tracks: user-provided OR held-out (not in schedule) ---
    if args.eval_tracks.strip():
        eval_tracks = [normalize_track_name(t) for t in args.eval_tracks.split(",") if t.strip()]
    else:
        held_out = [t for t in available_tracks if t not in set(schedule_tracks)]
        # “random out of 4 the model hasn't seen” (if available)
        random.shuffle(held_out)
        eval_tracks = held_out[:4] if len(held_out) >= 4 else held_out

    # If no held-out maps exist, eval on current schedule track(s)
    if not eval_tracks:
        eval_tracks = schedule_tracks[:1]

    run_id = time.strftime("%Y%m%d-%H%M%S")
    runs_dir = ROOT / "runs"
    ckpt_root = ROOT / "checkpoints" / run_id
    ckpt_root.mkdir(parents=True, exist_ok=True)

    print("=== Training setup ===")
    print("run_id:", run_id)
    print("schedule_tracks:", schedule_tracks)
    print("eval_tracks:", eval_tracks)
    print("available_tracks:", available_tracks[:10], ("..." if len(available_tracks) > 10 else ""))
    print("tensorboard_log:", runs_dir)
    print("checkpoints:", ckpt_root)

    # Create initial env from first schedule track
    first_track = normalize_track_name(schedule[0]["track"])
    env = make_env_for_track(veh_cfg, first_track, render_mode=None)

    model = SAC(
        policy=sac_cfg["policy"],
        env=env,
        learning_rate=sac_cfg["learning_rate"],
        batch_size=sac_cfg["batch_size"],
        tau=sac_cfg["tau"],
        gamma=sac_cfg["gamma"],
        seed=seed,
        policy_kwargs=sac_cfg.get("policy_kwargs", None),
        verbose=1,
        tensorboard_log=str(runs_dir),
        device=args.device,
        ent_coef=sac_cfg.get("ent_coef", "auto"),
    )

    # Train per phase
    total_so_far = 0
    for phase_idx, phase in enumerate(schedule):
        track = normalize_track_name(phase["track"])
        steps = int(phase["steps"])
        eval_freq = int(phase.get("eval_freq", sac_cfg.get("eval_interval_steps", 5000)))

        # swap env if needed
        if phase_idx == 0:
            # already created
            pass
        else:
            try:
                env.close()
            except Exception:
                pass
            env = make_env_for_track(veh_cfg, track, render_mode=None)
            model.set_env(env)

        phase_dir = ckpt_root / f"{phase_idx:02d}_{track}"
        phase_dir.mkdir(parents=True, exist_ok=True)

        callbacks = []

        # checkpoint often
        save_every = int(sac_cfg.get("save_every_steps", 10000))
        callbacks.append(
            CheckpointCallback(
                save_freq=save_every,
                save_path=str(phase_dir),
                name_prefix="sac_f1",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )

        # random-map eval
        callbacks.append(
            RandomMapEvalCallback(
                vehicle_cfg=veh_cfg,
                eval_tracks=eval_tracks,
                eval_freq=eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                best_model_dir=phase_dir,
                deterministic=True,
                verbose=1,
            )
        )

        print(f"\n=== Phase {phase_idx} ===")
        print("track:", track)
        print("map_name:", map_name_from_track(track))
        print("map_dir:", resolve_map_dir(veh_cfg, track))
        print("centerline:", resolve_centerline_csv(veh_cfg, track))
        print("steps:", steps, "eval_freq:", eval_freq)

        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            tb_log_name=f"{run_id}",
            callback=callbacks,
            progress_bar=True,
        )

        total_so_far += steps

        # Save a phase snapshot
        model.save(str(phase_dir / "model_phase_end"))

    # Final save
    final_path = ckpt_root / "sac_final"
    model.save(str(final_path))
    print("\nSaved final model:", final_path)

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
