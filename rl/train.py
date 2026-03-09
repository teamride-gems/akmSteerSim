#!/usr/bin/env python3
"""
Multi-map SAC training script for akmSteerSim.

Features:
- Supports curriculum/map schedule from configs/sac.yaml (map_schedule)
- Evaluates on ALL held-out maps from sac.yaml OR user-provided eval list
- Training resets remain random
- Eval resets use fixed spawn indices for repeatability
- Saves checkpoints and best model
- TensorBoard logging under runs/<run_id>/
"""

from __future__ import annotations

import argparse
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from envs.f1tenth_sb3_env import F1TenthSACEnv


def list_tracks(assets_dir: Path) -> List[str]:
    if not assets_dir.exists():
        return []
    return sorted([p.name for p in assets_dir.iterdir() if p.is_dir()])


def normalize_track_name(track: str) -> str:
    return str(track).replace("_map", "").strip()


def map_name_from_track(track: str) -> str:
    t = normalize_track_name(track)
    return f"{t}_map"


def resolve_map_dir(track: str) -> Path:
    t = normalize_track_name(track)
    return ROOT / "assets" / "f1tenth_racetracks" / t


def resolve_centerline_csv(track: str) -> Path:
    t = normalize_track_name(track)
    return resolve_map_dir(t) / f"{t}_centerline.csv"


def make_env_for_track(vehicle_cfg: Dict[str, Any], track: str, render_mode=None):
    track = normalize_track_name(track)
    track_dir = resolve_map_dir(track)

    if not track_dir.exists():
        raise FileNotFoundError(f"Track folder not found: {track_dir}")

    cl = resolve_centerline_csv(track)
    if not cl.exists():
        raise FileNotFoundError(f"Centerline CSV not found for track '{track}': {cl}")

    cfg = deepcopy(vehicle_cfg)
    cfg.setdefault("sim", {})
    cfg["sim"]["map_name"] = map_name_from_track(track)
    cfg["sim"]["map_dir"] = str(track_dir)
    cfg["sim"]["track_name"] = track

    env = F1TenthSACEnv(
        vehicle_cfg=cfg,
        track_centerline_csv=str(cl),
        render_mode=render_mode,
    )
    return env


@dataclass
class EvalResult:
    track: str
    mean_reward: float
    mean_len: float


class RandomMapEvalCallback(BaseCallback):
    """
    Old behavior:
      Every eval_freq steps, pick ONE random held-out track and evaluate on it.
    Kept here so you can switch back easily.
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

        self.logger.record("eval/mean_reward", res.mean_reward)
        self.logger.record("eval/mean_ep_len", res.mean_len)
        self.logger.record("eval/track_idx", float(self.eval_tracks.index(res.track)))
        self.logger.dump(self.num_timesteps)

        if res.mean_reward > self.best_mean_reward:
            self.best_mean_reward = res.mean_reward
            save_path = self.best_model_dir / "best_model"
            self.model.save(str(save_path))
            if self.verbose:
                print(
                    f"[eval] New best mean_reward={res.mean_reward:.3f} "
                    f"on track={res.track} -> saved {save_path}"
                )

        if self.verbose:
            print(
                f"[eval] step={self.num_timesteps} "
                f"track={res.track} mean_reward={res.mean_reward:.3f} "
                f"mean_len={res.mean_len:.1f}"
            )

        return True


class HeldoutMapsEvalCallback(BaseCallback):
    """
    New behavior:
      Every eval_freq steps, evaluate on ALL held-out tracks.
      Eval uses fixed spawn indices for repeatability.
      Training remains random because training env.reset() does not pass spawn_index.
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
        self.eval_tracks = [normalize_track_name(t) for t in eval_tracks]
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.best_model_dir = best_model_dir
        self.deterministic = deterministic

        self.best_mean_reward = -float("inf")
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

    def _fixed_spawn_indices(self, env) -> List[int]:
        """
        Use evenly spaced spawn indices along the centerline.
        Same track + same n_eval_episodes => same eval spawn points every time.
        """
        n_points = int(env.centerline.shape[0])
        if n_points <= 3:
            return [1] * self.n_eval_episodes

        idxs = np.linspace(1, n_points - 2, num=self.n_eval_episodes, dtype=int).tolist()
        return [int(i) for i in idxs]

    def _run_eval(self, track: str) -> EvalResult:
        env = make_env_for_track(self.vehicle_cfg, track, render_mode=None)
        rewards = []
        lengths = []

        try:
            spawn_indices = self._fixed_spawn_indices(env)

            for ep_idx, spawn_idx in enumerate(spawn_indices):
                # Fixed / repeatable eval spawn:
                obs, info = env.reset(
                    seed=1000 + ep_idx,
                    options={"spawn_index": spawn_idx},
                )

                # If you ever want random eval spawns again, use this instead:
                # obs, info = env.reset()

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

        if not self.eval_tracks:
            return True

        results = [self._run_eval(track) for track in self.eval_tracks]

        overall_mean_reward = sum(r.mean_reward for r in results) / len(results)
        overall_mean_len = sum(r.mean_len for r in results) / len(results)

        self.logger.record("eval/heldout_mean_reward", overall_mean_reward)
        self.logger.record("eval/heldout_mean_ep_len", overall_mean_len)

        for res in results:
            track_key = res.track.replace(" ", "_")
            self.logger.record(f"eval/{track_key}_mean_reward", res.mean_reward)
            self.logger.record(f"eval/{track_key}_mean_ep_len", res.mean_len)

        self.logger.dump(self.num_timesteps)

        if overall_mean_reward > self.best_mean_reward:
            self.best_mean_reward = overall_mean_reward
            save_path = self.best_model_dir / "best_model"
            self.model.save(str(save_path))
            if self.verbose:
                print(
                    f"[eval] New best heldout_mean_reward={overall_mean_reward:.3f} "
                    f"-> saved {save_path}"
                )

        if self.verbose:
            print(
                f"[eval] step={self.num_timesteps} "
                f"heldout_mean_reward={overall_mean_reward:.3f} "
                f"heldout_mean_len={overall_mean_len:.1f}"
            )
            for res in results:
                print(
                    f"    track={res.track} "
                    f"mean_reward={res.mean_reward:.3f} "
                    f"mean_len={res.mean_len:.1f}"
                )

        return True


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
    ap.add_argument(
        "--eval_tracks",
        default="",
        help="Optional comma-separated tracks for eval. Overrides sac.yaml heldout_tracks.",
    )
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
        sim_cfg = veh_cfg.get("sim", {})
        raw_map = sim_cfg.get("map_name", "Sakhir_map")
        default_track = normalize_track_name(raw_map)
        schedule = [{
            "track": default_track,
            "steps": int(sac_cfg["train_steps"]),
            "eval_freq": int(sac_cfg["eval_interval_steps"]),
        }]

    schedule_tracks = [normalize_track_name(item["track"]) for item in schedule]

    if args.eval_tracks.strip():
        eval_tracks = [normalize_track_name(t) for t in args.eval_tracks.split(",") if t.strip()]
    else:
        cfg_heldout = [normalize_track_name(t) for t in sac_cfg.get("heldout_tracks", [])]

        if cfg_heldout:
            missing = [t for t in cfg_heldout if t not in available_tracks]
            if missing:
                raise ValueError(
                    f"heldout_tracks contains tracks not found in assets: {missing}\n"
                    f"Available tracks: {available_tracks}"
                )

            overlap = [t for t in cfg_heldout if t in set(schedule_tracks)]
            if overlap:
                raise ValueError(
                    f"heldout_tracks overlaps with training schedule: {overlap}"
                )

            eval_tracks = cfg_heldout
        else:
            eval_tracks = [t for t in available_tracks if t not in set(schedule_tracks)]

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
    print("tensorboard_log:", runs_dir)
    print("checkpoints:", ckpt_root)

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

    for phase_idx, phase in enumerate(schedule):
        track = normalize_track_name(phase["track"])
        steps = int(phase["steps"])
        eval_freq = int(phase.get("eval_freq", sac_cfg.get("eval_interval_steps", 5000)))

        if phase_idx != 0:
            try:
                env.close()
            except Exception:
                pass
            env = make_env_for_track(veh_cfg, track, render_mode=None)
            model.set_env(env)

        phase_dir = ckpt_root / f"{phase_idx:02d}_{track}"
        phase_dir.mkdir(parents=True, exist_ok=True)

        callbacks = []

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

        # New behavior: evaluate ALL held-out tracks with fixed spawns
        callbacks.append(
            HeldoutMapsEvalCallback(
                vehicle_cfg=veh_cfg,
                eval_tracks=eval_tracks,
                eval_freq=eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                best_model_dir=phase_dir,
                deterministic=True,
                verbose=1,
            )
        )

        # Old behavior: evaluate ONE random held-out track with random spawns
        # Uncomment this block and comment out the block above if you want to go back.
        #
        # callbacks.append(
        #     RandomMapEvalCallback(
        #         vehicle_cfg=veh_cfg,
        #         eval_tracks=eval_tracks,
        #         eval_freq=eval_freq,
        #         n_eval_episodes=args.n_eval_episodes,
        #         best_model_dir=phase_dir,
        #         deterministic=True,
        #         verbose=1,
        #     )
        # )

        print(f"\n=== Phase {phase_idx} ===")
        print("track:", track)
        print("steps:", steps, "eval_freq:", eval_freq)

        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            tb_log_name=run_id,
            callback=callbacks,
            progress_bar=True,
        )

        model.save(str(phase_dir / "model_phase_end"))

    final_path = ckpt_root / "sac_final"
    model.save(str(final_path))
    print("\nSaved final model:", final_path)

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()