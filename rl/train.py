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
import pandas as pd
import math
# idk if these needed but just in case
import json
import hashlib
# also python mp is FAKE
import multiprocessing as mp


import yaml

# --- make repo root importable (fixes: ModuleNotFoundError: envs) ---
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from envs.f1tenth_sb3_env import F1TenthSACEnv
from scripts.random_pose_gen import deterministic_hash, generate_start_poses_from_centerline
# can replace parallel w singular if buggy
from scripts.run_eval import _run_single_episode, run_episodes_parallel, aggregate_results_list


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


# create a new validation callback
class ValidationCallback(BaseCallback):
    """
    does one validation cycle every eval_freq steps
    goes through all maps
    uses same start poses throughout entire validation cycle
    """
    def __init__(self,
                 vehicle_cfg: Dict[str,Any],
                 eval_config: Dict[str,Any],
                 eval_freq: int,
                 ckpt_dir: Path,
                 runs_root: Path,
                 verbose: int = 0):
        super().__init__(verbose=verbose)
        self.vehicle_cfg = vehicle_cfg
        self.eval_cfg = eval_config or {}
        self.eval_freq = int(eval_freq)
        self.ckpt_dir = Path(ckpt_dir)
        self.runs_root = Path(runs_root)
        self.master_seed = int(self.eval_cfg.get("master_seed", 123456))
        self.val_pool = [str(m) for m in self.eval_cfg.get("val_pool", [])]
        self.maps_per_cycle = int(self.eval_cfg.get("val_maps_per_cycle", 3))
        self.attempts_per_map = int(self.eval_cfg.get("val_attempts_per_map", 10))
        self.val_seeds = list(self.eval_cfg.get("val_seeds", [0,1,2]))
        self.deterministic = bool(self.eval_cfg.get("val_deterministic", True))
        self.timeout_per_meter = float(self.eval_cfg.get("val_timeout_per_meter", 0.5))
        self.min_spacing_frac = float(self.eval_cfg.get("val_min_spacing_frac", 0.02))
        self.yaw_jitter_deg = float(self.eval_cfg.get("val_yaw_jitter_deg", 5.0))
        self.lateral_jitter_m = float(self.eval_cfg.get("val_lateral_jitter_m", 0.05))
        self.workers = int(self.eval_cfg.get("val_workers", 4))

        # save good stats
        self.best_score = -1e9
        self.cycle_count = 0

        # create validation root dir
        self.val_root = self.runs_root / "validation"
        self.val_root.mkdir(parents=True, exist_ok=True)
    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        # trigger only on eval frequency
        if (self.num_timesteps % self.eval_freq) != 0:
            return True
        # compute cycle index deterministically
        cycle_index = self.num_timesteps // self.eval_freq
        self._run_cycle(cycle_index)
        self.cycle_count += 1
        return True
    # compute approximate timeout: track length * timeout_per_meter
    def _estimate_timeout_for_map(self, map_name: str):
        cl_csv = resolve_centerline_csv(self.vehicle_cfg, map_name)
        cdf = pd.read_csv(cl_csv)
        xs = cdf.iloc[:,0].to_numpy(); ys = cdf.iloc[:,1].to_numpy()
        ds = ((xs[1:] - xs[:-1])**2 + (ys[1:] - ys[:-1])**2)**0.5
        track_len = float(ds.sum())
        return float(track_len * self.timeout_per_meter + 5.0)

    # actual cycle code
    def _run_cycle(self, cycle_index: int):
        # deterministic RNG seed for this cycle
        cycle_seed = deterministic_hash(self.master_seed, cycle_index)

        # We should use ALL Maps, but in case we want smaller amount we allow random sampling
        rng = random.Random(cycle_seed)
        maps = list(self.val_pool)
        if len(maps) == 0:
            return
        rng.shuffle(maps)
        selected = maps[:self.maps_per_cycle] if len(maps) >= self.maps_per_cycle else maps
        cycle_dir = self.val_root / f"cycles/cycle_{cycle_index:05d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # save selected maps
        (cycle_dir / "maps.json").write_text(json.dumps(selected))

        # For each map, generate poses and save them
        per_map_pose_files = {}
        for m in selected:
            # find centerline csv
            cl_csv = resolve_centerline_csv(self.vehicle_cfg, m)

            # generate N poses total for all seeds: attempts_per_map * len(val_seeds)
            n_total = self.attempts_per_map * len(self.val_seeds)
            pose_seed = deterministic_hash(cycle_seed, m)
            poses = generate_start_poses_from_centerline(
                centerline_csv=str(cl_csv),
                n_poses=n_total,
                seed=pose_seed,
                min_spacing_frac=self.min_spacing_frac,
                yaw_jitter_rad=math.radians(self.yaw_jitter_deg),
                lateral_jitter_m=self.lateral_jitter_m
            )

            # Save as CSV
            df = pd.DataFrame(poses)
            pose_csv = cycle_dir / f"{m}_start_poses.csv"
            df.to_csv(pose_csv, index=False)
            per_map_pose_files[m] = str(pose_csv)
        # Now run evaluation for each policy & seed combination
        # For validation we evaluate the current model checkpoint saved by training.
        # Save current model to a temporary path and evaluate
        timestamp = int(time.time())
        ckpt_name = f"val_ckpt_step_{self.num_timesteps}"
        ckpt_path = str(self.ckpt_dir / ckpt_name)
        # save model
        self.model.save(ckpt_path)
        # For reproducibility, write metadata
        meta = {
            "cycle_index": cycle_index,
            "cycle_seed": cycle_seed,
            "maps": selected,
            "poses": per_map_pose_files,
            "num_timesteps": self.num_timesteps,
            "ckpt_path": ckpt_path
        }
        (cycle_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        # now run for each map: for simplicity run sequentially per map but episodes per map in parallel
        all_map_summaries = {}
        for m in selected:
            pose_csv = per_map_pose_files[m]
            df = pd.read_csv(pose_csv)
            # split poses for each seed deterministically
            # we assign contiguous blocks of attempts per seed
            n = len(self.val_seeds)
            per_seed = self.attempts_per_map
            map_results = []
            timeout_s = self._estimate_timeout_for_map(m)
            # run per-seed blocks
            for i, seed_val in enumerate(self.val_seeds):
                start_idx = i * per_seed
                end_idx = start_idx + per_seed
                poses_block = df.iloc[start_idx:end_idx].to_dict(orient="records")
                # run episodes in parallel
                results = run_episodes_parallel(
                    ckpt_path=f"{ckpt_path}.zip" if not str(ckpt_path).endswith(".zip") and not str(ckpt_path).endswith(".pkl") else ckpt_path,
                    vehicle_cfg=self.vehicle_cfg,
                    track=m,
                    poses=poses_block,
                    deterministic=self.deterministic,
                    timeout_s=timeout_s,
                    workers=self.workers
                )
                map_results.extend(results)
            # aggregate and save per-map results
            agg = aggregate_results_list(map_results)
            all_map_summaries[m] = {"aggregate": agg, "raw": map_results}
            # save raw
            import json
            (cycle_dir / f"{m}_raw.json").write_text(json.dumps(map_results, indent=2))
            (cycle_dir / f"{m}_summary.json").write_text(json.dumps(agg, indent=2))
        # compute combined validation score across maps (weighted average equal)
        scores = []
        for m in selected:
            agg = all_map_summaries[m]["aggregate"]
            # create composite score (configurable weights)
            w = self.eval_cfg.get("selection_weights", {"success_rate":1.0, "feasibility_violations":-0.5, "normalized_time":-0.2})
            success = float(agg.get("success_rate", 0.0))
            fv = float(agg.get("mean_feasibility_violations", 0.0))
            # normalized time: use mean_duration_s / track_length (we compute track_length)
            from rl.train import resolve_centerline_csv
            cl_csv = resolve_centerline_csv(self.vehicle_cfg, m)
            # quick centerline length:
            cdf = pd.read_csv(cl_csv)
            # approximate length by sum of distances
            xs = cdf.iloc[:,0].to_numpy(); ys = cdf.iloc[:,1].to_numpy()
            ds = ((xs[1:] - xs[:-1])**2 + (ys[1:] - ys[:-1])**2)**0.5
            track_len = float(ds.sum())
            mean_dur = float(agg.get("mean_duration_s", 0.0))
            norm_time = mean_dur / max(1e-6, track_len)
            score = (w.get("success_rate",1.0) * success) + (w.get("feasibility_violations",-0.5) * fv) + (w.get("normalized_time",-0.2) * norm_time)
            scores.append(score)
        val_score = float(sum(scores) / len(scores))
        # save combined summary
        (cycle_dir / "combined_summary.json").write_text(json.dumps({"val_score": val_score}, indent=2))
        # if better than best_score, save this ckpt as best
        if val_score > self.best_score:
            self.best_score = val_score
            best_path = self.ckpt_dir / "best_validation_model"
            self.model.save(str(best_path))
            (cycle_dir / "best_marker.txt").write_text("best")
            if self.verbose:
                print(f"[Validation] New best score {val_score:.4f} at step {self.num_timesteps}, saved {best_path}")
    
"""

class RandomMapEvalCallback(BaseCallback):
    
    #Every eval_freq steps:
    #  - pick a random track from eval_tracks
    #  - create a fresh env for that track
    #  - run n_eval_episodes with deterministic policy
    #  - log eval/mean_reward, eval/mean_ep_len, eval/track_name
    #  - optionally save best model
   #
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

"""
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
    eval_cfg = sac_cfg.get("evaluation", {})

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
    # currently set up so eval tracks kinda useless since our validation uses set in yaml
    if args.eval_tracks.strip():
        eval_tracks = [normalize_track_name(t) for t in args.eval_tracks.split(",") if t.strip()]
    else:
        eval_tracks = [t for t in eval_cfg.get("val_pool", []) if t not in set(schedule_tracks)]

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

        # upgraded valididation map eval
        callbacks.append(
            ValidationCallback(
                vehicle_cfg=veh_cfg,
                eval_config=eval_cfg,
                eval_freq=eval_freq,
                ckpt_dir=phase_dir,
                runs_root=runs_dir,
                verbose=1,
            )
        )


        # # random-map eval
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
