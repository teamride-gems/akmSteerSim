#!/usr/bin/env python3
"""
SAC training script for akmSteerSim.

Run:
  python rl/train.py --vehicle_cfg configs/vehicle.yaml --sac_cfg configs/sac.yaml
"""

import argparse
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
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


# ----------------------------
# Helpers
# ----------------------------

def list_tracks(assets_dir: Path) -> List[str]:
    if not assets_dir.exists():
        return []
    return sorted([p.name for p in assets_dir.iterdir() if p.is_dir()])


def normalize_track_name(track: str) -> str:
    return str(track).replace("_map", "").strip()


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
    cfg["sim"]["map_name"] = f"{track}_map"
    cfg["sim"]["map_dir"] = str(track_dir)
    cfg["sim"]["track_name"] = track

    return F1TenthSACEnv(
        vehicle_cfg=cfg,
        track_centerline_csv=str(cl),
        render_mode=render_mode,
    )


# ----------------------------
# Episode result container
# ----------------------------

@dataclass
class EpisodeResult:
    reward: float = 0.0
    length: int = 0
    term_reason: str = "unknown"
    normalized_progress: float = 0.0
    mean_lateral_error: float = 0.0
    max_lateral_error: float = 0.0
    mean_heading_error: float = 0.0
    mean_speed: float = 0.0
    mean_abs_steer_rate: float = 0.0
    steer_tv: float = 0.0  # total variation of steering
    steer_clip_frac: float = 0.0
    speed_clip_frac: float = 0.0
    mean_steer_clip_mag: float = 0.0
    mean_speed_clip_mag: float = 0.0
    min_lidar: float = 0.0


def run_eval_episode(model, env, seed: int, spawn_idx: int, deterministic: bool = True) -> EpisodeResult:
    """Run one evaluation episode and collect all paper-relevant metrics."""
    obs, info = env.reset(seed=seed, options={"spawn_index": spawn_idx})

    # accumulators
    ep_reward = 0.0
    ep_len = 0
    lat_errors = []
    head_errors = []
    speeds = []
    abs_steer_rates = []
    steer_cmds = []
    min_lidars = []
    steer_clips = 0
    speed_clips = 0
    steer_clip_mags = []
    speed_clip_mags = []

    done = False
    last_info = info

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        ep_reward += float(reward)
        ep_len += 1

        lat_errors.append(abs(float(info.get("lateral_error", 0.0))))
        head_errors.append(abs(float(info.get("heading_error", 0.0))))
        speeds.append(float(info.get("speed", 0.0)))
        abs_steer_rates.append(abs(float(info.get("steer_rate", 0.0))))
        steer_cmds.append(float(info.get("steer_cmd", 0.0)))
        min_lidars.append(float(info.get("min_lidar", 10.0)))

        if info.get("steer_clipped", False):
            steer_clips += 1
        if info.get("speed_clipped", False):
            speed_clips += 1
        steer_clip_mags.append(float(info.get("steer_clip_mag", 0.0)))
        speed_clip_mags.append(float(info.get("speed_clip_mag", 0.0)))

        last_info = info
        done = bool(terminated or truncated)

    # steering total variation
    steer_arr = np.array(steer_cmds)
    steer_tv = float(np.sum(np.abs(np.diff(steer_arr)))) if len(steer_arr) > 1 else 0.0

    n = max(1, ep_len)
    return EpisodeResult(
        reward=ep_reward,
        length=ep_len,
        term_reason=last_info.get("term_reason", "unknown"),
        normalized_progress=float(last_info.get("normalized_progress", 0.0)),
        mean_lateral_error=float(np.mean(lat_errors)) if lat_errors else 0.0,
        max_lateral_error=float(np.max(lat_errors)) if lat_errors else 0.0,
        mean_heading_error=float(np.mean(head_errors)) if head_errors else 0.0,
        mean_speed=float(np.mean(speeds)) if speeds else 0.0,
        mean_abs_steer_rate=float(np.mean(abs_steer_rates)) if abs_steer_rates else 0.0,
        steer_tv=steer_tv,
        steer_clip_frac=steer_clips / n,
        speed_clip_frac=speed_clips / n,
        mean_steer_clip_mag=float(np.mean(steer_clip_mags)) if steer_clip_mags else 0.0,
        mean_speed_clip_mag=float(np.mean(speed_clip_mags)) if speed_clip_mags else 0.0,
        min_lidar=float(np.min(min_lidars)) if min_lidars else 0.0,
    )


# ----------------------------
# Eval callback with full metric collection
# ----------------------------

class HeldoutMapsEvalCallback(BaseCallback):
    """
    Every eval_freq steps, evaluate on all held-out tracks with fixed spawns.
    Logs all paper-relevant metrics to TensorBoard and saves JSON results.
    """
    def __init__(
        self,
        vehicle_cfg: Dict[str, Any],
        eval_tracks: List[str],
        eval_freq: int,
        n_eval_episodes: int,
        results_dir: Path,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.vehicle_cfg = vehicle_cfg
        self.eval_tracks = [normalize_track_name(t) for t in eval_tracks]
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.results_dir = Path(results_dir)
        self.deterministic = deterministic

        self.best_mean_reward = -float("inf")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _fixed_spawn_indices(self, env) -> List[int]:
        n_points = int(env.centerline.shape[0])
        if n_points <= 3:
            return [1] * self.n_eval_episodes
        return np.linspace(1, n_points - 2, num=self.n_eval_episodes, dtype=int).tolist()

    def _eval_track(self, track: str) -> List[EpisodeResult]:
        env = make_env_for_track(self.vehicle_cfg, track, render_mode=None)
        results = []
        try:
            spawn_indices = self._fixed_spawn_indices(env)
            for ep_idx, spawn_idx in enumerate(spawn_indices):
                result = run_eval_episode(
                    self.model, env,
                    seed=1000 + ep_idx,
                    spawn_idx=spawn_idx,
                    deterministic=self.deterministic,
                )
                results.append(result)
        finally:
            try:
                env.close()
            except Exception:
                pass
        return results

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or (self.num_timesteps % self.eval_freq) != 0:
            return True
        if not self.eval_tracks:
            return True

        all_results = {}
        for track in self.eval_tracks:
            all_results[track] = self._eval_track(track)

        # --- aggregate and log ---
        all_episodes = []
        for track, episodes in all_results.items():
            all_episodes.extend(episodes)
            n = len(episodes)
            if n == 0:
                continue

            tk = track.replace(" ", "_")

            # per-track metrics
            self.logger.record(f"eval/{tk}/mean_reward", np.mean([e.reward for e in episodes]))
            self.logger.record(f"eval/{tk}/mean_progress", np.mean([e.normalized_progress for e in episodes]))
            self.logger.record(f"eval/{tk}/crash_rate", sum(1 for e in episodes if e.term_reason == "crash") / n)
            self.logger.record(f"eval/{tk}/completion_rate", sum(1 for e in episodes if e.normalized_progress >= 0.95) / n)
            self.logger.record(f"eval/{tk}/mean_lateral_error", np.mean([e.mean_lateral_error for e in episodes]))
            self.logger.record(f"eval/{tk}/mean_steer_rate", np.mean([e.mean_abs_steer_rate for e in episodes]))
            self.logger.record(f"eval/{tk}/steer_clip_frac", np.mean([e.steer_clip_frac for e in episodes]))

        # overall metrics
        n_total = len(all_episodes)
        if n_total > 0:
            self.logger.record("eval/mean_reward", np.mean([e.reward for e in all_episodes]))
            self.logger.record("eval/mean_progress", np.mean([e.normalized_progress for e in all_episodes]))
            self.logger.record("eval/crash_rate", sum(1 for e in all_episodes if e.term_reason == "crash") / n_total)
            self.logger.record("eval/completion_rate", sum(1 for e in all_episodes if e.normalized_progress >= 0.95) / n_total)
            self.logger.record("eval/mean_lateral_error", np.mean([e.mean_lateral_error for e in all_episodes]))
            self.logger.record("eval/mean_heading_error", np.mean([e.mean_heading_error for e in all_episodes]))
            self.logger.record("eval/mean_speed", np.mean([e.mean_speed for e in all_episodes]))
            self.logger.record("eval/mean_steer_rate", np.mean([e.mean_abs_steer_rate for e in all_episodes]))
            self.logger.record("eval/mean_steer_tv", np.mean([e.steer_tv for e in all_episodes]))
            self.logger.record("eval/steer_clip_frac", np.mean([e.steer_clip_frac for e in all_episodes]))
            self.logger.record("eval/speed_clip_frac", np.mean([e.speed_clip_frac for e in all_episodes]))
            self.logger.record("eval/mean_ep_len", np.mean([e.length for e in all_episodes]))

        self.logger.dump(self.num_timesteps)

        # save JSON snapshot
        snapshot = {
            "timestep": self.num_timesteps,
            "tracks": {},
        }
        for track, episodes in all_results.items():
            snapshot["tracks"][track] = [vars(e) for e in episodes]

        json_path = self.results_dir / f"eval_{self.num_timesteps:09d}.json"
        json_path.write_text(json.dumps(snapshot, indent=2))

        # best model
        mean_reward = np.mean([e.reward for e in all_episodes]) if all_episodes else -float("inf")
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(str(self.results_dir / "best_model"))
            if self.verbose:
                print(f"[eval] New best mean_reward={mean_reward:.3f} at step {self.num_timesteps}")

        if self.verbose:
            print(
                f"[eval] step={self.num_timesteps} "
                f"reward={mean_reward:.3f} "
                f"progress={np.mean([e.normalized_progress for e in all_episodes]):.3f} "
                f"crash_rate={sum(1 for e in all_episodes if e.term_reason == 'crash') / max(1, n_total):.2f} "
                f"lat_err={np.mean([e.mean_lateral_error for e in all_episodes]):.4f} "
                f"steer_rate={np.mean([e.mean_abs_steer_rate for e in all_episodes]):.4f}"
            )

        return True


# ----------------------------
# Main
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
    ap.add_argument("--seed", type=int, default=None, help="Override seed from config")
    ap.add_argument("--action_space", default=None, help="Override action space from vehicle config")
    ap.add_argument("--ablate_geometry", action="store_true", help="Zero out e_head/e_lat in observations")
    ap.add_argument("--train_track", default=None, help="Override training track")
    ap.add_argument("--eval_tracks", default=None, help="Comma-separated eval tracks")
    ap.add_argument("--n_eval_episodes", type=int, default=10)
    ap.add_argument("--run_id", default=None, help="Custom run ID (default: auto timestamp)")
    args = ap.parse_args()

    veh_cfg = load_yaml(ROOT / args.vehicle_cfg)
    sac_cfg = load_yaml(ROOT / args.sac_cfg)

    # --- CLI overrides ---
    seed = args.seed if args.seed is not None else int(sac_cfg.get("seed", 0))
    set_random_seed(seed)

    if args.action_space:
        veh_cfg["action_space"] = args.action_space

    if args.ablate_geometry:
        veh_cfg["ablate_centerline_features"] = True

    action_space_name = veh_cfg.get("action_space", "steer_speed")

    # --- tracks ---
    if args.train_track:
        train_track = normalize_track_name(args.train_track)
    else:
        schedule = sac_cfg.get("map_schedule", None)
        if schedule:
            train_track = normalize_track_name(schedule[0]["track"])
        else:
            raw_map = veh_cfg.get("sim", {}).get("map_name", "Sakhir_map")
            train_track = normalize_track_name(raw_map)

    if args.eval_tracks:
        eval_tracks = [normalize_track_name(t) for t in args.eval_tracks.split(",") if t.strip()]
    else:
        eval_tracks = [normalize_track_name(t) for t in sac_cfg.get("heldout_tracks", [])]
        if not eval_tracks:
            eval_tracks = [train_track]

    # --- paths ---
    ablate_tag = "_ablated" if args.ablate_geometry else ""
    run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}_{action_space_name}{ablate_tag}_s{seed}"
    runs_dir = ROOT / "runs"
    ckpt_dir = ROOT / "checkpoints" / run_id
    results_dir = ckpt_dir / "eval_results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # save full config for reproducibility
    run_meta = {
        "run_id": run_id,
        "seed": seed,
        "action_space": action_space_name,
        "ablate_geometry": args.ablate_geometry,
        "train_track": train_track,
        "eval_tracks": eval_tracks,
        "vehicle_cfg": veh_cfg,
        "sac_cfg": sac_cfg,
    }
    (ckpt_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str))

    print("=== Training setup ===")
    print(f"  run_id:        {run_id}")
    print(f"  seed:          {seed}")
    print(f"  action_space:  {action_space_name}")
    print(f"  ablate_geom:   {args.ablate_geometry}")
    print(f"  train_track:   {train_track}")
    print(f"  eval_tracks:   {eval_tracks}")
    print(f"  checkpoints:   {ckpt_dir}")

    # --- env + model ---
    env = make_env_for_track(veh_cfg, train_track, render_mode=None)

    train_steps = int(sac_cfg.get("train_steps", 500_000))
    eval_freq = int(sac_cfg.get("eval_interval_steps", 5000))
    save_every = int(sac_cfg.get("save_every_steps", 25000))

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

    callbacks = [
        CheckpointCallback(
            save_freq=save_every,
            save_path=str(ckpt_dir),
            name_prefix="sac",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
        HeldoutMapsEvalCallback(
            vehicle_cfg=veh_cfg,
            eval_tracks=eval_tracks,
            eval_freq=eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            results_dir=results_dir,
            deterministic=True,
            verbose=1,
        ),
    ]

    model.learn(
        total_timesteps=train_steps,
        reset_num_timesteps=True,
        tb_log_name=run_id,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(str(ckpt_dir / "sac_final"))
    print(f"\nSaved final model: {ckpt_dir / 'sac_final'}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()