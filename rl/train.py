#!/usr/bin/env python3
"""
SAC training script for akmSteerSim.

Run:
  python rl/train.py --vehicle_cfg configs/vehicle.yaml --sac_cfg configs/sac.yaml

Resume from checkpoint:
  python rl/train.py --resume checkpoints/<run_id>/sac_final.zip
"""

import argparse
import json
import time
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

from rl.common import (
    normalize_track_name,
    make_env_for_track,
    make_lr_schedule,
    arc_length_spawn_indices,
    EpisodeResult,
    run_eval_episode,
    log_episode_metrics,
)


# ----------------------------
# Early stopping callback
# ----------------------------

class EarlyStoppingOnPlateau(BaseCallback):
    """
    Stop training if eval mean reward has not improved for `patience`
    consecutive eval windows.

    Works by reading from the logger — attach *after* the eval callback
    in the callback list so that eval metrics are logged first.
    """
    def __init__(self, eval_freq: int, patience: int = 10, min_delta: float = 0.0, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.eval_freq = int(eval_freq)
        self.patience = patience
        self.min_delta = min_delta
        self._best_reward = -float("inf")
        self._no_improve_count = 0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or (self.num_timesteps % self.eval_freq) != 0:
            return True

        current_reward = self.logger.name_to_value.get("eval/mean_reward", None)
        if current_reward is None:
            return True

        if current_reward > self._best_reward + self.min_delta:
            self._best_reward = current_reward
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        if self.verbose and self._no_improve_count > 0:
            print(
                f"[early_stop] No improvement for {self._no_improve_count}/{self.patience} "
                f"eval windows (best={self._best_reward:.3f}, current={current_reward:.3f})"
            )

        if self._no_improve_count >= self.patience:
            if self.verbose:
                print(
                    f"[early_stop] Stopping training at step {self.num_timesteps}: "
                    f"no improvement for {self.patience} consecutive eval windows."
                )
            return False

        return True


# ----------------------------
# Eval callback
# ----------------------------

class HeldoutMapsEvalCallback(BaseCallback):
    """
    Every eval_freq steps, evaluate on all eval tracks with fixed spawns.
    Logs all paper-relevant metrics (with std) to TensorBoard and saves
    JSON results.

    Eval environments are created once and reused across eval calls.
    Spawn indices are computed via arc-length normalization for
    consistent spatial coverage across tracks with different centerline
    resolutions.
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

        self._envs: Dict[str, Any] = {}
        self._spawn_indices: Dict[str, List[int]] = {}
        self._train_start_time: float = time.time()

    def _get_env(self, track: str):
        if track not in self._envs:
            self._envs[track] = make_env_for_track(self.vehicle_cfg, track, render_mode=None)
        return self._envs[track]

    def _get_spawn_indices(self, track: str, env) -> List[int]:
        if track not in self._spawn_indices:
            self._spawn_indices[track] = arc_length_spawn_indices(
                env.centerline, self.n_eval_episodes
            )
        return self._spawn_indices[track]

    def _eval_track(self, track: str) -> List[EpisodeResult]:
        env = self._get_env(track)
        spawn_indices = self._get_spawn_indices(track, env)

        results = []
        for ep_idx, spawn_idx in enumerate(spawn_indices):
            result = run_eval_episode(
                self.model, env,
                seed=1000 + ep_idx,
                spawn_idx=spawn_idx,
                deterministic=self.deterministic,
            )
            results.append(result)
        return results

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or (self.num_timesteps % self.eval_freq) != 0:
            return True
        if not self.eval_tracks:
            return True

        eval_start = time.time()

        all_results = {}
        for track in self.eval_tracks:
            all_results[track] = self._eval_track(track)

        # --- per-track and overall logging ---
        all_episodes = []
        for track, episodes in all_results.items():
            all_episodes.extend(episodes)
            tk = track.replace(" ", "_")
            log_episode_metrics(self.logger, f"eval/{tk}", episodes)

        log_episode_metrics(self.logger, "eval", all_episodes)

        # Wall-clock timing
        elapsed_train = time.time() - self._train_start_time
        elapsed_eval = time.time() - eval_start
        self.logger.record("time/wall_clock_hours", elapsed_train / 3600.0)
        self.logger.record("time/eval_seconds", elapsed_eval)

        self.logger.dump(self.num_timesteps)

        # JSON snapshot
        snapshot = {
            "timestep": self.num_timesteps,
            "wall_clock_hours": elapsed_train / 3600.0,
            "tracks": {},
        }
        for track, episodes in all_results.items():
            snapshot["tracks"][track] = [vars(e) for e in episodes]

        json_path = self.results_dir / f"eval_{self.num_timesteps:09d}.json"
        json_path.write_text(json.dumps(snapshot, indent=2))

        # Best model
        mean_reward = np.mean([e.reward for e in all_episodes]) if all_episodes else -float("inf")
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(str(self.results_dir / "best_model"))
            if self.verbose:
                print(f"[eval] New best mean_reward={mean_reward:.3f} at step {self.num_timesteps}")

        if self.verbose:
            n_total = max(1, len(all_episodes))
            print(
                f"[eval] step={self.num_timesteps} "
                f"reward={mean_reward:.3f} "
                f"progress={np.mean([e.normalized_progress for e in all_episodes]):.3f} "
                f"crash_rate={sum(1 for e in all_episodes if e.term_reason == 'crash') / n_total:.2f} "
                f"lat_err={np.mean([e.mean_lateral_error for e in all_episodes]):.4f} "
                f"steer_tv/step={np.mean([e.steer_tv_per_step for e in all_episodes]):.4f} "
                f"({elapsed_eval:.1f}s)"
            )

        return True

    def _on_training_end(self) -> None:
        for env in self._envs.values():
            try:
                env.close()
            except Exception:
                pass
        self._envs.clear()


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
    ap.add_argument("--early_stop_patience", type=int, default=None,
                     help="Stop after N eval windows with no improvement (default: from config)")
    ap.add_argument("--save_replay_buffer", action="store_true",
                     help="Save replay buffer at checkpoints (large files)")
    ap.add_argument("--resume", default=None,
                     help="Path to a saved model .zip to resume training from")
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

    # --- training track ---
    if args.train_track:
        train_track = normalize_track_name(args.train_track)
    else:
        configured = sac_cfg.get("train_track", None)
        if configured:
            train_track = normalize_track_name(configured)
        else:
            raw_map = veh_cfg.get("sim", {}).get("map_name", "Sakhir_map")
            train_track = normalize_track_name(raw_map)

    # --- eval tracks (always include training track for train/generalization comparison) ---
    if args.eval_tracks:
        eval_tracks = [normalize_track_name(t) for t in args.eval_tracks.split(",") if t.strip()]
    else:
        eval_tracks = [normalize_track_name(t) for t in sac_cfg.get("heldout_tracks", [])]
    if train_track not in eval_tracks:
        eval_tracks.insert(0, train_track)

    # --- paths ---
    ablate_tag = "_ablated" if args.ablate_geometry else ""
    run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}_{action_space_name}{ablate_tag}_s{seed}"
    runs_dir = ROOT / "runs"
    ckpt_dir = ROOT / "checkpoints" / run_id
    results_dir = ckpt_dir / "eval_results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- env ---
    env = make_env_for_track(veh_cfg, train_track, render_mode=None)

    train_steps = int(sac_cfg.get("train_steps", 500_000))
    eval_freq = int(sac_cfg.get("eval_interval_steps", 5000))
    save_every = int(sac_cfg.get("save_every_steps", 25000))
    lr_schedule = make_lr_schedule(sac_cfg)
    target_entropy = sac_cfg.get("target_entropy", "auto")

    # --- model (new or resumed) ---
    resuming = args.resume is not None
    if resuming:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume model not found: {resume_path}")
        model = SAC.load(
            str(resume_path),
            env=env,
            device=args.device,
            tensorboard_log=str(runs_dir),
        )
        print(f"  Resumed from: {resume_path}")
    else:
        model = SAC(
            policy=sac_cfg["policy"],
            env=env,
            learning_rate=lr_schedule,
            batch_size=sac_cfg["batch_size"],
            tau=sac_cfg["tau"],
            gamma=sac_cfg["gamma"],
            buffer_size=int(sac_cfg.get("buffer_size", 1_000_000)),
            learning_starts=int(sac_cfg.get("learning_starts", 100)),
            seed=seed,
            policy_kwargs=sac_cfg.get("policy_kwargs", None),
            verbose=1,
            tensorboard_log=str(runs_dir),
            device=args.device,
            ent_coef=sac_cfg.get("ent_coef", "auto"),
            target_entropy=target_entropy,
        )

    # --- save run metadata ---
    run_meta = {
        "run_id": run_id,
        "seed": seed,
        "action_space": action_space_name,
        "ablate_geometry": args.ablate_geometry,
        "train_track": train_track,
        "eval_tracks": eval_tracks,
        "n_eval_episodes": args.n_eval_episodes,
        "resumed_from": args.resume,
        "obs_space": str(env.observation_space),
        "act_space": str(env.action_space),
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
    print(f"  obs_space:     {env.observation_space}")
    print(f"  act_space:     {env.action_space}")
    if hasattr(model, "target_entropy"):
        print(f"  target_entropy: {model.target_entropy}")
    print(f"  lr_schedule:   {type(lr_schedule).__name__ if callable(lr_schedule) else lr_schedule}")
    print(f"  checkpoints:   {ckpt_dir}")

    # --- callbacks ---
    callbacks = [
        CheckpointCallback(
            save_freq=save_every,
            save_path=str(ckpt_dir),
            name_prefix="sac",
            save_replay_buffer=args.save_replay_buffer,
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

    early_stop_patience = args.early_stop_patience or sac_cfg.get("early_stop_patience", None)
    if early_stop_patience is not None:
        callbacks.append(
            EarlyStoppingOnPlateau(
                eval_freq=eval_freq,
                patience=int(early_stop_patience),
                min_delta=float(sac_cfg.get("early_stop_min_delta", 0.0)),
                verbose=1,
            )
        )

    # --- train ---
    model.learn(
        total_timesteps=train_steps,
        reset_num_timesteps=not resuming,
        tb_log_name=run_id,
        callback=callbacks,
        progress_bar=True,
    )

    # --- save final model ---
    model.save(str(ckpt_dir / "sac_final"))
    if args.save_replay_buffer:
        model.save_replay_buffer(str(ckpt_dir / "sac_final_replay_buffer"))
    print(f"\nSaved final model: {ckpt_dir / 'sac_final'}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()