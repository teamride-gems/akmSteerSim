#!/usr/bin/env python3
"""
Patched SAC training script with clean validation/test separation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.action_spaces_utils import get_policy_dim
from rl.common import (
    normalize_track_name,
    make_env_for_track,
    make_lr_schedule,
    arc_length_spawn_indices,
    run_eval_episode,
    log_episode_metrics,
    summarize_episodes,
    model_selection_score,
)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(path.read_text())


def _csv_to_tracks(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    tracks = [normalize_track_name(t) for t in raw.split(",") if t.strip()]
    return tracks or []


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def resolve_target_entropy(sac_cfg: Dict[str, Any], action_space_name: str):
    te = sac_cfg.get("target_entropy", "auto")
    if te in (None, "auto", "auto_dim"):
        return "auto"
    if isinstance(te, str):
        return float(te)
    return te


class EarlyStoppingOnMetric(BaseCallback):
    def __init__(self, metric_name: str, eval_freq: int, patience: int, min_delta: float = 0.0, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.metric_name = metric_name
        self.eval_freq = int(eval_freq)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self._best_value = -float("inf")
        self._no_improve_count = 0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or (self.num_timesteps % self.eval_freq) != 0:
            return True

        current_value = self.logger.name_to_value.get(self.metric_name, None)
        if current_value is None:
            return True

        if current_value > self._best_value + self.min_delta:
            self._best_value = float(current_value)
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        if self.verbose and self._no_improve_count > 0:
            print(
                f"[early_stop] No improvement for {self._no_improve_count}/{self.patience} "
                f"windows on {self.metric_name} (best={self._best_value:.6f}, current={float(current_value):.6f})"
            )

        if self._no_improve_count >= self.patience:
            if self.verbose:
                print(f"[early_stop] Stopping at step {self.num_timesteps}.")
            return False
        return True


class ValidationEvalCallback(BaseCallback):
    """
    Evaluates:
      - validation tracks: for model selection / early stopping
      - optionally the train track: for diagnostics only
    Never evaluates the test split during training.
    """

    def __init__(
        self,
        vehicle_cfg: Dict[str, Any],
        train_track: Optional[str],
        validation_tracks: List[str],
        eval_freq: int,
        n_eval_episodes: int,
        results_dir: Path,
        report_train_track: bool = True,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.vehicle_cfg = vehicle_cfg
        self.train_track = normalize_track_name(train_track) if train_track else None
        self.validation_tracks = [normalize_track_name(t) for t in validation_tracks]
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.results_dir = Path(results_dir)
        self.report_train_track = bool(report_train_track and self.train_track is not None)
        self.deterministic = deterministic

        self.best_validation_score = -float("inf")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._envs: Dict[str, Any] = {}
        self._spawn_indices: Dict[str, List[int]] = {}
        self._train_start_time = time.time()

    def _get_env(self, track: str):
        if track not in self._envs:
            self._envs[track] = make_env_for_track(self.vehicle_cfg, track, render_mode=None)
        return self._envs[track]

    def _get_spawn_indices(self, track: str, env) -> List[int]:
        if track not in self._spawn_indices:
            self._spawn_indices[track] = arc_length_spawn_indices(env.centerline, self.n_eval_episodes)
        return self._spawn_indices[track]

    def _eval_track(self, track: str):
        env = self._get_env(track)
        spawn_indices = self._get_spawn_indices(track, env)
        results = []
        for ep_idx, spawn_idx in enumerate(spawn_indices):
            results.append(
                run_eval_episode(
                    self.model,
                    env,
                    seed=1000 + ep_idx,
                    spawn_idx=spawn_idx,
                    deterministic=self.deterministic,
                )
            )
        return results

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or (self.num_timesteps % self.eval_freq) != 0:
            return True
        if not self.validation_tracks:
            return True

        eval_start = time.time()

        validation_results = {track: self._eval_track(track) for track in self.validation_tracks}
        validation_flat = [ep for episodes in validation_results.values() for ep in episodes]
        validation_summary = summarize_episodes(validation_flat)
        validation_score = model_selection_score(validation_summary)

        for track, episodes in validation_results.items():
            log_episode_metrics(self.logger, f"eval_validation/{track}", episodes)
        log_episode_metrics(self.logger, "eval_validation", validation_flat)
        self.logger.record("eval_validation/model_selection_score", validation_score)

        train_summary = None
        if self.report_train_track and self.train_track:
            train_results = self._eval_track(self.train_track)
            train_summary = summarize_episodes(train_results)
            log_episode_metrics(self.logger, "eval_train", train_results)

        elapsed_train = time.time() - self._train_start_time
        elapsed_eval = time.time() - eval_start
        self.logger.record("time/wall_clock_hours", elapsed_train / 3600.0)
        self.logger.record("time/eval_seconds", elapsed_eval)
        self.logger.dump(self.num_timesteps)

        snapshot = {
            "timestep": self.num_timesteps,
            "wall_clock_hours": elapsed_train / 3600.0,
            "validation_tracks": {
                track: [vars(ep) for ep in episodes]
                for track, episodes in validation_results.items()
            },
            "validation_summary": validation_summary,
            "validation_model_selection_score": validation_score,
        }
        if train_summary is not None:
            snapshot["train_track"] = self.train_track
            snapshot["train_summary"] = train_summary

        json_path = self.results_dir / f"eval_{self.num_timesteps:09d}.json"
        json_path.write_text(json.dumps(snapshot, indent=2))

        if validation_score > self.best_validation_score:
            self.best_validation_score = validation_score
            self.model.save(str(self.results_dir / "best_validation_model"))
            if self.verbose:
                print(
                    f"[eval] New best validation checkpoint at step {self.num_timesteps} "
                    f"(score={validation_score:.3f}, completion={validation_summary.get('completion_rate', 0.0):.3f}, "
                    f"progress={validation_summary.get('mean_progress', 0.0):.3f})"
                )

        if self.verbose:
            print(
                f"[eval] step={self.num_timesteps} "
                f"val_completion={validation_summary.get('completion_rate', 0.0):.3f} "
                f"val_progress={validation_summary.get('mean_progress', 0.0):.3f} "
                f"val_crash={validation_summary.get('crash_rate', 0.0):.3f} "
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle_cfg", default="configs/vehicle.yaml")
    ap.add_argument("--sac_cfg", default="configs/sac.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--action_space", default=None)
    ap.add_argument("--ablate_geometry", action="store_true")
    ap.add_argument("--train_track", default=None)
    ap.add_argument("--validation_tracks", default=None)
    ap.add_argument("--test_tracks", default=None)
    ap.add_argument("--eval_tracks", default=None, help="Legacy alias for validation tracks")
    ap.add_argument("--n_eval_episodes", type=int, default=10)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--early_stop_patience", type=int, default=None)
    ap.add_argument("--save_replay_buffer", action="store_true")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    veh_cfg = load_yaml(ROOT / args.vehicle_cfg)
    sac_cfg = load_yaml(ROOT / args.sac_cfg)

    seed = args.seed if args.seed is not None else int(sac_cfg.get("seed", 0))
    set_random_seed(seed)

    if args.action_space:
        veh_cfg["action_space"] = args.action_space
    if args.ablate_geometry:
        veh_cfg["ablate_centerline_features"] = True

    action_space_name = veh_cfg.get("action_space", "steer_speed")

    if args.train_track:
        train_track = normalize_track_name(args.train_track)
    else:
        configured = sac_cfg.get("train_track")
        if configured:
            train_track = normalize_track_name(configured)
        else:
            raw_map = veh_cfg.get("sim", {}).get("map_name", "Sakhir_map")
            train_track = normalize_track_name(raw_map)

    validation_override = args.validation_tracks or args.eval_tracks
    if validation_override is not None:
        validation_tracks = _csv_to_tracks(validation_override) or []
    else:
        validation_tracks = [normalize_track_name(t) for t in sac_cfg.get("validation_tracks", sac_cfg.get("heldout_tracks", []))]

    if args.test_tracks is not None:
        test_tracks = _csv_to_tracks(args.test_tracks) or []
    else:
        test_tracks = [normalize_track_name(t) for t in sac_cfg.get("test_tracks", [])]

    validation_tracks = [t for t in validation_tracks if t != train_track]
    test_tracks = [t for t in test_tracks if t != train_track and t not in validation_tracks]
    validation_tracks = _dedupe_keep_order(validation_tracks)
    test_tracks = _dedupe_keep_order(test_tracks)

    ablate_tag = "_ablated" if args.ablate_geometry else ""
    run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}_{action_space_name}{ablate_tag}_s{seed}"
    runs_dir = ROOT / "runs"
    ckpt_dir = ROOT / "checkpoints" / run_id
    results_dir = ckpt_dir / "eval_results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    env = make_env_for_track(veh_cfg, train_track, render_mode=None)

    train_steps = int(sac_cfg.get("train_steps", 500_000))
    eval_freq = int(sac_cfg.get("eval_interval_steps", 5_000))
    save_every = int(sac_cfg.get("save_every_steps", 25_000))
    lr_schedule = make_lr_schedule(sac_cfg)
    target_entropy = resolve_target_entropy(sac_cfg, action_space_name)

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
            policy_kwargs=sac_cfg.get("policy_kwargs"),
            verbose=1,
            tensorboard_log=str(runs_dir),
            device=args.device,
            ent_coef=sac_cfg.get("ent_coef", "auto"),
            target_entropy=target_entropy,
        )

    run_meta = {
        "run_id": run_id,
        "seed": seed,
        "action_space": action_space_name,
        "action_dim": int(get_policy_dim(action_space_name)),
        "ablate_geometry": args.ablate_geometry,
        "train_track": train_track,
        "validation_tracks": validation_tracks,
        "test_tracks": test_tracks,
        "n_eval_episodes": args.n_eval_episodes,
        "selection_metric": "completion_then_progress",
        "selected_checkpoint_name": "best_validation_model.zip",
        "resumed_from": args.resume,
        "obs_space": str(env.observation_space),
        "act_space": str(env.action_space),
        "vehicle_cfg": veh_cfg,
        "sac_cfg": sac_cfg,
    }
    (ckpt_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str))

    print("=== Training setup ===")
    print(f"  run_id:            {run_id}")
    print(f"  seed:              {seed}")
    print(f"  action_space:      {action_space_name}")
    print(f"  action_dim:        {get_policy_dim(action_space_name)}")
    print(f"  ablate_geom:       {args.ablate_geometry}")
    print(f"  train_track:       {train_track}")
    print(f"  validation_tracks: {validation_tracks}")
    print(f"  test_tracks:       {test_tracks}")
    print(f"  target_entropy:    {target_entropy}")
    print(f"  checkpoints:       {ckpt_dir}")

    callbacks: List[BaseCallback] = [
        CheckpointCallback(
            save_freq=save_every,
            save_path=str(ckpt_dir),
            name_prefix="sac",
            save_replay_buffer=args.save_replay_buffer,
            save_vecnormalize=False,
        )
    ]

    if validation_tracks:
        callbacks.append(
            ValidationEvalCallback(
                vehicle_cfg=veh_cfg,
                train_track=train_track,
                validation_tracks=validation_tracks,
                eval_freq=eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                results_dir=results_dir,
                report_train_track=True,
                deterministic=True,
                verbose=1,
            )
        )

        early_stop_patience = args.early_stop_patience or sac_cfg.get("early_stop_patience")
        if early_stop_patience is not None:
            callbacks.append(
                EarlyStoppingOnMetric(
                    metric_name="eval_validation/model_selection_score",
                    eval_freq=eval_freq,
                    patience=int(early_stop_patience),
                    min_delta=float(sac_cfg.get("early_stop_min_delta", 0.0)),
                    verbose=1,
                )
            )

    model.learn(
        total_timesteps=train_steps,
        reset_num_timesteps=not resuming,
        tb_log_name=run_id,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(str(ckpt_dir / "sac_final"))
    if args.save_replay_buffer:
        model.save_replay_buffer(str(ckpt_dir / "sac_final_replay_buffer"))

    try:
        env.close()
    except Exception:
        pass

    print(f"\nSaved final model: {ckpt_dir / 'sac_final'}")


if __name__ == "__main__":
    main()
