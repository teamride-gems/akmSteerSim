"""
Shared training/evaluation utilities.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.f1tenth_sb3_env import F1TenthSACEnv


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


def make_lr_schedule(cfg: Dict[str, Any]) -> Union[float, Callable[[float], float]]:
    lr = cfg.get("learning_rate", 3e-4)

    if isinstance(lr, (int, float)):
        return float(lr)

    schedule_type = lr.get("schedule", "constant")
    initial = float(lr.get("initial", 3e-4))
    final = float(lr.get("final", 0.0))

    if schedule_type == "constant":
        return initial
    if schedule_type == "linear":
        def linear_schedule(progress_remaining: float) -> float:
            return final + (initial - final) * progress_remaining
        return linear_schedule
    if schedule_type == "cosine":
        def cosine_schedule(progress_remaining: float) -> float:
            return final + 0.5 * (initial - final) * (1.0 + np.cos(np.pi * (1.0 - progress_remaining)))
        return cosine_schedule

    raise ValueError(f"Unknown LR schedule type: {schedule_type}")


def compute_arc_length_cumulative(centerline: np.ndarray) -> np.ndarray:
    diffs = np.diff(centerline[:, :2], axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def arc_length_spawn_indices(centerline: np.ndarray, n_spawns: int) -> List[int]:
    n_points = centerline.shape[0]
    if n_points <= 3:
        return [1] * n_spawns

    cum_arc = compute_arc_length_cumulative(centerline)
    total_length = cum_arc[-1]
    margin = total_length * 0.01
    target_lengths = np.linspace(margin, total_length - margin, num=n_spawns)

    indices = []
    for target in target_lengths:
        idx = int(np.searchsorted(cum_arc, target, side="right"))
        idx = int(np.clip(idx, 1, n_points - 2))
        indices.append(idx)
    return indices


MIN_LIDAR_SENTINEL = float("inf")


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
    steer_tv: float = 0.0
    steer_tv_per_step: float = 0.0
    steer_clip_frac: float = 0.0
    speed_clip_frac: float = 0.0
    mean_steer_clip_mag: float = 0.0
    mean_speed_clip_mag: float = 0.0
    min_lidar: float = 0.0
    mean_reward_progress: float = 0.0
    mean_reward_a_long_pen: float = 0.0
    mean_reward_a_lat_pen: float = 0.0
    mean_reward_time_pen: float = 0.0
    mean_reward_crash_pen: float = 0.0


def run_eval_episode(model, env, seed: int, spawn_idx: int, deterministic: bool = True) -> EpisodeResult:
    obs, _ = env.reset(seed=seed, options={"spawn_index": spawn_idx})

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
    reward_progress = []
    reward_a_long_pen = []
    reward_a_lat_pen = []
    reward_time_pen = []
    reward_crash_pen = []

    done = False
    info: Dict[str, Any] = {}
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
        min_lidars.append(float(info.get("min_lidar", MIN_LIDAR_SENTINEL)))

        if info.get("steer_clipped", False):
            steer_clips += 1
        if info.get("speed_clipped", False):
            speed_clips += 1
        steer_clip_mags.append(float(info.get("steer_clip_mag", 0.0)))
        speed_clip_mags.append(float(info.get("speed_clip_mag", 0.0)))

        rb = info.get("reward_breakdown", {})
        reward_progress.append(float(rb.get("progress", 0.0)))
        reward_a_long_pen.append(float(rb.get("a_long_pen", 0.0)))
        reward_a_lat_pen.append(float(rb.get("a_lat_pen", 0.0)))
        reward_time_pen.append(float(rb.get("time_pen", 0.0)))
        reward_crash_pen.append(float(rb.get("crash_pen", 0.0)))

        done = bool(terminated or truncated)

    steer_arr = np.array(steer_cmds, dtype=float)
    steer_tv = float(np.sum(np.abs(np.diff(steer_arr)))) if len(steer_arr) > 1 else 0.0
    n = max(1, ep_len)
    real_lidars = [v for v in min_lidars if v < MIN_LIDAR_SENTINEL]

    return EpisodeResult(
        reward=ep_reward,
        length=ep_len,
        term_reason=info.get("term_reason", "unknown"),
        normalized_progress=float(info.get("normalized_progress", 0.0)),
        mean_lateral_error=float(np.mean(lat_errors)) if lat_errors else 0.0,
        max_lateral_error=float(np.max(lat_errors)) if lat_errors else 0.0,
        mean_heading_error=float(np.mean(head_errors)) if head_errors else 0.0,
        mean_speed=float(np.mean(speeds)) if speeds else 0.0,
        mean_abs_steer_rate=float(np.mean(abs_steer_rates)) if abs_steer_rates else 0.0,
        steer_tv=steer_tv,
        steer_tv_per_step=steer_tv / n,
        steer_clip_frac=steer_clips / n,
        speed_clip_frac=speed_clips / n,
        mean_steer_clip_mag=float(np.mean(steer_clip_mags)) if steer_clip_mags else 0.0,
        mean_speed_clip_mag=float(np.mean(speed_clip_mags)) if speed_clip_mags else 0.0,
        min_lidar=float(np.min(real_lidars)) if real_lidars else 0.0,
        mean_reward_progress=float(np.mean(reward_progress)) if reward_progress else 0.0,
        mean_reward_a_long_pen=float(np.mean(reward_a_long_pen)) if reward_a_long_pen else 0.0,
        mean_reward_a_lat_pen=float(np.mean(reward_a_lat_pen)) if reward_a_lat_pen else 0.0,
        mean_reward_time_pen=float(np.mean(reward_time_pen)) if reward_time_pen else 0.0,
        mean_reward_crash_pen=float(np.mean(reward_crash_pen)) if reward_crash_pen else 0.0,
    )


def log_episode_metrics(logger, prefix: str, episodes: List[EpisodeResult]) -> None:
    n = len(episodes)
    if n == 0:
        return

    def _mean(attr: str) -> float:
        return float(np.mean([getattr(e, attr) for e in episodes]))

    def _std(attr: str) -> float:
        return float(np.std([getattr(e, attr) for e in episodes]))

    logger.record(f"{prefix}/mean_reward", _mean("reward"))
    logger.record(f"{prefix}/std_reward", _std("reward"))
    logger.record(f"{prefix}/mean_progress", _mean("normalized_progress"))
    logger.record(f"{prefix}/completion_rate", sum(1 for e in episodes if e.normalized_progress >= 0.95) / n)
    logger.record(f"{prefix}/crash_rate", sum(1 for e in episodes if e.term_reason == "crash") / n)
    logger.record(f"{prefix}/mean_lateral_error", _mean("mean_lateral_error"))
    logger.record(f"{prefix}/std_lateral_error", _std("mean_lateral_error"))
    logger.record(f"{prefix}/mean_heading_error", _mean("mean_heading_error"))
    logger.record(f"{prefix}/mean_speed", _mean("mean_speed"))
    logger.record(f"{prefix}/mean_steer_rate", _mean("mean_abs_steer_rate"))
    logger.record(f"{prefix}/mean_steer_tv", _mean("steer_tv"))
    logger.record(f"{prefix}/mean_steer_tv_per_step", _mean("steer_tv_per_step"))
    logger.record(f"{prefix}/steer_clip_frac", _mean("steer_clip_frac"))
    logger.record(f"{prefix}/speed_clip_frac", _mean("speed_clip_frac"))
    logger.record(f"{prefix}/mean_ep_len", _mean("length"))
    logger.record(f"{prefix}/mean_reward_progress", _mean("mean_reward_progress"))
    logger.record(f"{prefix}/mean_reward_a_long_pen", _mean("mean_reward_a_long_pen"))
    logger.record(f"{prefix}/mean_reward_a_lat_pen", _mean("mean_reward_a_lat_pen"))
    logger.record(f"{prefix}/mean_reward_time_pen", _mean("mean_reward_time_pen"))
    logger.record(f"{prefix}/mean_reward_crash_pen", _mean("mean_reward_crash_pen"))


def summarize_episodes(episodes: List[EpisodeResult]) -> Dict[str, float]:
    n = len(episodes)
    if n == 0:
        return {}

    def _mean(attr: str) -> float:
        return float(np.mean([getattr(e, attr) for e in episodes]))

    def _std(attr: str) -> float:
        return float(np.std([getattr(e, attr) for e in episodes]))

    return {
        "n_episodes": n,
        "mean_reward": _mean("reward"),
        "std_reward": _std("reward"),
        "mean_progress": _mean("normalized_progress"),
        "completion_rate": sum(1 for e in episodes if e.normalized_progress >= 0.95) / n,
        "crash_rate": sum(1 for e in episodes if e.term_reason == "crash") / n,
        "timeout_rate": sum(1 for e in episodes if e.term_reason == "timeout") / n,
        "mean_length": _mean("length"),
        "mean_lateral_error": _mean("mean_lateral_error"),
        "std_lateral_error": _std("mean_lateral_error"),
        "max_lateral_error": float(np.max([e.max_lateral_error for e in episodes])),
        "mean_heading_error": _mean("mean_heading_error"),
        "mean_speed": _mean("mean_speed"),
        "mean_steer_rate": _mean("mean_abs_steer_rate"),
        "mean_steer_tv": _mean("steer_tv"),
        "mean_steer_tv_per_step": _mean("steer_tv_per_step"),
        "steer_clip_frac": _mean("steer_clip_frac"),
        "speed_clip_frac": _mean("speed_clip_frac"),
        "mean_reward_progress": _mean("mean_reward_progress"),
        "mean_reward_a_long_pen": _mean("mean_reward_a_long_pen"),
        "mean_reward_a_lat_pen": _mean("mean_reward_a_lat_pen"),
        "mean_reward_time_pen": _mean("mean_reward_time_pen"),
        "mean_reward_crash_pen": _mean("mean_reward_crash_pen"),
    }


def model_selection_score(summary: Dict[str, float]) -> float:
    """
    Scalarized lexicographic score:
      1) completion rate
      2) mean progress
      3) lower crash rate
      4) reward as weak tie-breaker (clamped to prevent overflow)
    """
    # Clamp reward contribution so it can never override the completion
    # or progress tiers even at extreme episode lengths / reward scales.
    reward_contribution = float(np.clip(
        summary.get("mean_reward", 0.0), -1e4, 1e4
    ))
    return (
        1_000_000.0 * float(summary.get("completion_rate", 0.0))
        + 1_000.0 * float(summary.get("mean_progress", 0.0))
        - 10.0 * float(summary.get("crash_rate", 0.0))
        + 0.001 * reward_contribution
    )


def aggregate_track_group(episodes_by_track: Dict[str, List[EpisodeResult]]) -> Tuple[List[EpisodeResult], Dict[str, Dict[str, float]]]:
    flat: List[EpisodeResult] = []
    per_track_summary: Dict[str, Dict[str, float]] = {}
    for track, episodes in episodes_by_track.items():
        flat.extend(episodes)
        per_track_summary[track] = summarize_episodes(episodes)
    return flat, per_track_summary