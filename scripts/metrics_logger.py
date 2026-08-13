"""
CSV logging for episode summaries and per-step trajectories.

Designed for interactive rollouts (scripts/run_trained_policy.py, etc.).
Field sets are aligned with rl/common.EpisodeResult so that CSV data
is directly comparable to the JSON snapshots produced by eval.py.

For batch evaluation, use eval.py — it writes richer JSON output.
This logger is for ad-hoc runs where CSV is more convenient.
"""

import csv
from pathlib import Path
from typing import Dict, Optional


# ----------------------------
# Episode-level logger
# ----------------------------

# Aligned with EpisodeResult fields from rl/common.py
EPISODE_FIELDS = [
    "episode_id",
    "action_space",
    "track",
    "seed",
    "spawn_index",
    "reward",
    "length",
    "term_reason",
    "normalized_progress",
    "mean_lateral_error",
    "max_lateral_error",
    "mean_heading_error",
    "mean_speed",
    "mean_abs_steer_rate",
    "steer_tv",
    "steer_tv_per_step",
    "steer_clip_frac",
    "speed_clip_frac",
    "mean_steer_clip_mag",
    "mean_speed_clip_mag",
    "min_lidar",
]

# ----------------------------
# Step-level logger
# ----------------------------

STEP_FIELDS = [
    "episode_id",
    "step",
    "x",
    "y",
    "yaw",
    "speed",
    "steer_cmd",
    "speed_cmd",
    "steer_rate",
    "lateral_error",
    "heading_error",
    "min_lidar",
    "reward",
]


class EpisodeLogger:
    """
    Logs one row per episode to a CSV file.

    Usage:
        with EpisodeLogger("metrics/episodes.csv") as logger:
            logger.log(episode_id=0, action_space="steer_speed", ...)
    """

    def __init__(self, csv_path: str = "metrics/episodes.csv"):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=EPISODE_FIELDS)
        self._writer.writeheader()

    def log(self, **kwargs) -> None:
        """
        Log one episode. Accepts any subset of EPISODE_FIELDS as kwargs.
        Missing fields are written as empty strings.
        """
        row = {f: kwargs.get(f, "") for f in EPISODE_FIELDS}
        self._writer.writerow(row)
        self._file.flush()

    def log_result(
        self,
        episode_id: int,
        result,
        action_space: str = "",
        track: str = "",
        seed: int = -1,
        spawn_index: int = -1,
    ) -> None:
        """
        Log directly from an EpisodeResult dataclass (from rl/common.py).
        """
        from dataclasses import asdict
        d = asdict(result)
        d["episode_id"] = episode_id
        d["action_space"] = action_space
        d["track"] = track
        d["seed"] = seed
        d["spawn_index"] = spawn_index
        self.log(**d)

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class StepLogger:
    """
    Logs one row per timestep to a CSV file.
    Captures the per-step trajectory data needed for paper figures
    (steering profiles, tracking error over time, etc.).

    Usage:
        with StepLogger("metrics/trajectory.csv") as logger:
            logger.log(episode_id=0, step=1, info_dict)
    """

    def __init__(self, csv_path: str = "metrics/trajectory.csv"):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=STEP_FIELDS)
        self._writer.writeheader()

    def log(self, episode_id: int, info: Dict, reward: float = 0.0) -> None:
        """
        Log one timestep from the env's info dict.

        Args:
            episode_id: which episode this step belongs to
            info: the info dict returned by env.step()
            reward: the scalar reward for this step
        """
        pose = info.get("pose", [0.0, 0.0, 0.0])
        row = {
            "episode_id": int(episode_id),
            "step": int(info.get("step", 0)),
            "x": float(pose[0]),
            "y": float(pose[1]),
            "yaw": float(pose[2]),
            "speed": float(info.get("speed", 0.0)),
            "steer_cmd": float(info.get("steer_cmd", 0.0)),
            "speed_cmd": float(info.get("speed_cmd", 0.0)),
            "steer_rate": float(info.get("steer_rate", 0.0)),
            "lateral_error": float(info.get("lateral_error", 0.0)),
            "heading_error": float(info.get("heading_error", 0.0)),
            "min_lidar": float(info.get("min_lidar", 0.0)),
            "reward": float(reward),
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()