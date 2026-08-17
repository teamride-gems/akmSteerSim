#!/usr/bin/env python3
"""Run the five-policy action-interface ensemble falsification study.

The study has a fail-closed order:
  1. train only the missing preregistered members;
  2. require competence and coarse behavior matching on fixed ID starts;
  3. execute one fixed direct-action anchor under two shift families while all
     ensemble members run in shadow mode on the same observations;
  4. compare heterogeneous action-interface disagreement against a same-
     interface seed ensemble and a nonlinear temporal state-risk baseline.

This is a screening study.  Passing it authorizes a larger crossed-seed study;
failing any gate kills or redesigns the hypothesis without member replacement.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import zlib
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import yaml
from scipy.optimize import minimize
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.common import (  # noqa: E402
    arc_length_spawn_indices,
    make_env_for_track,
    run_eval_episode,
    summarize_episodes,
)
from utils.action_spaces_utils import raw_action_to_command  # noqa: E402
from utils.provenance import collect_provenance, utc_now_iso, write_json  # noqa: E402
from envs.f1tenth_sb3_env import STATE_N_SCALARS  # noqa: E402


SCORE_NAMES = ("heterogeneous", "homogeneous", "temporal_risk")


def _read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _checkpoint(run_dir: Path) -> Path:
    best = run_dir / "eval_results" / "best_validation_model.zip"
    final = run_dir / "sac_final.zip"
    if best.exists():
        return best
    if final.exists():
        return final
    raise FileNotFoundError(f"No selected or final checkpoint in {run_dir}")


def _complete_run(run_dir: Path, member: Mapping[str, Any], expected_steps: int) -> bool:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return False
    meta = _read_json(meta_path)
    return bool(
        meta.get("status") == "complete"
        and int(meta.get("final_num_timesteps", -1)) == expected_steps
        and meta.get("action_space") == member["action_space"]
        and int(meta.get("seed", -1)) == int(member["seed"])
        and (_checkpoint(run_dir).exists())
    )


def train_missing_members(cfg: Dict[str, Any], max_parallel: int | None = None) -> None:
    training = cfg["training"]
    sac_cfg_path = _resolve(training["sac_cfg"])
    sac_cfg = _read_yaml(sac_cfg_path)
    expected_steps = int(sac_cfg["train_steps"])
    device = str(training.get("device", "auto"))
    parallel = int(max_parallel or training.get("max_parallel", 1))
    if parallel <= 0:
        raise ValueError("max_parallel must be positive")

    pending: List[Tuple[str, Dict[str, Any], List[str]]] = []
    for name, member in cfg["members"].items():
        run_dir = _resolve(member["run_dir"])
        if _complete_run(run_dir, member, expected_steps):
            print(f"[reuse] {name}: {run_dir}")
            continue
        if run_dir.exists():
            raise RuntimeError(
                f"Preregistered run directory exists but is incomplete or mismatched: {run_dir}. "
                "The study fails closed; choose a new protocol version rather than overwriting it."
            )
        command = [
            sys.executable,
            str(ROOT / "rl" / "train.py"),
            "--vehicle_cfg", str(training["vehicle_cfg"]),
            "--sac_cfg", str(training["sac_cfg"]),
            "--action_space", str(member["action_space"]),
            "--seed", str(member["seed"]),
            "--run_id", run_dir.name,
            "--device", device,
        ]
        pending.append((name, member, command))

    log_dir = _resolve(cfg["outputs"]["root"]) / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    active: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    while pending or active:
        while pending and len(active) < parallel:
            name, member, command = pending.pop(0)
            stdout_path = log_dir / f"{name}.stdout.log"
            stderr_path = log_dir / f"{name}.stderr.log"
            stdout = stdout_path.open("w", encoding="utf-8")
            stderr = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            active[name] = {
                "process": process,
                "member": member,
                "stdout": stdout,
                "stderr": stderr,
                "started": time.time(),
            }
            print(f"[train] {name} pid={process.pid}")

        time.sleep(2.0)
        for name, state in list(active.items()):
            code = state["process"].poll()
            if code is None:
                continue
            state["stdout"].close()
            state["stderr"].close()
            member = state["member"]
            run_dir = _resolve(member["run_dir"])
            elapsed = time.time() - state["started"]
            if code == 0 and _complete_run(run_dir, member, expected_steps):
                print(f"[trained] {name} in {elapsed / 60.0:.1f} min")
            else:
                failures.append({"member": name, "return_code": code})
                print(f"[failed] {name} return_code={code}")
            del active[name]
    if failures:
        raise RuntimeError(f"Training failures: {failures}")


def load_members(cfg: Dict[str, Any], device: str | None = None):
    loaded: Dict[str, Dict[str, Any]] = {}
    use_device = device or str(cfg["training"].get("device", "auto"))
    for name, member in cfg["members"].items():
        run_dir = _resolve(member["run_dir"])
        meta = _read_json(run_dir / "run_meta.json")
        vehicle = _read_yaml(run_dir / "resolved_vehicle.yaml")
        checkpoint = _checkpoint(run_dir)
        model = SAC.load(str(checkpoint), device=use_device)
        loaded[name] = {
            "name": name,
            "model": model,
            "action_space": member["action_space"],
            "seed": int(member["seed"]),
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "vehicle_cfg": vehicle,
            "meta": meta,
        }
    return loaded


def evaluate_competence(cfg: Dict[str, Any], members) -> Dict[str, Any]:
    gate = cfg["competence"]
    track = str(gate["track"])
    starts = int(gate["starts"])
    per_member: Dict[str, Any] = {}
    for name, member in members.items():
        env = make_env_for_track(member["vehicle_cfg"], track, render_mode=None)
        spawn_indices = arc_length_spawn_indices(env.centerline, starts)
        episodes = []
        try:
            for episode_idx, spawn_idx in enumerate(spawn_indices):
                episodes.append(
                    run_eval_episode(
                        member["model"], env,
                        seed=20260816 + episode_idx,
                        spawn_idx=spawn_idx,
                        deterministic=True,
                    )
                )
        finally:
            env.close()
        per_member[name] = {
            "summary": summarize_episodes(episodes),
            "episodes": [vars(ep) for ep in episodes],
        }
        print(
            f"[competence] {name}: completion={per_member[name]['summary']['completion_rate']:.1%} "
            f"speed={per_member[name]['summary']['mean_speed']:.2f}"
        )

    anchor = per_member[cfg["anchor_member"]]["summary"]
    checks: Dict[str, Dict[str, bool]] = {}
    for name, item in per_member.items():
        summary = item["summary"]
        length_ratio = summary["mean_length"] / max(anchor["mean_length"], 1e-9)
        speed_ratio = summary["mean_speed"] / max(anchor["mean_speed"], 1e-9)
        steer_ratio = summary["mean_steer_rate"] / max(anchor["mean_steer_rate"], 1e-9)
        member_checks = {
            "completion": summary["completion_rate"] >= float(gate["minimum_completion_rate"]),
            "lap_time": abs(length_ratio - 1.0) <= float(gate["maximum_lap_time_ratio_error"]),
            "speed": abs(speed_ratio - 1.0) <= float(gate["maximum_speed_ratio_error"]),
            "lateral_error": summary["mean_lateral_error"] <= (
                anchor["mean_lateral_error"]
                + float(gate["maximum_mean_lateral_error_increase_m"])
            ),
            "steer_rate": float(gate["minimum_steer_rate_ratio"]) <= steer_ratio
            <= float(gate["maximum_steer_rate_ratio"]),
        }
        item["behavior_ratios"] = {
            "lap_time": length_ratio,
            "speed": speed_ratio,
            "steer_rate": steer_ratio,
        }
        item["checks"] = member_checks
        item["passed"] = bool(all(member_checks.values()))
        checks[name] = member_checks
    return {
        "passed": bool(all(item["passed"] for item in per_member.values())),
        "track": track,
        "fixed_starts": starts,
        "anchor": cfg["anchor_member"],
        "members": per_member,
        "thresholds": deepcopy(gate),
    }


def _stable_seed(*parts: Any) -> int:
    return zlib.crc32(":".join(map(str, parts)).encode("utf-8")) & 0xFFFFFFFF


def corrupt_observation(obs: np.ndarray, shift: Mapping[str, Any], rng) -> np.ndarray:
    result = np.asarray(obs, dtype=np.float32).copy()
    if shift["kind"] == "lidar_dropout":
        lidar = result[STATE_N_SCALARS:]
        mask = rng.random(lidar.shape[0]) < float(shift["probability"])
        lidar[mask] = float(shift.get("replacement", 1.0))
    return result


def apply_anchor_shift(action: np.ndarray, shift: Mapping[str, Any], state: Dict[str, Any]):
    result = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if shift["kind"] == "steering_delay":
        queue = state["steering_queue"]
        queue.append(float(result[0]))
        result[0] = float(queue.popleft())
    return result


def physical_command_vector(command: Mapping[str, float], robot_cfg: Mapping[str, Any]) -> np.ndarray:
    steer_range = float(robot_cfg["max_steering_angle"] - robot_cfg["min_steering_angle"])
    speed_range = float(robot_cfg["max_speed"] - robot_cfg["min_speed"])
    return np.array(
        [
            float(command["steering_angle"]) / max(steer_range, 1e-9),
            float(command["speed"]) / max(speed_range, 1e-9),
        ],
        dtype=float,
    )


def pairwise_distances(vectors: Sequence[np.ndarray]) -> np.ndarray:
    values = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            values.append(float(np.linalg.norm(vectors[i] - vectors[j])))
    return np.asarray(values, dtype=float)


def raw_risk_signals(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=float).reshape(-1)
    lidar = obs[STATE_N_SCALARS:]
    return np.array(
        [
            float(np.min(lidar)),
            float(np.mean(lidar)),
            abs(float(obs[5])),  # normalized lateral error
            abs(float(obs[4])),  # normalized heading error
            abs(float(obs[3])),  # normalized yaw rate
            float(obs[0]),       # normalized speed
        ],
        dtype=float,
    )


def temporal_features(history: deque[np.ndarray]) -> np.ndarray:
    values = np.stack(tuple(history), axis=0)
    current = values[-1]
    mean = np.mean(values, axis=0)
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    delta = values[-1] - values[0]
    return np.concatenate([current, mean, minimum, maximum, delta]).astype(float)


def collect_episode(
    cfg: Dict[str, Any], members, split: str, family: str, track: str,
    episode_idx: int, spawn_idx: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    anchor_name = cfg["anchor_member"]
    anchor = members[anchor_name]
    shift = cfg["shifts"][family]
    env = make_env_for_track(anchor["vehicle_cfg"], track, render_mode=None)
    seed = _stable_seed(cfg["study_name"], split, family, track, episode_idx)
    rng = np.random.default_rng(seed)
    obs, reset_info = env.reset(seed=seed, options={"spawn_index": int(spawn_idx)})
    obs = corrupt_observation(obs, shift, rng)
    window = max(1, int(math.ceil(float(cfg["failure_horizon_seconds"]) / env.dt)))
    history: deque[np.ndarray] = deque(maxlen=window)
    shadow_previous = {
        name: {"steering_angle": 0.0, "speed": float(reset_info.get("speed", 0.0))}
        for name in members
    }
    delay_steps = int(shift.get("policy_steps", 0)) if shift["kind"] == "steering_delay" else 0
    shift_state = {"steering_queue": deque([0.0] * delay_steps)}
    rows: List[Dict[str, Any]] = []
    terminal_info: Dict[str, Any] = {}
    done = False
    try:
        while not done:
            actions: Dict[str, np.ndarray] = {}
            commands: Dict[str, np.ndarray] = {}
            for name, member in members.items():
                action, _ = member["model"].predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                actions[name] = action
                command = raw_action_to_command(
                    member["action_space"], action, env._robot_config,
                    prev_command=shadow_previous[name], dt=env.dt,
                    apply_final_constraints=True,
                )
                shadow_previous[name] = command
                commands[name] = physical_command_vector(command, env._robot_config)

            hetero_pairs = pairwise_distances(
                [commands[name] for name in cfg["heterogeneous_ensemble"]]
            )
            homo_pairs = pairwise_distances(
                [commands[name] for name in cfg["homogeneous_ensemble"]]
            )
            history.append(raw_risk_signals(obs))
            row = {
                "split": split,
                "family": family,
                "track": track,
                "episode_key": f"{split}:{family}:{track}:{episode_idx}",
                "step": len(rows),
                "heterogeneous": float(np.median(hetero_pairs)),
                "homogeneous": float(np.median(homo_pairs)),
                "heterogeneous_pairs": hetero_pairs,
                "homogeneous_pairs": homo_pairs,
                "risk_features": temporal_features(history),
                "dt": float(env.dt),
            }
            rows.append(row)

            executed_action = apply_anchor_shift(actions[anchor_name], shift, shift_state)
            obs, _, terminated, truncated, terminal_info = env.step(executed_action)
            obs = corrupt_observation(obs, shift, rng)
            done = bool(terminated or truncated)
    finally:
        env.close()

    failed = terminal_info.get("term_reason") == "crash"
    horizon_steps = max(1, int(math.ceil(float(cfg["failure_horizon_seconds"]) / rows[0]["dt"])))
    for row in rows:
        steps_to_terminal = len(rows) - 1 - int(row["step"])
        row["label"] = int(failed and steps_to_terminal < horizon_steps)
        row["steps_to_failure"] = steps_to_terminal if failed else -1
    episode = {
        "episode_key": rows[0]["episode_key"],
        "split": split,
        "family": family,
        "track": track,
        "spawn_index": int(spawn_idx),
        "steps": len(rows),
        "dt": rows[0]["dt"],
        "term_reason": terminal_info.get("term_reason", "unknown"),
        "failed": bool(failed),
        "progress": float(terminal_info.get("normalized_progress", 0.0)),
    }
    return rows, episode


def collect_dataset(cfg: Dict[str, Any], members):
    rows: List[Dict[str, Any]] = []
    episodes: List[Dict[str, Any]] = []
    for split in ("calibration", "test"):
        split_cfg = cfg[split]
        starts_per_track = int(split_cfg["starts_per_track"])
        for family in cfg["shifts"]:
            for track in split_cfg["tracks"]:
                env = make_env_for_track(members[cfg["anchor_member"]]["vehicle_cfg"], track)
                spawn_indices = arc_length_spawn_indices(env.centerline, starts_per_track)
                env.close()
                for episode_idx, spawn_idx in enumerate(spawn_indices):
                    episode_rows, episode = collect_episode(
                        cfg, members, split, family, str(track), episode_idx, spawn_idx
                    )
                    rows.extend(episode_rows)
                    episodes.append(episode)
                    print(
                        f"[{split}] {family}/{track} {episode_idx + 1}/{starts_per_track}: "
                        f"{episode['term_reason']} steps={episode['steps']}"
                    )
    return rows, episodes


class TemporalLogisticRisk:
    def __init__(self, l2: float = 1.0):
        self.l2 = float(l2)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.constant: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, groups: Sequence[str]):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if np.unique(y).size < 2:
            self.constant = float(np.mean(y))
            return self
        self.mean = np.mean(x, axis=0)
        self.scale = np.std(x, axis=0)
        self.scale[self.scale < 1e-6] = 1.0
        z = (x - self.mean) / self.scale
        z = np.column_stack([np.ones(z.shape[0]), z])

        groups = np.asarray(groups)
        weights = np.zeros(y.shape[0], dtype=float)
        for group in np.unique(groups):
            idx = groups == group
            weights[idx] = 1.0 / max(int(np.sum(idx)), 1)
        for label in (0.0, 1.0):
            idx = y == label
            if np.any(idx):
                weights[idx] *= 0.5 / max(float(np.sum(weights[idx])), 1e-12)
        weights *= y.size / max(float(np.sum(weights)), 1e-12)

        def objective(beta):
            logits = np.clip(z @ beta, -40.0, 40.0)
            loss = np.sum(weights * (np.logaddexp(0.0, logits) - y * logits))
            loss += 0.5 * self.l2 * float(np.dot(beta[1:], beta[1:]))
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            gradient = z.T @ (weights * (probabilities - y))
            gradient[1:] += self.l2 * beta[1:]
            return float(loss), gradient

        fitted = minimize(
            objective, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B",
            options={"maxiter": 300, "ftol": 1e-10},
        )
        if not fitted.success:
            raise RuntimeError(f"Temporal risk model failed to converge: {fitted.message}")
        self.coef = np.asarray(fitted.x, dtype=float)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.constant is not None:
            return np.full(x.shape[0], self.constant, dtype=float)
        if self.mean is None or self.scale is None or self.coef is None:
            raise RuntimeError("TemporalLogisticRisk is not fitted")
        z = np.column_stack([np.ones(x.shape[0]), (x - self.mean) / self.scale])
        logits = np.clip(z @ self.coef, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": "L2 temporal logistic risk model",
            "l2": self.l2,
            "constant": self.constant,
            "mean": None if self.mean is None else self.mean.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
            "coef": None if self.coef is None else self.coef.tolist(),
        }


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = int(np.sum(y == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ranked = y[order]
    tp = np.cumsum(ranked == 1)
    fp = np.cumsum(ranked == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum(precision[ranked == 1]) / positives)


def threshold_at_fpr(y: np.ndarray, score: np.ndarray, target_fpr: float) -> float:
    negatives = np.asarray(score, dtype=float)[np.asarray(y, dtype=int) == 0]
    if negatives.size == 0:
        return float("inf")
    return float(np.quantile(negatives, 1.0 - target_fpr, method="higher"))


def risk_coverage(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    order = np.argsort(np.asarray(score, dtype=float))
    result = {}
    for coverage in (0.50, 0.70, 0.80, 0.90, 0.95, 1.00):
        keep = max(1, int(math.ceil(coverage * y.size)))
        result[f"{coverage:.2f}"] = float(np.mean(y[order[:keep]]))
    return result


def score_metrics(rows: List[Dict[str, Any]], score_name: str, threshold: float):
    y = np.asarray([row["label"] for row in rows], dtype=int)
    score = np.asarray([row[score_name] for row in rows], dtype=float)
    predicted = score >= threshold
    positives = y == 1
    negatives = ~positives
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["episode_key"], []).append(row)
    warnings = []
    detected = 0
    safe_alarm_edges = 0
    safe_seconds = 0.0
    failure_events = 0
    for episode_rows in grouped.values():
        episode_rows.sort(key=lambda row: row["step"])
        failed = any(row["label"] for row in episode_rows)
        alarms = np.asarray([row[score_name] >= threshold for row in episode_rows], dtype=bool)
        if failed:
            failure_events += 1
            eligible = [row for row in episode_rows if row["label"] and row[score_name] >= threshold]
            if eligible:
                detected += 1
                earliest = min(eligible, key=lambda row: row["step"])
                warnings.append(float(earliest["steps_to_failure"] * earliest["dt"]))
            else:
                warnings.append(0.0)
        else:
            safe_seconds += sum(float(row["dt"]) for row in episode_rows)
            previous = False
            for alarm in alarms:
                if alarm and not previous:
                    safe_alarm_edges += 1
                previous = bool(alarm)
    return {
        "auprc": average_precision(y, score),
        "recall_at_calibrated_fpr": float(np.mean(predicted[positives])) if np.any(positives) else float("nan"),
        "frame_false_positive_rate": float(np.mean(predicted[negatives])) if np.any(negatives) else float("nan"),
        "median_warning_seconds_misses_zero": float(np.median(warnings)) if warnings else float("nan"),
        "event_recall": detected / failure_events if failure_events else float("nan"),
        "failure_events": failure_events,
        "safe_alarm_events_per_minute": safe_alarm_edges / max(safe_seconds / 60.0, 1e-12),
        "risk_coverage": risk_coverage(y, score),
    }


def _paired_macro_bootstrap(
    test_rows: List[Dict[str, Any]], a: str, b: str, samples: int, seed: int,
) -> List[float]:
    rng = np.random.default_rng(seed)
    by_family: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in test_rows:
        by_family.setdefault(row["family"], {}).setdefault(row["episode_key"], []).append(row)
    draws = []
    for _ in range(samples):
        family_diffs = []
        for episodes in by_family.values():
            keys = list(episodes)
            selected = rng.choice(keys, size=len(keys), replace=True)
            sampled = [row for key in selected for row in episodes[str(key)]]
            y = np.asarray([row["label"] for row in sampled], dtype=int)
            ap_a = average_precision(y, np.asarray([row[a] for row in sampled]))
            ap_b = average_precision(y, np.asarray([row[b] for row in sampled]))
            if np.isfinite(ap_a) and np.isfinite(ap_b):
                family_diffs.append(ap_a - ap_b)
        if family_diffs:
            draws.append(float(np.mean(family_diffs)))
    return draws


def _finite_or_none(value: Any):
    if isinstance(value, dict):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def analyze_dataset(cfg: Dict[str, Any], rows, episodes):
    calibration = [row for row in rows if row["split"] == "calibration"]
    test = [row for row in rows if row["split"] == "test"]
    x_cal = np.stack([row["risk_features"] for row in calibration])
    y_cal = np.asarray([row["label"] for row in calibration], dtype=int)
    groups_cal = [row["episode_key"] for row in calibration]
    model = TemporalLogisticRisk(l2=1.0).fit(x_cal, y_cal, groups_cal)
    for row, score in zip(calibration, model.predict(x_cal)):
        row["temporal_risk"] = float(score)
    x_test = np.stack([row["risk_features"] for row in test])
    for row, score in zip(test, model.predict(x_test)):
        row["temporal_risk"] = float(score)

    fpr = float(cfg["target_frame_false_positive_rate"])
    thresholds = {}
    for score_name in SCORE_NAMES:
        thresholds[score_name] = threshold_at_fpr(
            y_cal, np.asarray([row[score_name] for row in calibration]), fpr
        )

    families: Dict[str, Any] = {}
    for family in cfg["shifts"]:
        family_rows = [row for row in test if row["family"] == family]
        families[family] = {
            score: score_metrics(family_rows, score, thresholds[score])
            for score in SCORE_NAMES
        }

    macro = {}
    for score in SCORE_NAMES:
        macro[score] = {
            metric: float(np.nanmean([families[family][score][metric] for family in families]))
            for metric in (
                "auprc", "recall_at_calibrated_fpr", "frame_false_positive_rate",
                "median_warning_seconds_misses_zero", "event_recall",
                "safe_alarm_events_per_minute",
            )
        }

    samples = int(cfg["bootstrap_samples"])
    seed = int(cfg["bootstrap_seed"])
    comparisons = {}
    for offset, baseline in enumerate(("homogeneous", "temporal_risk")):
        draws = _paired_macro_bootstrap(
            test, "heterogeneous", baseline, samples, seed + offset
        )
        point = macro["heterogeneous"]["auprc"] - macro[baseline]["auprc"]
        comparisons[f"heterogeneous_minus_{baseline}"] = {
            "macro_auprc_difference": point,
            "paired_episode_bootstrap_ci95": (
                np.quantile(draws, [0.025, 0.975]).tolist() if draws else [float("nan"), float("nan")]
            ),
            "bootstrap_draws": len(draws),
        }

    rules = cfg["pass_rules"]
    gain_homo = comparisons["heterogeneous_minus_homogeneous"]
    gain_risk = comparisons["heterogeneous_minus_temporal_risk"]
    warning_required = float(cfg["fallback_latency_seconds"]) + float(
        rules["minimum_warning_margin_over_fallback_seconds"]
    )
    family_wins = {
        family: bool(
            families[family]["heterogeneous"]["auprc"] > families[family]["homogeneous"]["auprc"]
            and families[family]["heterogeneous"]["auprc"] > families[family]["temporal_risk"]["auprc"]
        )
        for family in families
    }
    pass_checks = {
        "gain_over_homogeneous": gain_homo["macro_auprc_difference"] >= float(
            rules["minimum_macro_auprc_gain_over_homogeneous"]
        ),
        "gain_over_temporal_risk": gain_risk["macro_auprc_difference"] >= float(
            rules["minimum_macro_auprc_gain_over_temporal_risk"]
        ),
        "ci_over_homogeneous_excludes_zero": gain_homo["paired_episode_bootstrap_ci95"][0] > 0.0,
        "ci_over_temporal_risk_excludes_zero": gain_risk["paired_episode_bootstrap_ci95"][0] > 0.0,
        "usable_warning": macro["heterogeneous"]["median_warning_seconds_misses_zero"] >= warning_required,
        "family_wins": all(family_wins.values()),
    }
    if not bool(rules.get("require_ci_excludes_zero", True)):
        pass_checks["ci_over_homogeneous_excludes_zero"] = True
        pass_checks["ci_over_temporal_risk_excludes_zero"] = True
    if not bool(rules.get("require_win_every_screen_family", True)):
        pass_checks["family_wins"] = True

    return {
        "thresholds_calibrated_at_frame_fpr": thresholds,
        "temporal_risk_model": model.as_dict(),
        "calibration": {
            "frames": len(calibration),
            "positive_frames": int(np.sum(y_cal)),
            "episodes": len({row["episode_key"] for row in calibration}),
        },
        "test": {
            "frames": len(test),
            "positive_frames": int(sum(row["label"] for row in test)),
            "episodes": len({row["episode_key"] for row in test}),
            "failure_episodes": sum(episode["failed"] for episode in episodes if episode["split"] == "test"),
        },
        "families": families,
        "macro": macro,
        "comparisons": comparisons,
        "family_wins": family_wins,
        "pass_checks": pass_checks,
        "passed": bool(all(pass_checks.values())),
        "warning_required_seconds": warning_required,
    }


def save_dataset(path: Path, rows: List[Dict[str, Any]], episodes: List[Dict[str, Any]]) -> None:
    episode_keys = sorted({row["episode_key"] for row in rows})
    episode_index = {key: idx for idx, key in enumerate(episode_keys)}
    arrays = {
        "risk_features": np.stack([row["risk_features"] for row in rows]),
        "heterogeneous_score": np.asarray([row["heterogeneous"] for row in rows]),
        "homogeneous_score": np.asarray([row["homogeneous"] for row in rows]),
        "temporal_risk_score": np.asarray([row.get("temporal_risk", np.nan) for row in rows]),
        "heterogeneous_pair_scores": np.stack([row["heterogeneous_pairs"] for row in rows]),
        "homogeneous_pair_scores": np.stack([row["homogeneous_pairs"] for row in rows]),
        "label": np.asarray([row["label"] for row in rows], dtype=np.int8),
        "episode_index": np.asarray([episode_index[row["episode_key"]] for row in rows], dtype=np.int32),
        "step": np.asarray([row["step"] for row in rows], dtype=np.int32),
        "steps_to_failure": np.asarray([row["steps_to_failure"] for row in rows], dtype=np.int32),
        "episode_keys": np.asarray(episode_keys),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".episodes.json").write_text(
        json.dumps(episodes, indent=2), encoding="utf-8"
    )


def write_report(path: Path, result: Dict[str, Any]) -> None:
    lines = [
        "# Action-interface ensemble screening study",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "This is the preregistered five-policy falsification screen, not a confirmatory population study.",
        "",
        "## Competence and behavior matching",
        "",
        "| Member | Completion | Mean speed | Mean lateral error | Passed |",
        "|---|---:|---:|---:|:---:|",
    ]
    for name, member in result["competence"]["members"].items():
        summary = member["summary"]
        lines.append(
            f"| {name} | {summary['completion_rate']:.1%} | {summary['mean_speed']:.2f} m/s | "
            f"{summary['mean_lateral_error']:.3f} m | {'yes' if member['passed'] else 'no'} |"
        )
    if result.get("analysis"):
        analysis = result["analysis"]
        lines.extend([
            "", "## Test results", "",
            "| Monitor | Macro AUPRC | Recall | Warning |",
            "|---|---:|---:|---:|",
        ])
        for score in SCORE_NAMES:
            metric = analysis["macro"][score]
            lines.append(
                f"| {score} | {metric['auprc']:.3f} | {metric['recall_at_calibrated_fpr']:.3f} | "
                f"{metric['median_warning_seconds_misses_zero']:.3f} s |"
            )
        lines.extend(["", "## Pass checks", ""])
        for key, passed in analysis["pass_checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'}: {key}")
    else:
        lines.extend([
            "", "OOD monitoring evaluation was not run because the competence/behavior gate failed.",
        ])
    lines.extend([
        "", "## Interpretation", "",
        result["interpretation"], "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/action_ensemble_screen.yaml")
    parser.add_argument("--phase", choices=["all", "train", "evaluate"], default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_parallel", type=int, default=None)
    parser.add_argument(
        "--continue_after_failed_competence", action="store_true",
        help="Diagnostic only; the formal default fails closed before OOD evaluation.",
    )
    args = parser.parse_args()

    cfg_path = _resolve(args.config)
    cfg = _read_yaml(cfg_path)
    output_root = _resolve(cfg["outputs"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    if args.phase in ("all", "train"):
        train_missing_members(cfg, max_parallel=args.max_parallel)
        if args.phase == "train":
            return

    members = load_members(cfg, device=args.device)
    competence = evaluate_competence(cfg, members)
    result: Dict[str, Any] = {
        "study": cfg["study_name"],
        "generated_at_utc": utc_now_iso(),
        "protocol": cfg,
        "competence": competence,
        "analysis": None,
        "provenance": collect_provenance(ROOT),
    }
    if not competence["passed"] and not args.continue_after_failed_competence:
        result["decision"] = "KILL_AT_COMPETENCE_GATE"
        failed = [name for name, item in competence["members"].items() if not item["passed"]]
        result["interpretation"] = (
            "The action-interface monitoring hypothesis was not tested because the preregistered "
            f"competence/behavior-matching gate failed for: {', '.join(failed)}. Continuing would "
            "confound disagreement with policy quality. No member replacement is allowed."
        )
    else:
        rows, episodes = collect_dataset(cfg, members)
        analysis = analyze_dataset(cfg, rows, episodes)
        result["analysis"] = analysis
        result["episodes"] = episodes
        result["decision"] = "PASS_TO_CROSSED_SEED_STUDY" if analysis["passed"] else "KILL_MONITORING_HYPOTHESIS"
        result["interpretation"] = (
            "The five-policy screen passed every preregistered effect, uncertainty, family, and "
            "warning-time rule; proceed to a crossed-seed confirmatory study."
            if analysis["passed"] else
            "At least one preregistered monitoring rule failed. Do not expand to nine or fifteen "
            "training runs without changing and preregistering a substantively new hypothesis."
        )
        save_dataset(output_root / "screen_dataset.npz", rows, episodes)

    clean_result = _finite_or_none(result)
    write_json(_resolve(cfg["outputs"]["report_json"]), clean_result)
    write_report(_resolve(cfg["outputs"]["report_markdown"]), clean_result)
    print(json.dumps({
        "decision": clean_result["decision"],
        "competence_passed": clean_result["competence"]["passed"],
        "analysis_passed": None if clean_result["analysis"] is None else clean_result["analysis"]["passed"],
        "report": str(_resolve(cfg["outputs"]["report_markdown"])),
    }, indent=2))


if __name__ == "__main__":
    main()
