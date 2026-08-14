#!/usr/bin/env python3
"""Run the preregistered non-training Gate 1 decoder-state audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.provenance import collect_provenance, utc_now_iso, write_json


class Regressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, int(width)), nn.ReLU()))
            previous = int(width)
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def _standardize(train: np.ndarray, other: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (train - mean) / scale, (other - mean) / scale, mean, scale


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    cfg: Dict,
    seed: int,
) -> np.ndarray:
    x_train_n, x_test_n, _, _ = _standardize(x_train, x_test)
    y_train_n, _, y_mean, y_scale = _standardize(y_train, y_train)

    rng = np.random.default_rng(seed)
    validation_size = max(1, int(round(0.15 * len(x_train_n))))
    order = rng.permutation(len(x_train_n))
    validation_idx = order[:validation_size]
    fitting_idx = order[validation_size:]
    if fitting_idx.size == 0:
        raise ValueError("Training fold is too small for a validation split.")

    torch.manual_seed(seed)
    model = Regressor(
        x_train_n.shape[1],
        y_train_n.shape[1],
        cfg["hidden_sizes"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_fn = nn.MSELoss()
    batch_size = int(cfg["batch_size"])
    patience = int(cfg["early_stopping_patience"])
    best_loss = float("inf")
    best_state = None
    stale = 0

    x_tensor = torch.as_tensor(x_train_n, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_train_n, dtype=torch.float32)
    for _ in range(int(cfg["maximum_epochs"])):
        model.train()
        for start in range(0, fitting_idx.size, batch_size):
            batch = fitting_idx[start : start + batch_size]
            prediction = model(x_tensor[batch])
            loss = loss_fn(prediction, y_tensor[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_fn(model(x_tensor[validation_idx]), y_tensor[validation_idx]).item()
            )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("Supervised audit model did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x_test_n), batch_size):
            batch = torch.as_tensor(x_test_n[start : start + batch_size], dtype=torch.float32)
            predictions.append(model(batch).numpy())
    return np.concatenate(predictions, axis=0) * y_scale + y_mean


def _group_folds(groups: np.ndarray, folds: int, seed: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups)
    if unique.size < folds:
        raise ValueError(f"Need at least {folds} episodes; found {unique.size}.")
    shuffled = np.random.default_rng(seed).permutation(unique)
    for held_out in np.array_split(shuffled, folds):
        test = np.isin(groups, held_out)
        yield np.flatnonzero(~test), np.flatnonzero(test)


def _cross_validated_predictions(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: Dict,
    seed_offset: int,
) -> np.ndarray:
    predictions = np.empty_like(y, dtype=np.float32)
    for fold, (train_idx, test_idx) in enumerate(
        _group_folds(groups, int(cfg["folds"]), int(cfg["seed"]))
    ):
        predictions[test_idx] = _fit_predict(
            x[train_idx], y[train_idx], x[test_idx], cfg, int(cfg["seed"]) + seed_offset + fold
        )
    return predictions


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict:
    mse = np.mean((target - prediction) ** 2, axis=0)
    variance = np.var(target, axis=0)
    nmse = mse / np.maximum(variance, 1e-12)
    r2 = 1.0 - nmse
    return {
        "mse": mse.tolist(),
        "nmse": nmse.tolist(),
        "r2": r2.tolist(),
    }


def _matched_history_contrast(
    observation: np.ndarray,
    action: np.ndarray,
    register: np.ndarray,
    command: np.ndarray,
    cfg: Dict,
) -> Dict:
    features = np.concatenate((observation, action), axis=1)
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    keep = scale >= 1e-6
    standardized = (features[:, keep] - mean[keep]) / scale[keep]
    tree = cKDTree(standardized)
    k = min(int(cfg["neighbors"]) + 1, len(standardized))
    distances, neighbors = tree.query(standardized, k=k, workers=-1)
    dimension_scale = np.sqrt(max(1, standardized.shape[1]))
    caliper = float(cfg["maximum_standardized_rms_distance"])
    minimum_h = float(cfg["minimum_register_separation_rad"])

    pairs = set()
    pair_distance = []
    register_separation = []
    command_effect = []
    for i in range(len(standardized)):
        candidates = []
        for distance, j in zip(np.atleast_1d(distances[i])[1:], np.atleast_1d(neighbors[i])[1:]):
            j = int(j)
            rms_distance = float(distance / dimension_scale)
            h_difference = abs(float(register[i, 0] - register[j, 0]))
            if rms_distance <= caliper and h_difference >= minimum_h:
                candidates.append((h_difference, rms_distance, j))
        if not candidates:
            continue
        h_difference, rms_distance, j = max(candidates)
        pair = (min(i, j), max(i, j))
        if pair in pairs:
            continue
        pairs.add(pair)
        pair_distance.append(rms_distance)
        register_separation.append(h_difference)
        command_effect.append(abs(float(command[i, 0] - command[j, 0])))

    return {
        "n_pairs": len(pairs),
        "median_standardized_rms_distance": (
            float(np.median(pair_distance)) if pair_distance else None
        ),
        "median_register_separation_rad": (
            float(np.median(register_separation)) if register_separation else None
        ),
        "median_command_effect_rad": (
            float(np.median(command_effect)) if command_effect else None
        ),
        "command_effect_quantiles_rad": (
            np.quantile(command_effect, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
            if command_effect else []
        ),
    }


def gate_decision(metrics: Dict, cfg: Dict) -> Tuple[bool, Dict[str, bool]]:
    thresholds = cfg["thresholds"]
    matched_cfg = cfg["matched_histories"]
    checks = {
        "register_not_recoverable": (
            metrics["register_from_observation"]["r2"][0]
            < float(thresholds["maximum_register_steer_r2"])
        ),
        "register_improves_command_prediction": (
            metrics["steer_nmse_reduction"]
            >= float(thresholds["minimum_steer_nmse_reduction"])
        ),
        "limiter_activates": (
            metrics["steer_limiter_activation_fraction"]
            >= float(thresholds["minimum_steer_limiter_activation_fraction"])
        ),
        "enough_matched_histories": (
            metrics["matched_histories"]["n_pairs"]
            >= int(matched_cfg["minimum_pairs"])
        ),
        "matched_histories_have_effect": (
            metrics["matched_histories"]["median_command_effect_rad"] is not None
            and metrics["matched_histories"]["median_command_effect_rad"]
            >= float(thresholds["minimum_matched_command_effect_rad"])
        ),
    }
    return all(checks.values()), checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--config", default="configs/decoder_state_gate.yaml")
    parser.add_argument("--output", default="experiments/gate1/decoder_state_audit.json")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    loaded = [np.load(path) for path in args.datasets]
    required = {
        "policy_observation",
        "raw_action",
        "previous_command",
        "executed_command",
        "episode_id",
        "episode_complete",
        "steer_limiter_active",
    }
    for path, dataset in zip(args.datasets, loaded):
        missing = required.difference(dataset.files)
        if missing:
            raise KeyError(f"Dataset {path} is missing {sorted(missing)}.")

    arrays = {key: np.concatenate([dataset[key] for dataset in loaded], axis=0) for key in required}
    episode_id = arrays["episode_id"].astype(np.int64)
    # Prevent episode identifiers from separate files colliding.
    if len(loaded) > 1:
        episode_chunks = []
        offset = 0
        for dataset in loaded:
            ids = dataset["episode_id"].astype(np.int64)
            ids = ids - ids.min() + offset
            episode_chunks.append(ids)
            offset = int(ids.max()) + 1
        episode_id = np.concatenate(episode_chunks)

    mask = np.ones(len(episode_id), dtype=bool)
    if bool(cfg["data"]["require_completed_episodes"]):
        mask &= arrays["episode_complete"].astype(bool)
    selected = np.flatnonzero(mask)
    selected_episodes = np.unique(episode_id[selected])
    if selected_episodes.size < int(cfg["data"]["minimum_episodes"]):
        raise ValueError(
            f"Gate 1 requires {cfg['data']['minimum_episodes']} qualifying episodes; "
            f"found {selected_episodes.size}."
        )
    if selected.size < int(cfg["data"]["minimum_transitions"]):
        raise ValueError(
            f"Gate 1 requires {cfg['data']['minimum_transitions']} qualifying transitions; "
            f"found {selected.size}."
        )

    maximum = int(cfg["data"]["maximum_transitions"])
    if selected.size > maximum:
        selected = np.sort(
            np.random.default_rng(int(cfg["cross_validation"]["seed"])).choice(
                selected, size=maximum, replace=False
            )
        )

    observation = arrays["policy_observation"][selected].astype(np.float32)
    action = arrays["raw_action"][selected].astype(np.float32)
    register = arrays["previous_command"][selected].astype(np.float32)
    command = arrays["executed_command"][selected].astype(np.float32)
    groups = episode_id[selected]
    cv_cfg = cfg["cross_validation"]

    m_prediction = _cross_validated_predictions(observation, register, groups, cv_cfg, 0)
    oa = np.concatenate((observation, action), axis=1)
    f_prediction = _cross_validated_predictions(oa, command, groups, cv_cfg, 100)
    oah = np.concatenate((observation, action, register), axis=1)
    g_prediction = _cross_validated_predictions(oah, command, groups, cv_cfg, 200)

    m_metrics = _regression_metrics(register, m_prediction)
    f_metrics = _regression_metrics(command, f_prediction)
    g_metrics = _regression_metrics(command, g_prediction)
    steer_nmse_reduction = float(
        (f_metrics["nmse"][0] - g_metrics["nmse"][0])
        / max(f_metrics["nmse"][0], 1e-12)
    )
    metrics = {
        "n_transitions": int(len(selected)),
        "n_episodes": int(np.unique(groups).size),
        "register_from_observation": m_metrics,
        "command_from_observation_action": f_metrics,
        "command_from_observation_action_register": g_metrics,
        "steer_nmse_reduction": steer_nmse_reduction,
        "steer_limiter_activation_fraction": float(
            np.mean(arrays["steer_limiter_active"][selected])
        ),
        "matched_histories": _matched_history_contrast(
            observation, action, register, command, cfg["matched_histories"]
        ),
    }
    passed, checks = gate_decision(metrics, cfg)
    report = {
        "gate": "Gate 1 decoder-state relevance",
        "generated_at_utc": utc_now_iso(),
        "passed": passed,
        "checks": checks,
        "metrics": metrics,
        "config": cfg,
        "datasets": [str(Path(path).resolve()) for path in args.datasets],
        "channel_order": ["steering", "speed"],
        "provenance": collect_provenance(ROOT),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(json.dumps({"passed": passed, "checks": checks, "metrics": metrics}, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
