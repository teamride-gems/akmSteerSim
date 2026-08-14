#!/usr/bin/env python3
"""Evaluate the preregistered direct-action Gate 0 criterion.

The independent replicate is a trained seed. Each seed is evaluated on the
same 50 arc-length-spaced Sakhir starts. Gate 0 passes only when at least four
of five seeds complete at least 90% of those laps without collision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.common import (
    arc_length_spawn_indices,
    make_env_for_track,
    run_eval_episode,
    summarize_episodes,
)
from utils.provenance import collect_provenance, utc_now_iso, write_json


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iqm(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if values.size == 0:
        return float("nan")
    lower = int(np.floor(0.25 * values.size))
    upper = int(np.ceil(0.75 * values.size))
    return float(np.mean(values[lower:upper]))


def _stratified_bootstrap(seed_outcomes: List[np.ndarray], n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n_seeds = len(seed_outcomes)
    aggregate = np.empty(n_boot, dtype=float)
    iqm = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        selected = rng.integers(0, n_seeds, size=n_seeds)
        per_seed = []
        for seed_idx in selected:
            outcomes = seed_outcomes[int(seed_idx)]
            sampled = outcomes[rng.integers(0, outcomes.size, size=outcomes.size)]
            per_seed.append(float(np.mean(sampled)))
        aggregate[b] = float(np.mean(per_seed))
        iqm[b] = _iqm(np.asarray(per_seed))
    return {
        "mean_completion_ci95": np.quantile(aggregate, [0.025, 0.975]).tolist(),
        "iqm_completion_ci95": np.quantile(iqm, [0.025, 0.975]).tolist(),
    }


def _resolve_checkpoint(run_dir: Path, requested: str) -> Path:
    if requested == "best_validation":
        path = run_dir / "eval_results" / "best_validation_model.zip"
    elif requested == "final":
        path = run_dir / "sac_final.zip"
    else:
        path = Path(requested)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="Five checkpoint run directories")
    parser.add_argument("--track", default="Sakhir")
    parser.add_argument("--starts", type=int, default=50)
    parser.add_argument("--checkpoint", default="best_validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="experiments/gate0/gate0_result.json")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260814)
    args = parser.parse_args()

    if len(args.run_dirs) != 5:
        raise ValueError(f"Gate 0 requires exactly five trained seeds; got {len(args.run_dirs)}.")
    if args.starts < 50:
        raise ValueError("Gate 0 requires at least 50 fixed starts per seed.")

    run_results = []
    seed_outcomes: List[np.ndarray] = []
    seen_seeds = set()
    reference_spawn_indices = None

    for raw_run_dir in args.run_dirs:
        run_dir = Path(raw_run_dir).resolve()
        meta = _load_json(run_dir / "run_meta.json")
        if meta.get("status") != "complete":
            raise ValueError(f"Run is not complete: {run_dir}")
        if meta.get("action_space") != "steer_speed":
            raise ValueError(f"Gate 0 only accepts steer_speed runs: {run_dir}")
        seed = int(meta["seed"])
        if seed in seen_seeds:
            raise ValueError(f"Duplicate training seed {seed}.")
        seen_seeds.add(seed)

        vehicle_cfg = yaml.safe_load((run_dir / "resolved_vehicle.yaml").read_text(encoding="utf-8"))
        env = make_env_for_track(vehicle_cfg, args.track, render_mode=None)
        checkpoint = _resolve_checkpoint(run_dir, args.checkpoint)
        model = SAC.load(str(checkpoint), env=env, device=args.device)
        spawn_indices = arc_length_spawn_indices(env.centerline, args.starts)
        if reference_spawn_indices is None:
            reference_spawn_indices = spawn_indices
        elif spawn_indices != reference_spawn_indices:
            raise RuntimeError("Fixed-start construction changed between runs.")

        episodes = []
        try:
            for episode_idx, spawn_idx in enumerate(spawn_indices):
                episodes.append(
                    run_eval_episode(
                        model,
                        env,
                        seed=args.bootstrap_seed + episode_idx,
                        spawn_idx=spawn_idx,
                        deterministic=True,
                    )
                )
        finally:
            env.close()

        outcomes = np.asarray(
            [episode.term_reason == "lap_complete" for episode in episodes], dtype=float
        )
        seed_outcomes.append(outcomes)
        summary = summarize_episodes(episodes)
        run_results.append(
            {
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "seed": seed,
                "summary": summary,
                "passes_seed_threshold": bool(summary["completion_rate"] >= 0.90),
                "episodes": [vars(episode) for episode in episodes],
            }
        )
        print(
            f"seed={seed} completion={summary['completion_rate']:.3f} "
            f"crash={summary['crash_rate']:.3f}"
        )

    per_seed_completion = np.asarray([float(np.mean(x)) for x in seed_outcomes])
    passing_seeds = int(np.sum(per_seed_completion >= 0.90))
    bootstrap = _stratified_bootstrap(
        seed_outcomes, args.bootstrap_samples, args.bootstrap_seed
    )
    result = {
        "gate": "Gate 0 direct-action SAC functionality",
        "generated_at_utc": utc_now_iso(),
        "criterion": {
            "required_passing_seeds": 4,
            "total_seeds": 5,
            "minimum_seed_completion_rate": 0.90,
            "fixed_starts_per_seed": args.starts,
            "track": args.track,
            "checkpoint_rule": args.checkpoint,
        },
        "passed": bool(passing_seeds >= 4),
        "passing_seeds": passing_seeds,
        "mean_completion_rate": float(np.mean(per_seed_completion)),
        "iqm_completion_rate": _iqm(per_seed_completion),
        **bootstrap,
        "spawn_indices": reference_spawn_indices,
        "runs": run_results,
        "provenance": collect_provenance(ROOT),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(f"Gate 0 {'PASS' if result['passed'] else 'FAIL'}: {passing_seeds}/5 seeds")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
