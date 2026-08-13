#!/usr/bin/env python3
"""Retired experiment entry point.

This file intentionally cannot launch experiments. Its former implementation
mixed obsolete checkpoint and metric schemas and could associate stale outputs
with a new condition.
"""

raise SystemExit(
    "run_experiment.py is retired because its artifact semantics are unsafe. "
    "Use scripts/run_one_experiment.py for one condition, scripts/sweep.py for "
    "paper-scale sweeps, or scripts/run_repro_baseline.py for the Rung 2 baseline."
)
