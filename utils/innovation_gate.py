"""Pure helpers for the prospective Innovation-Gated Steering study."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


def zero_mean_clipped_gaussian(
    seed: int,
    length: int,
    standard_deviation: float,
    max_abs: float,
) -> np.ndarray:
    """Generate a deterministic bounded schedule with exactly zero mean."""
    if length < 2:
        raise ValueError("length must be at least two")
    if standard_deviation <= 0.0 or max_abs <= 0.0:
        raise ValueError("standard_deviation and max_abs must be positive")
    rng = np.random.default_rng(int(seed))
    values = rng.normal(0.0, float(standard_deviation), size=int(length))
    values -= float(np.mean(values))
    peak = float(np.max(np.abs(values)))
    if peak > float(max_abs):
        values *= float(max_abs) / peak
    # Remove the final floating-point residual without violating the bound.
    values[-1] -= float(np.sum(values))
    if float(np.max(np.abs(values))) > float(max_abs) + 1e-12:
        raise RuntimeError("zero-mean correction violated the jitter bound")
    return values.astype(float)


def innovation_gate(values: Sequence[float], threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Hold the last accepted value until the innovation reaches threshold."""
    source = np.asarray(values, dtype=float).reshape(-1)
    if source.size == 0:
        raise ValueError("values must be nonempty")
    if not np.all(np.isfinite(source)):
        raise ValueError("values must be finite")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    output = np.empty_like(source)
    events = np.zeros(source.size, dtype=bool)
    output[0] = source[0]
    events[0] = True
    held = float(source[0])
    for index in range(1, source.size):
        if abs(float(source[index]) - held) >= float(threshold):
            held = float(source[index])
            events[index] = True
        output[index] = held
    return output, events


def phase_shift_noninitial_events(events: Sequence[bool], shift: int) -> np.ndarray:
    """Circularly shift noninitial event timing while preserving cardinality."""
    mask = np.asarray(events, dtype=bool).reshape(-1)
    if mask.size == 0 or not bool(mask[0]):
        raise ValueError("events must be nonempty and accept the initial command")
    if shift == 0:
        raise ValueError("phase-placebo shift must be nonzero")
    shifted = mask.copy()
    if mask.size > 1:
        shifted[1:] = np.roll(mask[1:], int(shift))
    if int(np.sum(shifted)) != int(np.sum(mask)):
        raise RuntimeError("phase shift changed update cardinality")
    return shifted


def replay_event_mask(values: Sequence[float], events: Sequence[bool]) -> np.ndarray:
    """Accept contemporaneous values at fixed event times and hold otherwise."""
    source = np.asarray(values, dtype=float).reshape(-1)
    mask = np.asarray(events, dtype=bool).reshape(-1)
    if source.shape != mask.shape or source.size == 0:
        raise ValueError("values and events must be nonempty with identical shape")
    if not bool(mask[0]):
        raise ValueError("the initial command must be accepted")
    output = np.empty_like(source)
    held = float(source[0])
    for index in range(source.size):
        if bool(mask[index]):
            held = float(source[index])
        output[index] = held
    return output


def replay_accepted_values_at_events(
    accepted_values: Sequence[float],
    events: Sequence[bool],
) -> np.ndarray:
    """Replay an accepted-value sequence at a new event schedule.

    The first event must occur at index zero.  Unlike :func:`replay_event_mask`,
    this helper does not sample contemporaneous source commands.  It preserves
    the complete ordered sequence of accepted targets (and therefore their
    increments) while changing only the times at which those targets arrive.
    """
    targets = np.asarray(accepted_values, dtype=float).reshape(-1)
    mask = np.asarray(events, dtype=bool).reshape(-1)
    if targets.size == 0 or mask.size == 0:
        raise ValueError("accepted values and events must be nonempty")
    if not np.all(np.isfinite(targets)):
        raise ValueError("accepted values must be finite")
    if not bool(mask[0]):
        raise ValueError("the initial command must be accepted")
    if int(np.sum(mask)) != int(targets.size):
        raise ValueError("accepted-value count must equal event count")
    output = np.empty(mask.size, dtype=float)
    target_index = 0
    held = float(targets[0])
    for index, is_event in enumerate(mask):
        if bool(is_event):
            held = float(targets[target_index])
            target_index += 1
        output[index] = held
    if target_index != targets.size:
        raise RuntimeError("not all accepted values were replayed")
    return output


def relative_improvement(baseline: float, candidate: float, floor: float = 1e-9) -> float:
    return float((float(baseline) - float(candidate)) / max(abs(float(baseline)), floor))


def stratified_paired_bootstrap(
    values_by_checkpoint: Mapping[str, Sequence[float]],
    draws: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap the checkpoint-balanced mean of already paired effects."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    arrays = {
        str(key): np.asarray(value, dtype=float).reshape(-1)
        for key, value in values_by_checkpoint.items()
    }
    if not arrays or any(value.size == 0 for value in arrays.values()):
        raise ValueError("every checkpoint stratum must contain observations")
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("bootstrap values must be finite")
    rng = np.random.default_rng(int(seed))
    output = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        stratum_means = []
        for values in arrays.values():
            selected = rng.integers(0, values.size, size=values.size)
            stratum_means.append(float(np.mean(values[selected])))
        output[draw] = float(np.mean(stratum_means))
    return output


def earliest_ladder_verdict(
    g0: Mapping[str, bool],
    g1: Mapping[str, bool],
    g2: Mapping[str, bool],
    g3: Mapping[str, bool],
    g4: Mapping[str, bool],
) -> str:
    if not all(bool(value) for value in g0.values()):
        return "INVALID_G0"
    for rung, checks in (("G1", g1), ("G2", g2), ("G3", g3), ("G4", g4)):
        if not all(bool(value) for value in checks.values()):
            return f"KILL_AT_{rung}"
    return "PASS_TO_SMALL_TRAINING"


def percentile_interval(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    lower, upper = np.percentile(array, [2.5, 97.5])
    return float(lower), float(upper)
