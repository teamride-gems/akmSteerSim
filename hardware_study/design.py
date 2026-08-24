"""Deterministic source selection and blinded crossover construction."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise
from typing import Iterable

import numpy as np

from utils.innovation_gate import (
    phase_shift_noninitial_events,
    replay_accepted_values_at_events,
)


SEMANTIC_CONDITIONS = (
    "clean_a",
    "clean_b",
    "direct",
    "innovation_gate",
    "timing_placebo",
)


def condition_sequences(source: dict, shift: int = 2) -> dict[str, np.ndarray]:
    events = np.asarray(source["innovation_events"], dtype=bool)
    gated = np.asarray(source["innovation_output"], dtype=float)
    shifted = phase_shift_noninitial_events(events, int(shift))
    timing = replay_accepted_values_at_events(gated[events], shifted)
    clean = np.asarray(source["clean_actions"], dtype=float)[:, 0]
    return {
        "clean_a": clean.copy(),
        "clean_b": clean.copy(),
        "direct": np.asarray(source["regime_input"], dtype=float),
        "innovation_gate": gated,
        "timing_placebo": timing,
    }


def source_safety_metrics(source: dict) -> dict:
    sequences = condition_sequences(source)
    maximum_abs = max(float(np.max(np.abs(values))) for values in sequences.values())
    maximum_step = max(
        float(np.max(np.abs(np.diff(values)))) if values.size > 1 else 0.0
        for values in sequences.values()
    )
    original = np.asarray(source["innovation_events"], dtype=bool)
    shifted = phase_shift_noninitial_events(original, 2)
    return {
        "maximum_abs_normalized_steering": maximum_abs,
        "maximum_packet_step_normalized_steering": maximum_step,
        "timing_nonidentity": not np.array_equal(original, shifted),
    }


def _even_rank_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length < count:
        raise ValueError("cannot select the requested number of evenly spaced ranks")
    indices = np.rint(np.linspace(0, length - 1, count)).astype(int).tolist()
    if len(set(indices)) != count:
        raise RuntimeError("even-rank selection produced duplicate indices")
    return indices


def select_sources(rows: Iterable[dict], config: dict) -> list[dict]:
    selection = config["source_selection"]
    chosen = []
    for checkpoint in selection["checkpoints"]:
        eligible = []
        for source in rows:
            if source.get("regime") != selection["eligible_regime"]:
                continue
            if source.get("checkpoint") != checkpoint:
                continue
            if selection["require_survived"] and not bool(source.get("survived")):
                continue
            if selection["require_reference_not_terminated"] and bool(
                source.get("reference_terminated")
            ):
                continue
            safety = source_safety_metrics(source)
            if selection["require_nonidentity_timing_placebo"] and not safety[
                "timing_nonidentity"
            ]:
                continue
            if safety["maximum_abs_normalized_steering"] > float(
                selection["maximum_abs_normalized_steering"]
            ):
                continue
            if safety["maximum_packet_step_normalized_steering"] > float(
                selection["maximum_packet_step_normalized_steering"]
            ):
                continue
            eligible.append({**source, "hardware_safety": safety})
        eligible.sort(key=lambda row: (float(row["suppression_fraction"]), int(row["spawn"])))
        indices = _even_rank_indices(
            len(eligible), int(selection["sources_per_checkpoint"])
        )
        for rank_index in indices:
            item = eligible[rank_index]
            chosen.append(
                {
                    **item,
                    "eligible_rank": rank_index,
                    "eligible_count_checkpoint": len(eligible),
                }
            )
    expected = len(selection["checkpoints"]) * int(selection["sources_per_checkpoint"])
    if len(chosen) != expected:
        raise RuntimeError(f"selected {len(chosen)} sources, expected {expected}")
    return chosen


def williams_sequences_five() -> list[tuple[int, ...]]:
    base = (0, 1, 4, 2, 3)
    rotations = [tuple((value + offset) % 5 for value in base) for offset in range(5)]
    return rotations + [tuple(reversed(sequence)) for sequence in rotations]


def make_condition_key(seed: int) -> dict[str, str]:
    generator = np.random.default_rng(int(seed))
    codes = np.asarray(list("ABCDE"))[generator.permutation(5)].tolist()
    return dict(zip(codes, SEMANTIC_CONDITIONS))


def make_schedule(selected: list[dict], config: dict) -> tuple[list[dict], dict[str, str]]:
    randomization = config["randomization"]
    generator = np.random.default_rng(int(randomization["seed"]))
    key = make_condition_key(int(randomization["seed"]))
    semantic_to_code = {semantic: code for code, semantic in key.items()}
    templates = williams_sequences_five()
    block_specs = [
        (float(speed), source)
        for speed in config["command_bundle"]["speed_caps_mps"]
        for source in selected
    ]
    if len(block_specs) != int(randomization["total_blocks"]):
        raise RuntimeError("block count disagrees with the frozen design")
    generator.shuffle(block_specs)
    template_order = []
    while len(template_order) < len(block_specs):
        permutation = generator.permutation(len(templates)).tolist()
        template_order.extend(permutation)
    rows = []
    for sequence_index, ((speed, source), template_index) in enumerate(
        zip(block_specs, template_order), 1
    ):
        checkpoint = source["checkpoint"]
        spawn = int(source["spawn"])
        speed_token = str(speed).replace(".", "p")
        bundle_id = f"{checkpoint}_spawn{spawn}_speed{speed_token}"
        block_id = f"B{sequence_index:02d}_{checkpoint}_S{spawn}_V{speed_token}"
        condition_indices = templates[int(template_index)]
        semantic_order = [SEMANTIC_CONDITIONS[index] for index in condition_indices]
        for position, semantic in enumerate(semantic_order, 1):
            rows.append(
                {
                    "run_index": len(rows) + 1,
                    "run_id": f"HW{len(rows) + 1:03d}",
                    "block_index": sequence_index,
                    "block_id": block_id,
                    "run_position": position,
                    "checkpoint": checkpoint,
                    "spawn": spawn,
                    "speed_cap_mps": speed,
                    "bundle_id": bundle_id,
                    "condition_code": semantic_to_code[semantic],
                    "condition": semantic,
                    "template_index": int(template_index),
                }
            )
    return rows, key


def balance_summary(schedule: list[dict]) -> dict:
    position_counts = Counter(
        (row["condition_code"], int(row["run_position"])) for row in schedule
    )
    transition_counts = Counter()
    for block in sorted({row["block_id"] for row in schedule}):
        ordered = sorted(
            (row for row in schedule if row["block_id"] == block),
            key=lambda row: int(row["run_position"]),
        )
        for left, right in pairwise(row["condition_code"] for row in ordered):
            transition_counts[(left, right)] += 1
    return {
        "position_counts": {
            f"{condition}/position_{position}": count
            for (condition, position), count in sorted(position_counts.items())
        },
        "transition_counts": {
            f"{left}_to_{right}": count
            for (left, right), count in sorted(transition_counts.items())
        },
        "position_count_range": [min(position_counts.values()), max(position_counts.values())],
        "transition_count_range": [
            min(transition_counts.values()),
            max(transition_counts.values()),
        ],
        "runs": len(schedule),
        "blocks": len({row["block_id"] for row in schedule}),
    }
