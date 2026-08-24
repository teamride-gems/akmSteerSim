"""Frozen block-paired analysis for the physical transport study."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .integrity import verify_hash_chain, write_json


def _local_trajectory(run_dir: Path) -> np.ndarray:
    chain = verify_hash_chain(run_dir / "records.jsonl")
    if not chain["passed"]:
        raise RuntimeError(f"hash chain failed for {run_dir}: {chain['errors']}")
    records = chain["records"]
    starts = [row for row in records if row.get("record_type") == "run_start"]
    if len(starts) != 1:
        raise RuntimeError(f"expected one run-start record in {run_dir}")
    initial = starts[0]["initial_telemetry"]
    commands = [
        row
        for row in records
        if row.get("record_type") == "command" and row.get("phase") == "main"
    ]
    if len(commands) != 41:
        raise RuntimeError(f"expected 41 main commands in {run_dir}")
    grid = np.asarray(
        [float(row["planned_send_monotonic_s"]) for row in commands], dtype=float
    )
    telemetry = [
        row["telemetry"]
        for row in records
        if row.get("record_type") == "telemetry"
        and all(
            row["telemetry"].get(key) is not None
            and math.isfinite(float(row["telemetry"][key]))
            for key in ("received_monotonic_s", "x_m", "y_m")
        )
    ]
    telemetry.sort(key=lambda row: float(row["received_monotonic_s"]))
    times = np.asarray([float(row["received_monotonic_s"]) for row in telemetry])
    x = np.asarray([float(row["x_m"]) for row in telemetry])
    y = np.asarray([float(row["y_m"]) for row in telemetry])
    # Duplicate timestamps carry no extra interpolation information.
    unique_times, indices = np.unique(times, return_index=True)
    if unique_times.size < 2:
        raise RuntimeError(f"insufficient telemetry times in {run_dir}")
    world = np.column_stack(
        [
            np.interp(grid, unique_times, x[indices]),
            np.interp(grid, unique_times, y[indices]),
        ]
    )
    offset = world - np.asarray([float(initial["x_m"]), float(initial["y_m"])])
    yaw = float(initial["yaw_rad"])
    rotation = np.asarray(
        [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]
    )
    return offset @ rotation.T


def _rms_path(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left) - np.asarray(right)
    return float(np.sqrt(np.mean(np.sum(difference**2, axis=1))))


def _retry_code(manifest: dict) -> str | None:
    reason = str(manifest.get("abort_reason") or "")
    if "start_pose_out_of_tolerance" in reason or "start_heading_out_of_tolerance" in reason:
        return "start_pose_out_of_tolerance_before_arm"
    if "adapter_not_ready" in reason or "telemetry_stale" in reason:
        return "missing_or_stale_telemetry_before_first_motion"
    return None


def _attempts_for_run(run_container: Path) -> list[Path]:
    attempts = sorted(run_container.glob("attempt_*")) if run_container.is_dir() else []
    if not attempts and (run_container / "run_manifest.json").is_file():
        attempts = [run_container]
    return attempts


def _select_outcome(run_container: Path, config: dict) -> tuple[Path | None, list[str]]:
    attempts = _attempts_for_run(run_container)
    errors = []
    if not attempts:
        return None, ["missing_run_archive"]
    outcome_attempts = []
    for index, attempt in enumerate(attempts):
        manifest_path = attempt / "run_manifest.json"
        validation_path = attempt / "validation.json"
        launch_failure_path = attempt / "launch_failure.json"
        if launch_failure_path.is_file() and not manifest_path.is_file():
            launch_failure = json.loads(
                launch_failure_path.read_text(encoding="utf-8")
            )
            allowed = config["run_validity"]["allowed_pre_motion_technical_reruns"]
            if (
                launch_failure.get("motion_started")
                or launch_failure.get("retry_code") not in allowed
            ):
                errors.append(f"attempt_{index + 1}_unapproved_launch_failure")
            continue
        if not manifest_path.is_file() or not validation_path.is_file():
            errors.append(f"attempt_{index + 1}_missing_manifest_or_validation")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("completed") or manifest.get("motion_started"):
            outcome_attempts.append(attempt)
        else:
            code = _retry_code(manifest)
            allowed = config["run_validity"]["allowed_pre_motion_technical_reruns"]
            if code not in allowed:
                errors.append(f"attempt_{index + 1}_unapproved_pre_motion_exclusion")
    if len(outcome_attempts) != 1:
        errors.append(f"expected_one_outcome_attempt_found_{len(outcome_attempts)}")
        return None, errors
    selected = outcome_attempts[0]
    if attempts.index(selected) != len(attempts) - 1:
        errors.append("attempt_exists_after_outcome")
    validation = json.loads((selected / "validation.json").read_text(encoding="utf-8"))
    if not validation.get("eligible_outcome"):
        errors.append("selected_attempt_is_not_an_eligible_outcome")
    return selected, errors


def _cell_key(row: dict) -> str:
    return f"{row['checkpoint']}|{float(row['speed_cap_mps']):.1f}"


def _stratified_bootstrap(
    block_rows: list[dict], field: str, draws: int, seed: int
) -> dict:
    by_cell = defaultdict(list)
    for row in block_rows:
        by_cell[row["cell"]].append(float(row[field]))
    if len(by_cell) != 4 or any(not values for values in by_cell.values()):
        raise RuntimeError("the frozen analysis requires four nonempty speed/checkpoint cells")
    cell_means = {key: float(np.mean(values)) for key, values in by_cell.items()}
    estimate = float(np.mean(list(cell_means.values())))
    generator = np.random.default_rng(int(seed))
    samples = np.empty(int(draws), dtype=float)
    ordered_cells = sorted(by_cell)
    for draw in range(int(draws)):
        draw_means = []
        for cell in ordered_cells:
            values = np.asarray(by_cell[cell], dtype=float)
            indices = generator.integers(0, values.size, size=values.size)
            draw_means.append(float(np.mean(values[indices])))
        samples[draw] = float(np.mean(draw_means))
    return {
        "estimate_m": estimate,
        "bootstrap_lower_95_m": float(np.percentile(samples, 2.5)),
        "bootstrap_upper_95_m": float(np.percentile(samples, 97.5)),
        "cell_means_m": cell_means,
        "positive_in_all_cells": all(value > 0.0 for value in cell_means.values()),
    }


def analyze_study(
    output_root: Path,
    prepared_dir: Path,
    config: dict,
) -> dict:
    output_root = Path(output_root)
    prepared_dir = Path(prepared_dir)
    schedule = json.loads(
        (prepared_dir / "machine_schedule.json").read_text(encoding="utf-8")
    )
    invalid_reasons = []
    selected_attempts = {}
    adapter_names = set()
    for row in schedule:
        attempt, errors = _select_outcome(output_root / row["run_id"], config)
        invalid_reasons.extend(f"{row['run_id']}:{error}" for error in errors)
        if attempt is not None:
            selected_attempts[row["run_id"]] = attempt
            manifest = json.loads(
                (attempt / "run_manifest.json").read_text(encoding="utf-8")
            )
            adapter_names.add(manifest["adapter"])
    expected_ids = {row["run_id"] for row in schedule}
    observed_ids = {path.name for path in output_root.iterdir()} if output_root.is_dir() else set()
    extras = sorted(observed_ids - expected_ids)
    if extras:
        invalid_reasons.append(f"unexpected_run_directories:{extras}")
    if len(selected_attempts) != len(schedule):
        invalid_reasons.append(
            f"incomplete_schedule:{len(selected_attempts)}_of_{len(schedule)}"
        )

    trajectories = {}
    failures = set()
    if not invalid_reasons:
        for row in schedule:
            attempt = selected_attempts[row["run_id"]]
            manifest = json.loads(
                (attempt / "run_manifest.json").read_text(encoding="utf-8")
            )
            if manifest["completed"]:
                trajectories[row["run_id"]] = _local_trajectory(attempt)
            elif manifest["motion_started"]:
                failures.add(row["run_id"])
            else:
                invalid_reasons.append(f"{row['run_id']}:no_scientific_outcome")

    block_rows = []
    clean_repeats = []
    failure_fill = float(config["analysis"]["failure_fill_path_error_m"])
    if not invalid_reasons:
        for block_id in sorted({row["block_id"] for row in schedule}):
            rows = [row for row in schedule if row["block_id"] == block_id]
            by_condition = {row["condition"]: row for row in rows}
            if set(by_condition) != set(config["command_bundle"]["conditions"]):
                invalid_reasons.append(f"{block_id}:condition_set_mismatch")
                continue
            clean_ids = [by_condition[name]["run_id"] for name in ("clean_a", "clean_b")]
            if any(run_id in failures for run_id in clean_ids):
                invalid_reasons.append(f"{block_id}:clean_reference_failed_after_motion")
                continue
            clean_a = trajectories[clean_ids[0]]
            clean_b = trajectories[clean_ids[1]]
            clean_repeat = _rms_path(clean_a, clean_b)
            clean_repeats.append(clean_repeat)
            reference = 0.5 * (clean_a + clean_b)

            def candidate_error(condition: str) -> float:
                run_id = by_condition[condition]["run_id"]
                if run_id in failures:
                    return failure_fill
                return _rms_path(trajectories[run_id], reference)

            direct = candidate_error("direct")
            gate = candidate_error("innovation_gate")
            placebo = candidate_error("timing_placebo")
            base = rows[0]
            block_rows.append(
                {
                    "block_id": block_id,
                    "checkpoint": base["checkpoint"],
                    "speed_cap_mps": float(base["speed_cap_mps"]),
                    "cell": _cell_key(base),
                    "clean_repeat_rms_m": clean_repeat,
                    "direct_error_m": direct,
                    "innovation_gate_error_m": gate,
                    "timing_placebo_error_m": placebo,
                    "specificity_placebo_minus_gate_m": placebo - gate,
                    "downstream_harm_gate_minus_direct_m": gate - direct,
                }
            )

    analysis_config = config["analysis"]
    clean = None
    specificity = None
    harm = None
    if not invalid_reasons:
        clean_array = np.asarray(clean_repeats, dtype=float)
        clean = {
            "median_rms_m": float(np.median(clean_array)),
            "p95_rms_m": float(np.percentile(clean_array, 95)),
            "median_threshold_m": float(
                analysis_config["clean_repeat_median_rms_max_m"]
            ),
            "p95_threshold_m": float(analysis_config["clean_repeat_p95_rms_max_m"]),
        }
        clean["passed"] = (
            clean["median_rms_m"] <= clean["median_threshold_m"]
            and clean["p95_rms_m"] <= clean["p95_threshold_m"]
        )
        specificity = _stratified_bootstrap(
            block_rows,
            "specificity_placebo_minus_gate_m",
            int(analysis_config["bootstrap_draws"]),
            int(analysis_config["bootstrap_seed"]),
        )
        specificity["effect_threshold_m"] = float(
            analysis_config["specificity_placebo_minus_gate_effect_min_m"]
        )
        specificity["passed"] = (
            specificity["estimate_m"] >= specificity["effect_threshold_m"]
            and specificity["bootstrap_lower_95_m"]
            >= float(analysis_config["specificity_bootstrap_lower_min_m"])
            and (
                specificity["positive_in_all_cells"]
                or not analysis_config["require_positive_each_speed_checkpoint_cell"]
            )
        )
        harm = _stratified_bootstrap(
            block_rows,
            "downstream_harm_gate_minus_direct_m",
            int(analysis_config["bootstrap_draws"]),
            int(analysis_config["bootstrap_seed"]) + 1,
        )
        harm["effect_threshold_m"] = float(
            analysis_config["utility_gate_minus_direct_harm_min_m"]
        )
        harm["passed"] = (
            harm["estimate_m"] >= harm["effect_threshold_m"]
            and harm["bootstrap_lower_95_m"]
            >= float(analysis_config["utility_bootstrap_lower_min_m"])
            and (
                harm["positive_in_all_cells"]
                or not analysis_config["require_positive_each_speed_checkpoint_cell"]
            )
        )
        if not clean["passed"]:
            invalid_reasons.append("clean_repeatability_failed")

    if invalid_reasons:
        verdict = "INVALID"
    elif specificity["passed"] and harm["passed"]:
        verdict = "REPRODUCED_REVERSAL"
    elif specificity["passed"]:
        verdict = "SPECIFICITY_ONLY"
    elif harm["passed"]:
        verdict = "DOWNSTREAM_ONLY"
    else:
        verdict = "NOT_REPRODUCED"
    evidence_class = (
        "PHYSICAL_HARDWARE" if adapter_names == {"ros2_ackermann"} else "ENGINEERING_MOCK_ONLY"
    )
    return {
        "schema_version": 1,
        "study_id": config["study_id"],
        "verdict": verdict,
        "evidence_class": evidence_class,
        "invalid_reasons": invalid_reasons,
        "scheduled_runs": len(schedule),
        "selected_outcomes": len(selected_attempts),
        "post_motion_failures": len(failures),
        "adapter_names": sorted(adapter_names),
        "clean_repeatability": clean,
        "specificity": specificity,
        "downstream_harm": harm,
        "blocks": block_rows,
    }


def write_analysis(output_dir: Path, result: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "result.json", result)
    lines = [
        "# Frozen hardware-study analysis",
        "",
        f"**Verdict: {result['verdict']}**",
        "",
        f"- Evidence class: `{result['evidence_class']}`",
        f"- Selected outcomes: `{result['selected_outcomes']} / {result['scheduled_runs']}`",
        f"- Post-motion failures: `{result['post_motion_failures']}`",
    ]
    if result["invalid_reasons"]:
        lines.extend(["", "## Invalidity reasons", ""])
        lines.extend(f"- {reason}" for reason in result["invalid_reasons"])
    else:
        clean = result["clean_repeatability"]
        specificity = result["specificity"]
        harm = result["downstream_harm"]
        lines.extend(
            [
                f"- Clean-repeat median / p95: `{clean['median_rms_m']:.4f} / {clean['p95_rms_m']:.4f} m`",
                f"- Placebo minus gate: `{specificity['estimate_m']:.4f} m` (95% bootstrap `{specificity['bootstrap_lower_95_m']:.4f}, {specificity['bootstrap_upper_95_m']:.4f}`)",
                f"- Gate minus direct: `{harm['estimate_m']:.4f} m` (95% bootstrap `{harm['bootstrap_lower_95_m']:.4f}, {harm['bootstrap_upper_95_m']:.4f}`)",
            ]
        )
    lines.extend(
        [
            "",
            "Mock results certify only pipeline behavior; they are never physical scientific evidence.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
