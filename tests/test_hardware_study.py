import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import yaml

from hardware_study.adapters import MockAdapter
from hardware_study.analysis import _stratified_bootstrap
from hardware_study.bundles import load_bundle_from_manifest
from hardware_study.design import balance_summary, condition_sequences, select_sources
from hardware_study.execution import execute_run
from hardware_study.integrity import HashChainWriter, sha256_file, verify_hash_chain, write_json
from hardware_study.pilot import engineering_pilot_bundle
from hardware_study.safety import SafetyLimiter
from hardware_study.validation import validate_run
from scripts.run_hardware_study import next_expected_run, retry_code_for_attempt


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/hardware_study_v1.yaml"
SITE_PATH = ROOT / "configs/hardware_site_mock.yaml"
PREPARED = ROOT / "reproducibility/hardware_validation/study_v1/prepared"


class HardwareStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.site = yaml.safe_load(SITE_PATH.read_text(encoding="utf-8"))
        cls.schedule = json.loads(
            (PREPARED / "machine_schedule.json").read_text(encoding="utf-8")
        )

    def test_frozen_source_selection_is_deterministic(self):
        source_path = ROOT / self.config["sealed_sources"]["source_sequences"]["path"]
        sources = json.loads(source_path.read_text(encoding="utf-8"))
        selected = select_sources(sources, self.config)
        observed = {
            checkpoint: [
                int(row["spawn"])
                for row in selected
                if row["checkpoint"] == checkpoint
            ]
            for checkpoint in self.config["source_selection"]["checkpoints"]
        }
        self.assertEqual(observed["direct_s0"], [292, 11, 76, 855, 574, 1006])
        self.assertEqual(observed["direct_s4"], [638, 725, 574, 76, 855, 1006])

    def test_gate_and_placebo_match_targets_and_increments(self):
        source_path = ROOT / self.config["sealed_sources"]["source_sequences"]["path"]
        sources = json.loads(source_path.read_text(encoding="utf-8"))
        for source in select_sources(sources, self.config):
            sequences = condition_sequences(source)
            gate = sequences["innovation_gate"]
            placebo = sequences["timing_placebo"]
            gate_events = np.asarray(source["innovation_events"], dtype=bool)
            placebo_events = np.r_[True, np.diff(placebo) != 0.0]
            np.testing.assert_array_equal(gate[gate_events], placebo[placebo_events])
            np.testing.assert_array_equal(
                np.diff(gate[gate_events]), np.diff(placebo[placebo_events])
            )

    def test_schedule_is_complete_and_balanced(self):
        self.assertEqual(len(self.schedule), 120)
        self.assertEqual(len({row["run_id"] for row in self.schedule}), 120)
        self.assertEqual(len({row["block_id"] for row in self.schedule}), 24)
        expected = set(self.config["command_bundle"]["conditions"])
        for block in {row["block_id"] for row in self.schedule}:
            self.assertEqual(
                {row["condition"] for row in self.schedule if row["block_id"] == block},
                expected,
            )
        balance = balance_summary(self.schedule)
        self.assertGreaterEqual(balance["position_count_range"][0], 4)
        self.assertLessEqual(balance["position_count_range"][1], 6)
        self.assertGreaterEqual(balance["transition_count_range"][0], 4)
        self.assertLessEqual(balance["transition_count_range"][1], 6)

    def test_all_bundle_hashes_and_matching_checks(self):
        manifest = json.loads(
            (PREPARED / "bundle_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["bundle_count"], 24)
        for entry in manifest["bundles"]:
            bundle, loaded_entry = load_bundle_from_manifest(PREPARED, entry["bundle_id"])
            self.assertEqual(loaded_entry, entry)
            self.assertTrue(all(bundle["matching_checks"].values()))

    def test_safety_limiter_enforces_all_command_bounds(self):
        limiter = SafetyLimiter(1.5, 0.26, 3.5, 2.0)
        sent = limiter.apply(10.0, 10.0, 0.05)
        self.assertAlmostEqual(sent["steering_rad"], 0.175)
        self.assertAlmostEqual(sent["speed_mps"], 0.1)
        self.assertTrue(sent["target_steering_clipped"])
        self.assertTrue(sent["target_speed_clipped"])

    def test_hash_chain_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            with HashChainWriter(path) as writer:
                writer.write({"value": 1})
                writer.write({"value": 2})
            self.assertTrue(verify_hash_chain(path)["passed"])
            text = path.read_text(encoding="utf-8").replace('"value":2', '"value":3')
            path.write_text(text, encoding="utf-8")
            self.assertFalse(verify_hash_chain(path)["passed"])

    def test_mock_run_executes_and_passes_strict_validation(self):
        row = self.schedule[0]
        bundle, entry = load_bundle_from_manifest(PREPARED, row["bundle_id"])
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            freeze_path = Path(temporary) / "FREEZE.json"
            write_json(freeze_path, {"test_freeze": True})
            static = {
                "config_sha256": sha256_file(CONFIG_PATH),
                "site_sha256": sha256_file(SITE_PATH),
                "prepared_manifest_sha256": sha256_file(PREPARED / "PREPARED_MANIFEST.json"),
                "machine_schedule_sha256": sha256_file(PREPARED / "machine_schedule.json"),
                "bundle_sha256": entry["sha256"],
                "freeze_sha256": sha256_file(freeze_path),
                "runner_sha256": "0" * 64,
            }
            manifest = execute_run(
                MockAdapter(), bundle, row, self.config, self.site, run_dir, static
            )
            archive = run_dir / "frozen_inputs"
            archive.mkdir()
            shutil.copy2(CONFIG_PATH, archive / "hardware_study_v1.yaml")
            shutil.copy2(SITE_PATH, archive / "hardware_site.yaml")
            shutil.copy2(PREPARED / "PREPARED_MANIFEST.json", archive / "PREPARED_MANIFEST.json")
            shutil.copy2(freeze_path, archive / "FREEZE.json")
            write_json(archive / "schedule_row.json", row)
            write_json(archive / "command_bundle.json", bundle)
            result = validate_run(run_dir, self.config, bundle, row)
            self.assertTrue(manifest["completed"])
            self.assertTrue(result["technical_valid"], result)
            self.assertTrue(result["eligible_outcome"])

    def test_synthetic_pilots_are_bounded_and_not_study_conditions(self):
        for mode, maximum_speed in (("stands", 0.2), ("ground", 0.5)):
            bundle = engineering_pilot_bundle(mode, self.config)
            packets = bundle["conditions"]["engineering_pilot"]
            self.assertTrue(bundle["engineering_only"])
            self.assertEqual(len(packets), 81)
            self.assertLessEqual(max(row["target_speed_mps"] for row in packets), maximum_speed)
            self.assertLessEqual(max(abs(row["target_steering_rad"]) for row in packets), 0.05)
            self.assertNotIn("engineering_pilot", self.config["command_bundle"]["conditions"])

    def test_stratified_bootstrap_uses_equal_cell_weights(self):
        rows = []
        for cell, effect in zip(("a", "b", "c", "d"), (0.01, 0.02, 0.03, 0.04)):
            for _ in range(6):
                rows.append({"cell": cell, "effect": effect})
        result = _stratified_bootstrap(rows, "effect", 100, 5)
        self.assertAlmostEqual(result["estimate_m"], 0.025)
        self.assertAlmostEqual(result["bootstrap_lower_95_m"], 0.025)
        self.assertTrue(result["positive_in_all_cells"])

    def test_runner_enforces_schedule_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.assertEqual(
                next_expected_run(self.schedule, output, self.config), "HW001"
            )
            attempt = output / "HW001" / "attempt_001"
            attempt.mkdir(parents=True)
            write_json(
                attempt / "run_manifest.json",
                {"completed": True, "motion_started": True},
            )
            write_json(attempt / "validation.json", {"eligible_outcome": True})
            self.assertEqual(
                next_expected_run(self.schedule, output, self.config), "HW002"
            )

    def test_launch_failure_is_an_allowed_preserved_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary) / "attempt_001"
            attempt.mkdir()
            write_json(
                attempt / "launch_failure.json",
                {
                    "motion_started": False,
                    "retry_code": "logging_or_bag_start_failure_before_first_motion",
                },
            )
            self.assertEqual(
                retry_code_for_attempt(attempt),
                "logging_or_bag_start_failure_before_first_motion",
            )


if __name__ == "__main__":
    unittest.main()
