"""Regression tests for the Rung 2 reproducibility contract."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from scripts.preflight import tracked_artifact_violations
from scripts.run_repro_baseline import finite_numbers
from utils.provenance import git_provenance, package_versions


ROOT = Path(__file__).resolve().parents[1]


class ReproducibilityTests(unittest.TestCase):
    def test_generated_artifact_detector(self):
        violations = tracked_artifact_violations([
            "envs/f1tenth_sb3_env.py",
            "envs/__pycache__/f1tenth_sb3_env.cpython-310.pyc",
            "checkpoints/model.zip",
            "runs/events.out.tfevents",
        ])
        self.assertEqual(len(violations), 3)

    def test_git_provenance_contains_source_and_submodule_identity(self):
        provenance = git_provenance(ROOT)
        self.assertEqual(len(provenance["commit"]), 40)
        self.assertIn("dirty", provenance)
        self.assertTrue(provenance["submodules"])

    def test_package_versions_have_stable_keys(self):
        versions = package_versions(("numpy", "package-that-does-not-exist"))
        self.assertIsNotNone(versions["numpy"])
        self.assertIsNone(versions["package-that-does-not-exist"])

    def test_nonfinite_metric_detection_is_recursive(self):
        self.assertEqual(finite_numbers({"a": [1.0, 2.0]}), [])
        failures = finite_numbers({"summary": {"reward": float("nan")}})
        self.assertEqual(failures, ["root.summary.reward"])

    def test_legacy_runner_fails_closed(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_experiment.py")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("retired", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
