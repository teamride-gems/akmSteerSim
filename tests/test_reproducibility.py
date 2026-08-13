"""Regression tests for the Rung 2 reproducibility contract."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.preflight import tracked_artifact_violations
from scripts.run_repro_baseline import finite_numbers
from rl.common import run_eval_episode
from rl.eval import sha256_file
from utils.provenance import git_provenance, package_versions


ROOT = Path(__file__).resolve().parents[1]


class ReproducibilityTests(unittest.TestCase):
    def test_checkpoint_digest_is_stable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.zip"
            checkpoint.write_bytes(b"rung-2-checkpoint")
            self.assertEqual(
                sha256_file(checkpoint),
                "46c6512dae32bb31b3c197a165e599a62b9b1fd938b7f4fcc1a3be29db9f298d",
            )

    def test_terminal_collision_impulse_is_separate_from_actuator_acceleration(self):
        class Model:
            def predict(self, _obs, deterministic=True):
                return [0.0], None

        class Env:
            centerline = None

            def __init__(self):
                self.step_index = 0

            def reset(self, seed=None, options=None):
                return [0.0], {}

            def step(self, _action):
                self.step_index += 1
                crash = self.step_index == 2
                info = {
                    "term_reason": "crash" if crash else "running",
                    "crash": crash,
                    "a_long": 300.0 if crash else 2.0,
                    "a_lat": 120.0 if crash else 3.0,
                    "reward_breakdown": {"total": 0.0},
                }
                return [0.0], 0.0, crash, False, info

        result = run_eval_episode(Model(), Env(), seed=0, spawn_idx=1)
        self.assertEqual(result.max_abs_a_long, 300.0)
        self.assertEqual(result.max_abs_nonterminal_a_long, 2.0)
        self.assertEqual(result.max_abs_a_lat, 120.0)
        self.assertEqual(result.max_abs_nonterminal_a_lat, 3.0)

    def test_generated_artifact_detector(self):
        violations = tracked_artifact_violations([
            "envs/f1tenth_sb3_env.py",
            "envs/__pycache__/f1tenth_sb3_env.cpython-310.pyc",
            "checkpoints/model.zip",
            "runs/events.out.tfevents",
            "rollouts/policy.npz",
            "metrics/trajectory.csv",
            "experiments/20260402-233122/result.json",
        ])
        self.assertEqual(len(violations), 6)

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
