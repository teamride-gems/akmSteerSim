"""Unit tests for the preregistered Gate 0/1 decision logic."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import yaml
from stable_baselines3.common.logger import configure

from rl.common import EpisodeResult, log_episode_metrics
from scripts.audit_decoder_state import activation_precheck, gate_decision


class GateLadderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load(
            Path("configs/decoder_state_gate.yaml").read_text(encoding="utf-8")
        )

    @staticmethod
    def passing_metrics():
        return {
            "register_from_observation": {"r2": [0.50, 0.99]},
            "steer_nmse_reduction": 0.30,
            "steer_limiter_activation_fraction": 0.40,
            "matched_histories": {
                "n_pairs": 500,
                "median_command_effect_rad": 0.02,
            },
        }

    def test_gate1_requires_every_preregistered_check(self):
        passed, checks = gate_decision(self.passing_metrics(), self.cfg)
        self.assertTrue(passed)
        self.assertTrue(all(checks.values()))

    def test_gate1_kills_when_register_is_recoverable(self):
        metrics = self.passing_metrics()
        metrics["register_from_observation"]["r2"][0] = 0.95

        passed, checks = gate_decision(metrics, self.cfg)

        self.assertFalse(passed)
        self.assertFalse(checks["register_not_recoverable"])

    def test_gate1_kills_when_limiter_is_rare(self):
        metrics = self.passing_metrics()
        metrics["steer_limiter_activation_fraction"] = 0.05

        passed, checks = gate_decision(metrics, self.cfg)

        self.assertFalse(passed)
        self.assertFalse(checks["limiter_activates"])

    def test_long_track_name_metrics_do_not_collide_in_console_logger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = configure(temp_dir, ["stdout"])
            log_episode_metrics(
                logger,
                "eval_validation/BrandsHatch",
                [EpisodeResult()],
            )
            logger.dump(10)

    def test_gate1_activation_precheck_weights_physical_transitions(self):
        gate0_result = {
            "runs": [
                {
                    "episodes": [
                        {"term_reason": "lap_complete", "length": 100, "steer_clip_frac": 0.10},
                        {"term_reason": "lap_complete", "length": 300, "steer_clip_frac": 0.30},
                        {"term_reason": "crash", "length": 50, "steer_clip_frac": 1.0},
                    ]
                }
            ]
        }

        result = activation_precheck(gate0_result, self.cfg)

        self.assertAlmostEqual(result["steer_limiter_activation_fraction"], 0.25)
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
