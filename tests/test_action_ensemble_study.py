import unittest
from collections import deque

import numpy as np

from scripts.run_action_ensemble_study import (
    TemporalLogisticRisk,
    apply_anchor_shift,
    average_precision,
    corrupt_observation,
    pairwise_distances,
    physical_command_vector,
    score_metrics,
    temporal_features,
    threshold_at_fpr,
)
from envs.f1tenth_sb3_env import STATE_N_SCALARS


class ActionEnsembleStudyTests(unittest.TestCase):
    def test_pairwise_disagreement_uses_common_physical_coordinates(self):
        robot = {
            "min_steering_angle": -0.4,
            "max_steering_angle": 0.4,
            "min_speed": 0.5,
            "max_speed": 5.5,
        }
        a = physical_command_vector({"steering_angle": 0.0, "speed": 3.0}, robot)
        b = physical_command_vector({"steering_angle": 0.4, "speed": 3.0}, robot)
        c = physical_command_vector({"steering_angle": 0.0, "speed": 5.5}, robot)
        distances = pairwise_distances([a, b, c])
        self.assertEqual(distances.shape, (3,))
        self.assertAlmostEqual(distances[0], 0.5)
        self.assertAlmostEqual(distances[1], 0.5)
        self.assertAlmostEqual(distances[2], np.sqrt(0.5))

    def test_lidar_corruption_never_changes_proprioception(self):
        obs = np.arange(STATE_N_SCALARS + 21, dtype=np.float32)
        shifted = corrupt_observation(
            obs,
            {"kind": "lidar_dropout", "probability": 1.0, "replacement": 1.0},
            np.random.default_rng(1),
        )
        np.testing.assert_array_equal(shifted[:STATE_N_SCALARS], obs[:STATE_N_SCALARS])
        np.testing.assert_array_equal(shifted[STATE_N_SCALARS:], 1.0)
        self.assertFalse(np.shares_memory(obs, shifted))

    def test_steering_delay_preserves_current_speed(self):
        state = {"steering_queue": deque([0.0, 0.0])}
        first = apply_anchor_shift(
            np.array([0.7, -0.2]), {"kind": "steering_delay", "policy_steps": 2}, state
        )
        second = apply_anchor_shift(
            np.array([-0.4, 0.8]), {"kind": "steering_delay", "policy_steps": 2}, state
        )
        third = apply_anchor_shift(
            np.array([0.1, 0.3]), {"kind": "steering_delay", "policy_steps": 2}, state
        )
        np.testing.assert_allclose(first, [0.0, -0.2])
        np.testing.assert_allclose(second, [0.0, 0.8])
        np.testing.assert_allclose(third, [0.7, 0.3])

    def test_temporal_features_include_current_and_change(self):
        history = deque(
            [np.array([1.0, 2.0]), np.array([3.0, 1.0])], maxlen=4
        )
        features = temporal_features(history)
        self.assertEqual(features.shape, (10,))
        np.testing.assert_allclose(features[:2], [3.0, 1.0])
        np.testing.assert_allclose(features[-2:], [2.0, -1.0])

    def test_average_precision_and_calibration_threshold(self):
        y = np.array([0, 1, 0, 1])
        score = np.array([0.1, 0.9, 0.2, 0.8])
        self.assertAlmostEqual(average_precision(y, score), 1.0)
        threshold = threshold_at_fpr(y, score, 0.5)
        self.assertEqual(threshold, 0.2)

    def test_temporal_risk_model_learns_separable_signal(self):
        x = np.array([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        groups = [f"episode-{i}" for i in range(len(y))]
        model = TemporalLogisticRisk(l2=0.01).fit(x, y, groups)
        prediction = model.predict(x)
        self.assertGreater(prediction[-1], prediction[0])
        self.assertGreater(average_precision(y, prediction), 0.99)

    def test_warning_metric_counts_misses_as_zero(self):
        rows = []
        for episode, detected in (("a", True), ("b", False)):
            for step in range(5):
                label = int(step >= 3)
                rows.append(
                    {
                        "episode_key": episode,
                        "step": step,
                        "label": label,
                        "steps_to_failure": 4 - step,
                        "dt": 0.1,
                        "heterogeneous": 1.0 if detected and step == 3 else 0.0,
                    }
                )
        metrics = score_metrics(rows, "heterogeneous", threshold=0.5)
        self.assertAlmostEqual(metrics["event_recall"], 0.5)
        self.assertAlmostEqual(metrics["median_warning_seconds_misses_zero"], 0.05)


if __name__ == "__main__":
    unittest.main()
