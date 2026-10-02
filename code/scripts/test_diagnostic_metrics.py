"""Regression checks for reported diagnostic quantities, not model wiring."""
import unittest

import numpy as np
import torch

from code.scripts.latent_rollout import trajectory_metrics
from code.scripts.probe import r2_score


class DiagnosticMetricTests(unittest.TestCase):
    def test_reset_jumps_do_not_enter_response_or_event_alignment(self):
        obs = np.array([[0.], [1.], [3.], [1000.], [1003.], [1007.]])
        latent = np.array([[0.], [2.], [6.], [10.], [16.], [24.]])
        result = trajectory_metrics(obs, latent, np.array([0, 0, 0, 1, 1, 1]))
        self.assertEqual(result["n_transitions"], 4)
        self.assertAlmostEqual(result["responsiveness"], 2.0)
        self.assertAlmostEqual(result["event_rho"], 1.0)

    def test_constant_representation_has_undefined_event_correlation(self):
        obs = np.array([[0.], [1.], [3.], [6.]])
        result = trajectory_metrics(obs, np.ones((4, 2)), np.zeros(4))
        self.assertEqual(result["divergence"], 0.0)
        self.assertEqual(result["responsiveness"], 0.0)
        self.assertIsNone(result["event_rho"])

    def test_probe_r2_does_not_clip_bad_predictions_to_target_range(self):
        score, per_dim, constant = r2_score(torch.tensor([[0.], [3.]]),
                                           torch.tensor([[0.], [1.]]))
        self.assertEqual(score, -7.0)
        self.assertEqual(per_dim, [-7.0])
        self.assertEqual(constant, [False])


if __name__ == "__main__":
    unittest.main()
