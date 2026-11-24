"""Regressions for predictive gradients and temporal readout causality."""
import unittest

import torch
import torch.nn.functional as F

from code.sigreg import SIGReg
from code.stjewm import STJEWM


def small_model(mode):
    return STJEWM(
        state_dim=3, action_dim=2, d_hid=8, embed_dim=8,
        action_emb_dim=8, cell_n_layers=1, n_d=3, readout_mode=mode,
    )


class PredictiveTrainingTests(unittest.TestCase):
    def test_sigreg_is_invariant_to_repeating_identical_time_slices(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            embeddings = torch.randn(3, 2, 4)
            regularizer = SIGReg(knots=3, num_proj=4)
            torch.manual_seed(19)
            expected = regularizer(embeddings)
            torch.manual_seed(19)
            actual = regularizer(embeddings.repeat(2, 1, 1))
            torch.testing.assert_close(actual, expected)

    def test_online_readouts_support_predictive_action_gradients(self):
        for mode in ("rate_only", "membrane_readout", "raw_spike"):
            with self.subTest(mode=mode), torch.random.fork_rng(devices=[]):
                torch.manual_seed(17)
                model = small_model(mode)
                states = torch.randn(2, 4, 3)
                actions = torch.randn(2, 4, 2, requires_grad=True)
                observed = model(states, actions)["emb"]
                predicted = model.predict(observed[:, :1], actions[:, :1])
                goal_predicted = model(states[:, :3], actions[:, :3])["emb"][:, -1]
                loss = F.mse_loss(predicted, torch.randn_like(predicted))
                loss = loss + F.mse_loss(goal_predicted, torch.randn_like(goal_predicted))
                loss.backward()
                self.assertIsNotNone(actions.grad)
                self.assertTrue(torch.isfinite(actions.grad).all())
                self.assertGreater(float(actions.grad.abs().sum()), 0.0)

    def test_rate_readout_cannot_observe_a_future_spike(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = small_model("rate_only")
            hidden = torch.zeros(2, 6, 8)
            spikes = torch.zeros_like(hidden)
            spikes[:, 1] = 1.0
            trace = torch.zeros_like(hidden)
            prefix = model._readout(hidden[:, :1], spikes[:, :1], trace[:, :1])
            full_prefix = model._readout(hidden, spikes, trace)[:, :1]
            torch.testing.assert_close(full_prefix, prefix)

    def test_jepa_targets_are_the_next_frames_for_every_history_length(self):
        """The CEM planner reads `predict` as one-step-ahead; the loss must supervise it that way."""
        from code.train.train import prediction_terms

        class _OracleNextFrameModel:
            def __init__(self, frames):
                self.frames = frames

            def predict(self, ctx_emb, ctx_act):  # true frame k+1 at every position k
                return self.frames[:, 1:1 + ctx_emb.shape[1]]

        frames = torch.randn(2, 8, 4)
        actions = torch.randn(2, 8, 2)
        for history_size, t_pred in ((1, 1), (3, 1), (3, 3), (4, 2)):
            with self.subTest(history_size=history_size, t_pred=t_pred):
                pred, target = prediction_terms(
                    _OracleNextFrameModel(frames), frames, actions, history_size, t_pred)
                self.assertEqual(tuple(pred.shape), (2, t_pred, 4))
                torch.testing.assert_close(pred, target)


if __name__ == "__main__":
    unittest.main()
