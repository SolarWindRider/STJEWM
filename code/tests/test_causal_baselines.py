"""Regressions for future-token leakage through public baseline readouts."""
import unittest

import torch

from code.alif_timecell_baseline import ALIFTimecellBaseline
from code.lewm_transformer_baseline import LeWMTransformerBaseline
from code.stacked_lif_baseline import StackedLIFTraceOnly


class CausalBaselineTests(unittest.TestCase):
    def assert_prefix_causal(self, run, inputs, actions):
        full = run(inputs, actions)
        for length in (1, 4):
            changed_inputs = inputs.clone()
            changed_inputs[:, length:] = 0.5 - changed_inputs[:, length:]
            changed_actions = actions.clone()
            changed_actions[:, length:] = 0.5 - changed_actions[:, length:]
            variants = {
                "prefix_only": run(inputs[:, :length], actions[:, :length]),
                "future_inputs": run(changed_inputs, actions)[:, :length],
                "future_actions": run(inputs, changed_actions)[:, :length],
            }
            for variant, actual in variants.items():
                with self.subTest(length=length, variant=variant):
                    torch.testing.assert_close(
                        actual, full[:, :length], atol=1e-6, rtol=1e-5,
                    )

    def test_learned_lewm_forward_and_predict_are_prefix_causal(self):
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(7)
            model = LeWMTransformerBaseline(
                state_dim=3, action_dim=2, embed_dim=8, num_layers=2, num_heads=2,
            ).eval()
            # Pristine AdaLN-zero gates hide noncausal attention. Exercise the
            # learned, input-dependent attention branch rather than an identity.
            for block in model.blocks:
                block.adaLN[-1].weight.normal_(std=0.15)
                block.adaLN[-1].bias.normal_(std=0.15)
                block.adaLN[-1].bias[16:24].add_(0.75)
            obs = torch.randn(2, 7, 3)
            actions = torch.randn(2, 7, 2)
            latents = torch.randn(2, 7, 8)
            self.assert_prefix_causal(lambda x, a: model(x, a)["emb"], obs, actions)
            self.assert_prefix_causal(model.predict, latents, actions)

    def test_alif_forward_and_predict_are_prefix_causal(self):
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(11)
            model = ALIFTimecellBaseline(
                state_dim=3, action_dim=2, d_hid=8, n_layers=2,
            ).eval()
            obs = torch.randn(2, 7, 3)
            actions = torch.randn(2, 7, 2)
            latents = torch.randn(2, 7, 8)
            self.assert_prefix_causal(lambda x, a: model(x, a)["emb"], obs, actions)
            self.assert_prefix_causal(model.predict, latents, actions)

    def test_alif_rollout_keeps_the_requested_context_at_every_step(self):
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(17)
            model = ALIFTimecellBaseline(
                state_dim=3, action_dim=2, d_hid=8, n_layers=2,
            ).eval()
            initial = torch.randn(2, 3, 8)
            actions = torch.randn(2, 5, 2)
            history = initial.clone()
            expected = []
            for step in range(actions.shape[1]):
                window = actions[:, step:step + 3]
                window = torch.nn.functional.pad(window, (0, 0, 0, 3 - window.shape[1]))
                next_latent = model.predict(history[:, -3:], window)[:, -1:]
                expected.append(next_latent)
                history = torch.cat((history, next_latent), dim=1)
            torch.testing.assert_close(
                model.rollout(initial, actions, history_size=3),
                torch.cat(expected, dim=1),
            )

    def test_stacked_trace_has_causal_fixed_divisor_impulse_response(self):
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(13)
            model = StackedLIFTraceOnly(
                state_dim=1, action_dim=1, d_in=4, embed_dim=4, n_layers=1,
            ).eval()
            # Produce one real native-cell spike at t=1, not an all-zero
            # initialization that would also pass with symmetric pooling.
            for projector in (model.state_projector.proj, model.action_encoder.proj):
                projector[0].weight.zero_()
                projector[0].weight[0, 0] = 1.0
                projector[0].weight[1, 0] = -1.0
                projector[0].bias.zero_()
                projector[2].weight.copy_(torch.eye(4))
                projector[2].bias.zero_()
            model.stack.cells[0].w_in.weight.copy_(10.0 * torch.eye(4))
            model.stack.cells[0].w_in.bias.zero_()
            model.readout.weight.copy_(torch.eye(4))
            model.readout.bias.zero_()
            obs = torch.zeros(1, 7, 1)
            obs[:, 1, 0] = 2.0
            actions = torch.zeros_like(obs)
            expected = torch.zeros(1, 7, 4)
            expected[:, 1:5, 0] = 0.25
            result = model(obs, actions)
            torch.testing.assert_close(result["emb"], expected)
            torch.testing.assert_close(result["trace"], expected)
            self.assert_prefix_causal(lambda x, a: model(x, a)["emb"], obs, actions)
            latents = torch.zeros(1, 7, 4)
            latents[:, 1, 0] = 2.0
            self.assert_prefix_causal(model.predict, latents, actions)


if __name__ == "__main__":
    unittest.main()
