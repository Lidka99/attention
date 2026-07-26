import unittest

import torch

from attentionv3 import BudgetController, UCLAChannelAttention


class BudgetControllerTests(unittest.TestCase):
    def test_keeps_base_budget_for_confident_samples(self):
        controller = BudgetController(groups=10, budget=0.5, max_budget=0.8)
        utility = torch.arange(10, dtype=torch.float32).repeat(2, 1)
        uncertainty = torch.full_like(utility, -20.0)
        output = controller(utility, uncertainty)
        self.assertEqual(output.keep_count.tolist(), [5, 5])
        self.assertEqual(output.gate.sum(dim=1).tolist(), [5.0, 5.0])

    def test_uncertainty_can_spend_bounded_extra_budget(self):
        controller = BudgetController(groups=10, budget=0.5, max_budget=0.8,
                                      adaptive_extra=0.5)
        output = controller(torch.zeros(1, 10), torch.full((1, 10), 20.0))
        self.assertEqual(output.keep_count.item(), 8)
        self.assertEqual(output.gate.sum().item(), 8.0)

    def test_attention_expands_group_gate_to_channels(self):
        module = UCLAChannelAttention(channels=16, groups=4, hidden=8, budget=0.5)
        x = torch.randn(3, 16, 8, 8)
        y, decision = module(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(decision.gate.shape, (3, 4))
        self.assertTrue(torch.isfinite(y).all())

    def test_static_mode_has_the_same_gate_for_every_image(self):
        module = UCLAChannelAttention(channels=16, groups=4, hidden=8, budget=0.5, mode="static")
        module.eval()
        _, decision = module(torch.randn(3, 16, 8, 8))
        self.assertTrue(torch.equal(decision.gate[0], decision.gate[1]))
        self.assertEqual(module.controller.uncertainty_weight, 0.0)
        self.assertEqual(module.controller.adaptive_extra, 0.0)

    def test_invalid_grouping_is_rejected(self):
        with self.assertRaises(ValueError):
            UCLAChannelAttention(channels=15, groups=4)

    def test_training_gate_has_straight_through_gradients(self):
        controller = BudgetController(groups=6, budget=0.5, temperature=0.5)
        utility = torch.randn(2, 6, requires_grad=True)
        uncertainty = torch.randn(2, 6, requires_grad=True)
        output = controller(utility, uncertainty)
        self.assertTrue(torch.allclose(output.gate, output.gate.round(), atol=1e-6))
        output.gate.sum().backward()
        self.assertIsNotNone(utility.grad)
        self.assertIsNotNone(uncertainty.grad)
        self.assertGreater(utility.grad.abs().sum().item(), 0.0)
        self.assertGreater(uncertainty.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
