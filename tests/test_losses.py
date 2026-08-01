import unittest

import torch

from attentionv3 import BudgetController
from attentionv3.training import UCLALoss
from attentionv3.training.losses import brier_error_loss


class LossTests(unittest.TestCase):
    def _diagnostics(self, batch=4, groups=8):
        controller = BudgetController(groups=groups, budget=0.5, max_budget=0.75)
        utility = torch.randn(batch, groups, requires_grad=True)
        uncertainty = torch.randn(batch, groups, requires_grad=True)
        return [controller(utility, uncertainty), controller(utility, uncertainty)]

    def test_loss_breakdown_contains_all_terms(self):
        logits = torch.randn(4, 5, requires_grad=True)
        teacher = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        breakdown = UCLALoss(groups=8)(logits, targets, self._diagnostics(), teacher)
        self.assertTrue(torch.isfinite(breakdown.total))
        self.assertGreater(breakdown.classification.item(), 0.0)
        breakdown.total.backward()
        self.assertIsNotNone(logits.grad)

    def test_no_teacher_still_produces_valid_loss(self):
        logits = torch.randn(4, 5, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3])
        breakdown = UCLALoss(groups=8)(logits, targets, self._diagnostics())
        self.assertEqual(breakdown.distillation.item(), 0.0)
        self.assertTrue(torch.isfinite(breakdown.total))

    def test_soft_error_uncertainty_target_is_valid(self):
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        loss = brier_error_loss(logits, targets, self._diagnostics(), target="soft_error")
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
