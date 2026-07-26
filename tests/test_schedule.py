import unittest

from torch import nn

from attentionv3 import BudgetController
from attentionv3.training import BudgetCurriculum, UCLALoss, apply_curriculum


class ScheduleTests(unittest.TestCase):
    def test_three_phases_have_expected_budgets(self):
        schedule = BudgetCurriculum(2, 4, 2, 0.5, 0.75, 0.1, 0.01)
        self.assertEqual(schedule.state_for_epoch(0).phase, "warmup")
        self.assertEqual(schedule.state_for_epoch(0).budget, 1.0)
        self.assertEqual(schedule.state_for_epoch(2).phase, "sparsification")
        self.assertAlmostEqual(schedule.state_for_epoch(5).budget, 0.5)
        self.assertEqual(schedule.state_for_epoch(6).phase, "fine_tune")
        self.assertEqual(schedule.total_epochs, 8)

    def test_state_updates_every_controller_and_loss(self):
        model = nn.Sequential(BudgetController(groups=4), nn.Sequential(BudgetController(groups=4)))
        loss = UCLALoss(groups=4)
        state = BudgetCurriculum(0, 1, 0, 0.6, 0.8, 0.2, 0.03).state_for_epoch(0)
        apply_curriculum(model, loss, state)
        for module in model.modules():
            if isinstance(module, BudgetController):
                self.assertEqual(module.budget, 0.6)
                self.assertEqual(module.max_budget, 0.8)
        self.assertEqual(loss.target_budget, 0.6)
        self.assertEqual(loss.brier_weight, 0.2)


if __name__ == "__main__":
    unittest.main()
