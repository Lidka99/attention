import unittest

import torch

from attentionv3.training import counterfactual_stage_value_targets, stage_value_loss


class StageValueTargetTests(unittest.TestCase):
    def test_larger_ablation_regret_receives_larger_target_mass(self):
        targets = counterfactual_stage_value_targets(torch.tensor([1.0]), torch.tensor([[1.1, 2.0, 1.2]]))
        self.assertEqual(tuple(targets.shape), (1, 3))
        self.assertAlmostEqual(targets.sum().item(), 1.0, places=6)
        self.assertGreater(targets[0, 1].item(), targets[0, 0].item())

    def test_value_loss_prefers_matching_logits(self):
        targets = torch.tensor([[0.1, 0.8, 0.1]])
        good = stage_value_loss(torch.tensor([[0.0, 3.0, 0.0]]), targets)
        bad = stage_value_loss(torch.tensor([[3.0, 0.0, 0.0]]), targets)
        self.assertLess(good, bad)


if __name__ == "__main__":
    unittest.main()
