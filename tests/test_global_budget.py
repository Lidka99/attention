import unittest

import torch

from attentionv3 import GlobalBudgetAllocator


class GlobalBudgetTests(unittest.TestCase):
    def test_total_budget_is_exact_for_each_image(self):
        allocator = GlobalBudgetAllocator(stages=4, groups=4, budget=0.5)
        output = allocator(torch.randn(3, 4, 4), torch.randn(3, 4, 4))
        self.assertTrue(torch.equal(output.keep_count.sum(dim=1), torch.full((3,), 8)))
        self.assertTrue(torch.equal(output.gate.sum(dim=(1, 2)).long(), torch.full((3,), 8)))

    def test_uncertain_stage_receives_extra_groups(self):
        allocator = GlobalBudgetAllocator(stages=2, groups=4, budget=0.75)
        utility = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
        uncertainty = torch.tensor([[[-5.0] * 4, [5.0] * 4]])
        output = allocator(utility, uncertainty)
        self.assertGreater(output.keep_count[0, 1], output.keep_count[0, 0])


if __name__ == "__main__":
    unittest.main()
