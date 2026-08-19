import unittest

import torch

from attentionv3.training.oracle_actions import (legal_action_mask, masked_action_loss,
                                                  oracle_action_labels, transfer_actions)


class OracleActionTests(unittest.TestCase):
    def test_transfer_space_has_noop_and_twelve_directed_actions(self):
        self.assertEqual(len(transfer_actions()), 12)
        self.assertEqual(len(set(transfer_actions())), 12)

    def test_legal_mask_respects_floor_and_capacity(self):
        keep = torch.tensor([[4, 16, 10, 10]])
        legal = legal_action_mask(keep, min_groups=4, groups=16, transfer_groups=4)
        self.assertEqual(tuple(legal.shape), (1, 13))
        self.assertTrue(legal[0, 0])
        actions = transfer_actions()
        self.assertFalse(legal[0, 1 + actions.index((0, 2))])
        self.assertFalse(legal[0, 1 + actions.index((2, 1))])
        self.assertTrue(legal[0, 1 + actions.index((1, 0))])

    def test_oracle_ignores_illegal_lower_loss_and_masked_loss(self):
        legal = torch.tensor([[True, False, True], [True, True, False]])
        losses = torch.tensor([[3.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        labels = oracle_action_labels(losses, legal)
        self.assertTrue(torch.equal(labels, torch.tensor([2, 1])))
        good = masked_action_loss(torch.tensor([[0.0, -5.0, 3.0], [0.0, 3.0, -5.0]]), labels, legal)
        bad = masked_action_loss(torch.tensor([[3.0, -5.0, 0.0], [3.0, 0.0, -5.0]]), labels, legal)
        self.assertLess(good, bad)


if __name__ == "__main__":
    unittest.main()
