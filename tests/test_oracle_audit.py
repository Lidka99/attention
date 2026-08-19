import unittest

import torch

from attentionv3.training.oracle_audit import local_transfer_candidates


class OracleAuditTests(unittest.TestCase):
    def test_local_transfers_preserve_budget_and_respect_limits(self):
        keep = torch.tensor([[10, 10, 10, 10], [4, 16, 12, 8]])
        candidates = local_transfer_candidates(keep, min_groups=4, groups=16, transfer_groups=4)
        self.assertEqual(len(candidates), 12)
        for donor, recipient, valid, override in candidates:
            self.assertTrue(torch.equal(override.sum(dim=1), keep.sum(dim=1)))
            self.assertTrue((override >= 4).all())
            self.assertTrue((override <= 16).all())
            self.assertTrue(torch.equal(override[~valid], keep[~valid]))
            self.assertTrue(torch.equal(override[valid, donor], keep[valid, donor] - 4))
            self.assertTrue(torch.equal(override[valid, recipient], keep[valid, recipient] + 4))


if __name__ == "__main__":
    unittest.main()
