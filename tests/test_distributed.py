import unittest

from attentionv3.training.distributed import initialize_distributed


class DistributedTests(unittest.TestCase):
    def test_single_process_context_is_available_for_local_debugging(self):
        context = initialize_distributed(requested_device="cpu")
        self.assertFalse(context.enabled)
        self.assertTrue(context.is_main)
        self.assertEqual(context.world_size, 1)


if __name__ == "__main__":
    unittest.main()
