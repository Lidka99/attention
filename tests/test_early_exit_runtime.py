import unittest

import torch

from attentionv3.models.early_exit_resnet import EarlyExitResNet50
from attentionv3.models.early_exit_runtime import forward_to_exit


class EarlyExitRuntimeTests(unittest.TestCase):
    def test_prefix_forward_matches_full_forward(self):
        model = EarlyExitResNet50(num_classes=11, small_images=True).eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            full = model(images)
            for stage in (2, 3, 4):
                self.assertTrue(torch.allclose(forward_to_exit(model, images, stage), full[f"stage{stage}"]))
        with self.assertRaises(ValueError):
            forward_to_exit(model, images, 1)
