import unittest
import torch
from attentionv3.models.static_depth_resnet import StaticDepthResNet50

class StaticDepthResNetTests(unittest.TestCase):
    def test_stage3_small_image_model_has_logits_and_gradients(self):
        model = StaticDepthResNet50(9, exit_stage=3, small_images=True)
        logits = model(torch.randn(2, 3, 32, 32)); self.assertEqual(tuple(logits.shape), (2, 9))
        logits.mean().backward(); self.assertIsNotNone(model.head.weight.grad)

if __name__ == "__main__": unittest.main()
