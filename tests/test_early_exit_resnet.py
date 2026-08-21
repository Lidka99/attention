import unittest

import torch

from attentionv3.models.early_exit_resnet import EarlyExitResNet50


class EarlyExitResNetTests(unittest.TestCase):
    def test_small_image_model_returns_all_exit_logits_and_gradients(self):
        model = EarlyExitResNet50(num_classes=11, small_images=True)
        outputs = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(set(outputs), {"stage2", "stage3", "stage4"})
        self.assertTrue(all(tuple(logits.shape) == (2, 11) for logits in outputs.values()))
        sum(logits.mean() for logits in outputs.values()).backward()
        self.assertIsNotNone(model.head2.weight.grad)
        self.assertIsNotNone(model.head3.weight.grad)
        self.assertIsNotNone(model.head4.weight.grad)


if __name__ == "__main__":
    unittest.main()
