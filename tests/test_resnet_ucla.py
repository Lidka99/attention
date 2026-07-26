import unittest

import torch

from attentionv3.models import UCLAResNet50


class UCLAResNetTests(unittest.TestCase):
    def test_resnet50_returns_logits_and_stage_diagnostics(self):
        model = UCLAResNet50(num_classes=7, groups=16, hidden=8)
        model.eval()
        with torch.no_grad():
            logits, diagnostics = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(logits.shape, (1, 7))
        self.assertEqual(len(diagnostics), 4)
        self.assertEqual([item.gate.shape[1] for item in diagnostics], [16] * 4)

    def test_resnet_attention_is_trainable(self):
        model = UCLAResNet50(num_classes=3, groups=16, hidden=8)
        model.train()
        logits, diagnostics = model(torch.randn(1, 3, 64, 64))
        loss = logits.square().mean() + sum(item.gate.mean() for item in diagnostics)
        loss.backward()
        self.assertIsNotNone(model.attention[0].utility_head[0].weight.grad)
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
