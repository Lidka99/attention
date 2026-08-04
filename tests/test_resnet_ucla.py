import unittest

import torch

from attentionv3.models import GlobalUCLAResNet50, GlobalValueBudgetResNet50, UCLAResNet50


class UCLAResNetTests(unittest.TestCase):

    def test_value_budget_resnet_keeps_exact_budget_and_accepts_counterfactual_override(self):
        model = GlobalValueBudgetResNet50(num_classes=10, groups=4, hidden=8, budget=0.625)
        model.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.backbone.maxpool = torch.nn.Identity()
        images = torch.randn(2, 3, 32, 32)
        logits, diagnostics = model(images)
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(torch.all(torch.stack([item.keep_count for item in diagnostics]).sum(0) == 10))
        override = torch.tensor([[4, 3, 2, 1], [1, 2, 3, 4]])
        _, overridden = model(images, override)
        self.assertTrue(torch.equal(torch.stack([item.keep_count for item in overridden]).T, override))
    def test_resnet50_returns_logits_and_stage_diagnostics(self):
        model = UCLAResNet50(num_classes=7, groups=16, hidden=8)
        model.eval()
        with torch.no_grad():
            logits, diagnostics = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(logits.shape, (1, 7))
        self.assertEqual(len(diagnostics), 4)
        self.assertEqual([item.gate.shape[1] for item in diagnostics], [16] * 4)

    def test_resnet_propagates_controller_hyperparameters(self):
        model = UCLAResNet50(groups=16, uncertainty_weight=0.7, adaptive_extra=0.1, temperature=0.3)
        controller = model.attention[0].controller
        self.assertEqual(controller.uncertainty_weight, 0.7)
        self.assertEqual(controller.adaptive_extra, 0.1)
        self.assertEqual(controller.temperature, 0.3)

    def test_resnet_accepts_utility_only_mode(self):
        model = UCLAResNet50(num_classes=3, groups=16, hidden=8, mode="utility_only")
        self.assertEqual(model.attention[0].mode, "utility_only")
        self.assertEqual(model.attention[0].controller.adaptive_extra, 0.0)

    def test_resnet_attention_is_trainable(self):
        model = UCLAResNet50(num_classes=3, groups=16, hidden=8)
        model.train()
        logits, diagnostics = model(torch.randn(1, 3, 64, 64))
        loss = logits.square().mean() + sum(item.gate.mean() for item in diagnostics)
        loss.backward()
        self.assertIsNotNone(model.attention[0].utility_head[0].weight.grad)
        self.assertTrue(torch.isfinite(logits).all())

    def test_global_resnet_has_exact_shared_budget(self):
        model = GlobalUCLAResNet50(num_classes=7, groups=16, hidden=8, budget=0.625).eval()
        with torch.no_grad():
            _, diagnostics = model(torch.randn(2, 3, 32, 32))
        totals = torch.stack([item.keep_count for item in diagnostics]).sum(dim=0)
        self.assertTrue(torch.equal(totals, torch.full((2,), 40)))

    def test_global_resnet_utility_policy_is_trainable(self):
        model = GlobalUCLAResNet50(num_classes=3, groups=16, hidden=8, budget=0.625)
        logits, _ = model(torch.randn(2, 3, 32, 32))
        logits.square().mean().backward()
        self.assertIsNotNone(model.policy[-1].weight.grad)


if __name__ == "__main__":
    unittest.main()
