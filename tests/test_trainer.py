import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from attentionv3 import BudgetController
from attentionv3.training import UCLALoss, evaluate, train_one_epoch


class TinyUCLAModel(nn.Module):
    """Small model keeps the training-loop test fast and deterministic."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Linear(4, 8)
        self.utility = nn.Linear(8, 4)
        self.uncertainty = nn.Linear(8, 4)
        self.controller = BudgetController(groups=4, budget=0.5, max_budget=0.75)
        self.classifier = nn.Linear(8, 3)

    def forward(self, x):
        features = torch.relu(self.features(x))
        decision = self.controller(self.utility(features), self.uncertainty(features))
        channel_gate = decision.gate.repeat_interleave(2, dim=1)
        return self.classifier(features * channel_gate), [decision]


class TrainerTests(unittest.TestCase):
    def test_train_and_evaluate_report_expected_metrics(self):
        torch.manual_seed(7)
        loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randint(0, 3, (8,))), batch_size=4)
        student, teacher = TinyUCLAModel(), nn.Linear(4, 3)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        criterion = UCLALoss(groups=4)

        train = train_one_epoch(student, teacher, loader, optimizer, criterion)
        validation = evaluate(student, loader, criterion)

        self.assertEqual(train.samples, 8)
        self.assertEqual(validation.samples, 8)
        self.assertTrue(0.0 <= train.accuracy <= 1.0)
        self.assertTrue(0.0 < validation.mean_keep_ratio <= 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(train.loss)))
        self.assertFalse(any(parameter.requires_grad for parameter in teacher.parameters()))

    def test_ungated_model_reports_full_keep_ratio(self):
        class UngatedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = nn.Linear(4, 3)

            def forward(self, x):
                return self.classifier(x), []

        loader = DataLoader(TensorDataset(torch.randn(6, 4), torch.randint(0, 3, (6,))), batch_size=2)
        metrics = evaluate(UngatedModel(), loader, UCLALoss(groups=1))
        self.assertEqual(metrics.mean_keep_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
