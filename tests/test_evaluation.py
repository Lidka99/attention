import unittest

import torch
from torch import nn
from torch.utils.data import Dataset

from attentionv3.data import stratified_calibration_split
from attentionv3.evaluation import TemperatureScaler, benchmark_latency, compute_metrics


class TinyDataset(Dataset):
    samples = [("a", 0), ("b", 0), ("c", 0), ("d", 1), ("e", 1), ("f", 1)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return torch.tensor([index]), self.samples[index][1]


class EvaluationTests(unittest.TestCase):
    def test_stratified_split_is_disjoint_and_keeps_classes(self):
        calibration, test = stratified_calibration_split(TinyDataset(), fraction=1 / 3, seed=5)
        self.assertEqual(set(calibration.indices) & set(test.indices), set())
        self.assertEqual({TinyDataset.samples[index][1] for index in calibration.indices}, {0, 1})
        self.assertEqual(len(calibration) + len(test), 6)

    def test_temperature_scaling_and_metrics(self):
        logits = torch.tensor([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]])
        targets = torch.tensor([1, 1, 1, 1])
        scaler = TemperatureScaler()
        scaler.fit(logits, targets)
        self.assertGreater(scaler.temperature.item(), 0.0)
        self.assertLessEqual(torch.nn.functional.cross_entropy(scaler(logits), targets),
                             torch.nn.functional.cross_entropy(logits, targets))
        metrics = compute_metrics(logits, targets)
        self.assertEqual(metrics["samples"], 4)
        self.assertTrue(0.0 <= metrics["top1"] <= 1.0)

    def test_latency_benchmark_returns_percentiles(self):
        result = benchmark_latency(nn.Identity(), torch.randn(1, 3), warmup=0, runs=3)
        self.assertEqual(result["runs"], 3)
        self.assertGreaterEqual(result["p95_ms"], result["median_ms"])


if __name__ == "__main__":
    unittest.main()
