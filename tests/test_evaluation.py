import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

from attentionv3.data import stratified_calibration_split
from attentionv3.evaluation import (TemperatureScaler, benchmark_latency, bootstrap_mean_ci,
                                    compute_metrics, paired_accuracy_delta)


def load_evaluation_script():
    path = Path(__file__).parents[1] / "experiments" / "02_calibrate_and_report.py"
    spec = importlib.util.spec_from_file_location("calibrate_and_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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

    def test_bootstrap_and_paired_delta(self):
        low, high = bootstrap_mean_ci(torch.tensor([0.0, 1.0, 1.0, 0.0]), repeats=100, seed=1)
        self.assertLessEqual(low, 0.5)
        self.assertGreaterEqual(high, 0.5)
        paired = paired_accuracy_delta(torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([0, 1]), repeats=100)
        self.assertEqual(paired["accuracy_delta"], 0.5)

    def test_evaluation_model_uses_attention_mode_from_config(self):
        script = load_evaluation_script()
        config = {
            "num_classes": 100,
            "attention": {
                "groups_per_stage": 16, "hidden": 64, "budget": 0.65,
                "max_budget": 0.9, "uncertainty_weight": 0.5,
                "adaptive_extra": 0.25, "mode": "static",
            },
        }
        with patch.object(script, "UCLAResNet50") as model:
            script.build_model(config)
        self.assertEqual(model.call_args.args[-1], "static")


if __name__ == "__main__":
    unittest.main()
