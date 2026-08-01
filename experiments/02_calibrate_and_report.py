"""Calibrate a UCLA checkpoint and report held-out ImageNet-100 metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from attentionv3.data import build_imagenet100_loaders, stratified_calibration_split
from attentionv3.evaluation import TemperatureScaler, benchmark_latency, collect_logits, compute_metrics
from attentionv3.models import UCLAResNet50


def build_model(config: dict) -> UCLAResNet50:
    """Build the evaluated architecture with the exact training attention mode."""
    attention = config["attention"]
    return UCLAResNet50(
        config["num_classes"], attention["groups_per_stage"], attention["hidden"],
        attention["budget"], attention["max_budget"], attention["uncertainty_weight"],
        attention["adaptive_extra"], attention.get("temperature", 0.5),
        attention.get("mode", "ucla"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/imagenet100_resnet50.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/imagenet100_evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, validation_loader, manifest = build_imagenet100_loaders(
        Path(config["data_dir"]) / "train", Path(config["data_dir"]) / "val",
        output_dir / f"imagenet100_seed{config['seed']}_manifest.json", config["seed"],
        config["batch_size"], config["workers"], config["image_size"], config["num_classes"],
    )
    calibration_data, test_data = stratified_calibration_split(
        validation_loader.dataset, args.calibration_fraction, config["seed"]
    )
    loader_options = {"batch_size": config["batch_size"], "num_workers": config["workers"],
                      "pin_memory": True}
    calibration_loader = DataLoader(calibration_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    calibration_logits, calibration_targets = collect_logits(model, calibration_loader, device)
    scaler = TemperatureScaler()
    temperature = scaler.fit(calibration_logits, calibration_targets)
    test_logits, test_targets = collect_logits(model, test_loader, device)
    calibrated_logits = scaler(test_logits)
    latency = benchmark_latency(model, torch.randn(1, 3, config["image_size"], config["image_size"], device=device))
    report = {"manifest": manifest, "checkpoint": str(args.checkpoint), "temperature": temperature,
              "calibration_samples": len(calibration_data), "test_samples": len(test_data),
              "uncalibrated": compute_metrics(test_logits, test_targets),
              "calibrated": compute_metrics(calibrated_logits, test_targets), "latency": latency}
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
