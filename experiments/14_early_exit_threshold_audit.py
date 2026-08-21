"""Calibrate confidence early-exit thresholds on one split and test on another."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from attentionv3.data import build_tinyimagenet_loaders
from attentionv3.models.early_exit_resnet import EarlyExitResNet50


def load_model(config, checkpoint, device):
    model = EarlyExitResNet50(config["num_classes"], small_images=True)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    return model.to(device).eval()


def collect(model, loader, device):
    outputs, targets_all = [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            outputs.append(torch.stack([F.softmax(logits[name], 1).max(1).values for name in ("stage2", "stage3")], 1).cpu())
            outputs.append(torch.stack([logits[name].argmax(1) for name in ("stage2", "stage3", "stage4")], 1).cpu())
            targets_all.append(targets)
    confidence = torch.cat(outputs[0::2]); predictions = torch.cat(outputs[1::2]); targets = torch.cat(targets_all)
    return confidence, predictions, targets


def score(data, threshold2, threshold3):
    confidence, predictions, targets = data
    exit2 = confidence[:, 0] >= threshold2
    exit3 = ~exit2 & (confidence[:, 1] >= threshold3)
    chosen = torch.where(exit2, predictions[:, 0], torch.where(exit3, predictions[:, 1], predictions[:, 2]))
    cost = torch.where(exit2, 0.4375, torch.where(exit3, 0.8125, 1.0))
    return {"accuracy": chosen.eq(targets).float().mean().item(), "cost_fraction": cost.float().mean().item(),
            "exit2_fraction": exit2.float().mean().item(), "exit3_fraction": exit3.float().mean().item()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--target-cost", type=float, default=0.85); parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text()); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, validation = build_tinyimagenet_loaders(config["data_dir"], args.batch_size, workers=4)
    dataset = validation.dataset; midpoint = len(dataset) // 2
    calibration = DataLoader(Subset(dataset, range(midpoint)), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test = DataLoader(Subset(dataset, range(midpoint, len(dataset))), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = load_model(config, args.checkpoint, device); calibration_data, test_data = collect(model, calibration, device), collect(model, test, device)
    grid = torch.linspace(0.05, 0.95, 19).tolist(); candidates = []
    for threshold2 in grid:
        for threshold3 in grid:
            candidates.append((score(calibration_data, threshold2, threshold3), threshold2, threshold3))
    feasible = [item for item in candidates if item[0]["cost_fraction"] <= args.target_cost]
    selected, threshold2, threshold3 = max(feasible or candidates, key=lambda item: item[0]["accuracy"])
    report = {"target_cost": args.target_cost, "selected_thresholds": {"stage2": threshold2, "stage3": threshold3},
              "calibration": selected, "test": score(test_data, threshold2, threshold3),
              "full_depth_test_accuracy": predictions_accuracy(test_data, 2)}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


def predictions_accuracy(data, index):
    return data[1][:, index].eq(data[2]).float().mean().item()


if __name__ == "__main__":
    main()
