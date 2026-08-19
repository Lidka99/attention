"""Audit the label-informed upper bound of local stage-budget transfers."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from attentionv3.data import build_tinyimagenet_loaders
from attentionv3.models import GlobalValueBudgetResNet50
from attentionv3.training.oracle_audit import local_transfer_candidates


def build_model(config, checkpoint, device):
    attention = config["attention"]
    model = GlobalValueBudgetResNet50(config["num_classes"], attention["groups_per_stage"], attention["hidden"],
                                      attention["budget"], attention.get("min_groups_per_stage", 1),
                                      attention.get("allocation", "value"), attention.get("quota_temperature", 1.0))
    model.backbone.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.backbone.maxpool = torch.nn.Identity()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


def audit(model, loader, device, max_samples, transfer_groups):
    total = baseline_correct = oracle_correct = changed = legal = 0
    baseline_loss_total = oracle_loss_total = 0.0
    with torch.no_grad():
        for images, targets in loader:
            if total >= max_samples:
                break
            remaining = max_samples - total
            images, targets = images[:remaining].to(device), targets[:remaining].to(device)
            logits, diagnostics = model(images)
            baseline_loss = F.cross_entropy(logits, targets, reduction="none")
            best_loss, best_logits = baseline_loss.clone(), logits.clone()
            keep = torch.stack([item.keep_count for item in diagnostics], dim=1)
            for _, _, valid, override in local_transfer_candidates(
                    keep, model.min_groups_per_stage, model.groups, transfer_groups):
                if not valid.any():
                    continue
                candidate_logits, _ = model(images, override)
                candidate_loss = F.cross_entropy(candidate_logits, targets, reduction="none")
                improve = valid & candidate_loss.lt(best_loss)
                best_loss[improve] = candidate_loss[improve]
                best_logits[improve] = candidate_logits[improve]
                legal += int(valid.sum())
            baseline_correct += int(logits.argmax(1).eq(targets).sum())
            oracle_correct += int(best_logits.argmax(1).eq(targets).sum())
            changed += int(best_loss.lt(baseline_loss).sum())
            baseline_loss_total += baseline_loss.sum().item()
            oracle_loss_total += best_loss.sum().item()
            total += targets.numel()
    return {"samples": total, "transfer_groups": transfer_groups,
            "baseline_accuracy": baseline_correct / total, "oracle_accuracy": oracle_correct / total,
            "accuracy_gain_pp": 100 * (oracle_correct - baseline_correct) / total,
            "baseline_nll": baseline_loss_total / total, "oracle_nll": oracle_loss_total / total,
            "nll_reduction": (baseline_loss_total - oracle_loss_total) / total,
            "oracle_changed_fraction": changed / total,
            "mean_legal_transfers": legal / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--transfer-groups", type=int, default=4)
    args = parser.parse_args()
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, validation = build_tinyimagenet_loaders(config["data_dir"], config["batch_size"], config["workers"])
    report = audit(build_model(config, args.checkpoint, device), validation, device, args.max_samples, args.transfer_groups)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
