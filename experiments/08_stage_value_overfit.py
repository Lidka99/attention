"""Verify that the stage-value head can fit frozen counterfactual targets."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from attentionv3.data import build_tinyimagenet_loaders
from attentionv3.models import GlobalValueBudgetResNet50
from attentionv3.training import counterfactual_stage_value_targets, stage_value_loss


def build_model(config, checkpoint, device):
    attention = config["attention"]
    model = GlobalValueBudgetResNet50(config["num_classes"], attention["groups_per_stage"], attention["hidden"],
                                      attention["budget"], attention.get("min_groups_per_stage", 1),
                                      attention.get("allocation", "value"), attention.get("quota_temperature", 1.0))
    model.backbone.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.backbone.maxpool = torch.nn.Identity()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    return model.to(device)


def stem_features(model, images):
    x = model.backbone.conv1(images)
    x = model.backbone.bn1(x)
    x = model.backbone.relu(x)
    x = model.backbone.maxpool(x)
    return x.mean(dim=(2, 3))


def frozen_targets(model, images, targets, groups_to_transfer):
    model.eval()
    with torch.no_grad():
        logits, diagnostics = model(images)
        base_loss = F.cross_entropy(logits, targets, reduction="none")
        keep = torch.stack([item.keep_count for item in diagnostics], dim=1)
        losses = []
        for donor in range(4):
            override = keep.clone()
            for row in range(override.shape[0]):
                if override[row, donor] <= model.min_groups_per_stage:
                    continue
                recipients = [stage for stage in range(4)
                              if stage != donor and override[row, stage] < model.groups]
                if not recipients:
                    continue
                recipient = min(recipients, key=lambda stage: int(override[row, stage]))
                amount = min(groups_to_transfer, int(override[row, donor] - model.min_groups_per_stage),
                             int(model.groups - override[row, recipient]))
                override[row, donor] -= amount
                override[row, recipient] += amount
            candidate, _ = model(images, override)
            losses.append(F.cross_entropy(candidate, targets, reduction="none"))
    return counterfactual_stage_value_targets(base_loss, torch.stack(losses, dim=1))


def metrics(policy, features, targets):
    values = policy(features)[:, -4:]
    return {"loss": stage_value_loss(values, targets).item(),
            "agreement": values.argmax(1).eq(targets.argmax(1)).float().mean().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, validation = build_tinyimagenet_loaders(config["data_dir"], config["batch_size"], config["workers"])
    images, targets = next(iter(validation))
    images, targets = images.to(device), targets.to(device)
    model = build_model(config, args.checkpoint, device)
    targets_value = frozen_targets(model, images, targets, config["training"]["ablation_groups"])
    with torch.no_grad():
        features = stem_features(model, images).detach()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.policy.parameters(), lr=args.learning_rate, weight_decay=0.0)
    initial = metrics(model.policy, features, targets_value)
    model.policy.train()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = stage_value_loss(model.policy(features)[:, -4:], targets_value)
        loss.backward()
        optimizer.step()
    report = {"samples": targets.numel(), "steps": args.steps, "target_entropy":
              (-(targets_value * targets_value.clamp_min(1e-8).log()).sum(1)).mean().item(),
              "initial": initial, "final": metrics(model.policy, features, targets_value)}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
