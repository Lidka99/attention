"""Predict oracle transfers among future stages after observing ResNet layer1."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from attentionv3.data import build_tinyimagenet_loaders
from attentionv3.models import GlobalValueBudgetResNet50
from attentionv3.training.oracle_actions import masked_action_loss, oracle_action_labels


FUTURE_ACTIONS = ((1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2))


def build_model(config, checkpoint, device):
    attention = config["attention"]
    model = GlobalValueBudgetResNet50(config["num_classes"], attention["groups_per_stage"], attention["hidden"],
                                      attention["budget"], attention.get("min_groups_per_stage", 1),
                                      attention.get("allocation", "utility"), attention.get("quota_temperature", 1.0))
    model.backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.backbone.maxpool = nn.Identity()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    return model.to(device).eval()


def layer1_features(model, images):
    x = model.backbone.conv1(images)
    x = model.backbone.bn1(x)
    x = model.backbone.relu(x)
    return model.backbone.layer1(model.backbone.maxpool(x)).mean(dim=(2, 3))


def future_legal_mask(keep, min_groups, groups, transfer_groups):
    legal = [torch.ones(keep.shape[0], dtype=torch.bool, device=keep.device)]
    for donor, recipient in FUTURE_ACTIONS:
        legal.append((keep[:, donor] - transfer_groups >= min_groups)
                     & (keep[:, recipient] + transfer_groups <= groups))
    return torch.stack(legal, dim=1)


def collect(teacher, loader, device, max_samples, transfer_groups):
    features, labels, legal_masks, losses_all = [], [], [], []
    count = 0
    with torch.no_grad():
        for images, targets in loader:
            if count >= max_samples:
                break
            images, targets = images[:max_samples - count].to(device), targets[:max_samples - count].to(device)
            logits, diagnostics = teacher(images)
            keep = torch.stack([item.keep_count for item in diagnostics], dim=1)
            legal = future_legal_mask(keep, teacher.min_groups_per_stage, teacher.groups, transfer_groups)
            losses = [F.cross_entropy(logits, targets, reduction="none")]
            for action_index, (donor, recipient) in enumerate(FUTURE_ACTIONS):
                override = keep.clone()
                valid = legal[:, action_index + 1]
                override[valid, donor] -= transfer_groups
                override[valid, recipient] += transfer_groups
                candidate_logits, _ = teacher(images, override)
                losses.append(F.cross_entropy(candidate_logits, targets, reduction="none"))
            candidate_losses = torch.stack(losses, dim=1)
            features.append(layer1_features(teacher, images).cpu())
            labels.append(oracle_action_labels(candidate_losses, legal).cpu())
            legal_masks.append(legal.cpu())
            losses_all.append(candidate_losses.cpu())
            count += targets.numel()
    return tuple(torch.cat(values) for values in (features, labels, legal_masks, losses_all))


def evaluate(head, dataset, majority_label):
    features, labels, legal, losses = dataset
    with torch.no_grad():
        logits = head(features)
        selected = logits.masked_fill(~legal, torch.finfo(logits.dtype).min).argmax(dim=1)
        oracle_nll = losses.gather(1, labels[:, None]).mean().item()
        selected_nll = losses.gather(1, selected[:, None]).mean().item()
    return {"action_accuracy": selected.eq(labels).float().mean().item(),
            "majority_accuracy": labels.eq(majority_label).float().mean().item(),
            "oracle_nll": oracle_nll, "selected_nll": selected_nll,
            "mean_regret": selected_nll - oracle_nll}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--transfer-groups", type=int, default=4)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = build_model(config, args.checkpoint, device)
    train_loader, validation_loader = build_tinyimagenet_loaders(config["data_dir"], config["batch_size"], config["workers"])
    train = collect(teacher, train_loader, device, args.samples, args.transfer_groups)
    validation = collect(teacher, validation_loader, device, args.samples, args.transfer_groups)
    head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 7)).cpu()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.0)
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = masked_action_loss(head(train[0]), train[1], train[2])
        loss.backward()
        optimizer.step()
    majority = int(torch.bincount(train[1], minlength=7).argmax())
    report = {"feature_source": "layer1", "actions": ["noop", *[f"{a}->{b}" for a, b in FUTURE_ACTIONS]],
              "samples_per_split": args.samples, "steps": args.steps, "majority_action": majority,
              "train": evaluate(head, train, majority), "validation": evaluate(head, validation, majority)}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
