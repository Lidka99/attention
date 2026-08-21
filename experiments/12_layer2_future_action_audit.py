"""Audit whether layer2 features predict oracle transfers between stages 3 and 4."""

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


ACTIONS = ((2, 3), (3, 2))


def load_model(config, checkpoint, device):
    a = config["attention"]
    model = GlobalValueBudgetResNet50(config["num_classes"], a["groups_per_stage"], a["hidden"], a["budget"],
                                      a.get("min_groups_per_stage", 1), a.get("allocation", "utility"),
                                      a.get("quota_temperature", 1.0))
    model.backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.backbone.maxpool = nn.Identity()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    return model.to(device).eval()


def layer2_features(model, images):
    x = model.backbone.maxpool(model.backbone.relu(model.backbone.bn1(model.backbone.conv1(images))))
    return model.backbone.layer2(model.backbone.layer1(x)).mean(dim=(2, 3))


def collect(model, loader, device, limit, transfer):
    features, labels, masks, losses_all = [], [], [], []
    count = 0
    with torch.no_grad():
        for images, targets in loader:
            if count >= limit:
                break
            images, targets = images[:limit - count].to(device), targets[:limit - count].to(device)
            logits, diagnostics = model(images)
            keep = torch.stack([item.keep_count for item in diagnostics], dim=1)
            legal = torch.stack([torch.ones(keep.shape[0], dtype=torch.bool, device=device),
                                 (keep[:, 2] - transfer >= model.min_groups_per_stage) & (keep[:, 3] + transfer <= model.groups),
                                 (keep[:, 3] - transfer >= model.min_groups_per_stage) & (keep[:, 2] + transfer <= model.groups)], dim=1)
            losses = [F.cross_entropy(logits, targets, reduction="none")]
            for index, (donor, recipient) in enumerate(ACTIONS):
                override = keep.clone(); valid = legal[:, index + 1]
                override[valid, donor] -= transfer; override[valid, recipient] += transfer
                candidate, _ = model(images, override)
                losses.append(F.cross_entropy(candidate, targets, reduction="none"))
            losses = torch.stack(losses, dim=1)
            features.append(layer2_features(model, images).cpu()); labels.append(oracle_action_labels(losses, legal).cpu())
            masks.append(legal.cpu()); losses_all.append(losses.cpu()); count += targets.numel()
    return tuple(torch.cat(items) for items in (features, labels, masks, losses_all))


def evaluate(head, dataset, majority):
    features, labels, legal, losses = dataset
    with torch.no_grad():
        selected = head(features).masked_fill(~legal, torch.finfo(torch.float32).min).argmax(dim=1)
        oracle = losses.gather(1, labels[:, None]).mean().item()
        chosen = losses.gather(1, selected[:, None]).mean().item()
    return {"action_accuracy": selected.eq(labels).float().mean().item(), "majority_accuracy": labels.eq(majority).float().mean().item(),
            "oracle_nll": oracle, "selected_nll": chosen, "mean_regret": chosen - oracle}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=512); parser.add_argument("--steps", type=int, default=300); parser.add_argument("--transfer-groups", type=int, default=4)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text()); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.checkpoint, device)
    train_loader, validation_loader = build_tinyimagenet_loaders(config["data_dir"], config["batch_size"], config["workers"])
    train, validation = collect(model, train_loader, device, args.samples, args.transfer_groups), collect(model, validation_loader, device, args.samples, args.transfer_groups)
    head = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 3)).cpu(); optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.0)
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True); loss = masked_action_loss(head(train[0]), train[1], train[2]); loss.backward(); optimizer.step()
    majority = int(torch.bincount(train[1], minlength=3).argmax())
    report = {"feature_source": "layer2", "actions": ["noop", "2->3", "3->2"], "samples_per_split": args.samples,
              "steps": args.steps, "majority_action": majority, "train": evaluate(head, train, majority), "validation": evaluate(head, validation, majority)}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
