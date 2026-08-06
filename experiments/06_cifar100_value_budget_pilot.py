"""Short validation-driven pilot for counterfactual stage value-of-compute."""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from attentionv3.data import build_cifar100_train_validation_test_loaders
from attentionv3.models import GlobalValueBudgetResNet50
from attentionv3.training import (cleanup_distributed, counterfactual_stage_value_targets,
                                  initialize_distributed, stage_value_loss)


def evaluate(model, loader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            logits, _ = model(images.to(device)); targets = targets.to(device)
            correct += logits.argmax(1).eq(targets).sum().item(); total += targets.numel()
    values = torch.tensor([correct, total], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized(): dist.all_reduce(values)
    return (values[0] / values[1]).item()


def counterfactual_targets(model, images, targets, groups_to_transfer):
    """Measure loss after transferring one group away from each eligible stage."""
    losses = []
    was_training = model.training; model.eval()
    with torch.no_grad():
        base_logits, base_diagnostics = model(images)
        base_loss = F.cross_entropy(base_logits, targets, reduction="none")
        keep = torch.stack([item.keep_count for item in base_diagnostics], dim=1)
        for stage in range(4):
            override = keep.clone()
            for row in range(override.shape[0]):
                # A valid ablation preserves the configured floor, group cap,
                # and exact global total; otherwise retain the base allocation.
                if override[row, stage] <= model.min_groups_per_stage:
                    continue
                candidates = [index for index in range(4)
                              if index != stage and override[row, index] < model.groups]
                if not candidates:
                    continue
                target = min(candidates, key=lambda index: int(override[row, index]))
                transferable = min(groups_to_transfer, int(override[row, stage] - model.min_groups_per_stage),
                                   int(model.groups - override[row, target]))
                override[row, stage] -= transferable; override[row, target] += transferable
            logits, _ = model(images, override)
            losses.append(F.cross_entropy(logits, targets, reduction="none"))
    model.train(was_training)
    ablated = torch.stack(losses, dim=1)
    return counterfactual_stage_value_targets(base_loss.detach(), ablated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text())
    ctx = initialize_distributed("auto", config["distributed"]["world_size"])
    try:
        torch.manual_seed(config["seed"] + ctx.rank)
        train, validation, test = build_cifar100_train_validation_test_loaders(
            config["data_dir"], config["batch_size"], config["workers"], config["validation_size"],
            config["split_seed"], ctx.enabled, ctx.rank, ctx.world_size)
        a, t = config["attention"], config["training"]
        model = GlobalValueBudgetResNet50(config["num_classes"], a["groups_per_stage"], a["hidden"], a["budget"],
                                          a.get("min_groups_per_stage", 1), a.get("allocation", "value"), a.get("quota_temperature", 1.0))
        model.backbone.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False); model.backbone.maxpool = torch.nn.Identity()
        model = model.to(ctx.device)
        if ctx.enabled: model = DistributedDataParallel(model, device_ids=[ctx.local_rank], find_unused_parameters=True)
        if t.get("optimizer", "adamw") == "muon_hybrid":
            muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim == 2]
            adamw_params = [p for p in model.parameters() if p.requires_grad and p.ndim != 2]
            optimizers = [torch.optim.Muon(muon_params, lr=t["learning_rate"], weight_decay=t["weight_decay"]),
                          torch.optim.AdamW(adamw_params, lr=t["learning_rate"], weight_decay=t["weight_decay"])]
        else:
            optimizers = [torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])]
        output = Path(args.output_dir)
        if ctx.is_main: output.mkdir(parents=True, exist_ok=True)
        history = []; raw = model.module if ctx.enabled else model
        for epoch in range(t["epochs"]):
            if hasattr(train.sampler, "set_epoch"): train.sampler.set_epoch(epoch)
            model.train(); total = correct = value_total = value_batches = batches = agreement = entropy_total = 0
            keep_total = torch.zeros(4, device=ctx.device)
            for batch_index, (images, targets) in enumerate(train):
                images, targets = images.to(ctx.device), targets.to(ctx.device)
                [optimizer.zero_grad(set_to_none=True) for optimizer in optimizers]; logits, diagnostics = model(images)
                classification = F.cross_entropy(logits, targets); value = logits.new_zeros(())
                if a.get("allocation", "value") == "value" and batch_index % t["counterfactual_every"] == 0:
                    target = counterfactual_targets(raw, images, targets, t["ablation_groups"])
                    values = torch.stack([item.uncertainty for item in diagnostics], dim=1)
                    value = stage_value_loss(values, target)
                    value_batches += 1
                    agreement += values.argmax(1).eq(target.argmax(1)).float().sum().item()
                    entropy_total += (-(target * target.clamp_min(1e-8).log()).sum(1)).sum().item()
                (classification + t["value_weight"] * value).backward(); [optimizer.step() for optimizer in optimizers]
                total += targets.numel(); correct += logits.argmax(1).eq(targets).sum().item()
                value_total += value.detach().item(); batches += 1
                keep_total += torch.stack([item.keep_count.float().mean() for item in diagnostics])
            validation_accuracy = evaluate(raw, validation, ctx.device)
            if ctx.is_main:
                record = {"epoch": epoch + 1, "train_accuracy": correct / total,
                          "value_loss": value_total / max(1, value_batches),
                          "target_entropy": entropy_total / max(1, value_batches * config["batch_size"] * ctx.world_size),
                          "value_top1_agreement": agreement / max(1, value_batches * config["batch_size"] * ctx.world_size),
                          "mean_keep_per_stage": (keep_total / batches).tolist(),
                          "validation_accuracy": validation_accuracy}
                history.append(record); (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                torch.save({"epoch": epoch + 1, "model": raw.state_dict(), "config": config}, output / "latest.pt")
                print(json.dumps(record))
        test_accuracy = evaluate(raw, test, ctx.device) if t.get("evaluate_test", True) else None
        if ctx.is_main and test_accuracy is not None:
            (output / "final.json").write_text(json.dumps({"test_accuracy": test_accuracy}, indent=2) + "\n")
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__": main()
