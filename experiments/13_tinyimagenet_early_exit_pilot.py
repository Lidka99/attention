"""Train auxiliary exits and audit the label-informed potential of adaptive depth."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from attentionv3.data import build_tinyimagenet_loaders
from attentionv3.models.early_exit_resnet import EarlyExitResNet50
from attentionv3.training import cleanup_distributed, initialize_distributed


def evaluate(model, loader, device):
    model.eval(); totals = torch.zeros(6, dtype=torch.float64, device=device)
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device)); targets = targets.to(device)
            correct = [outputs[name].argmax(1).eq(targets) for name in ("stage2", "stage3", "stage4")]
            totals[:3] += torch.tensor([item.sum() for item in correct], dtype=torch.float64, device=device)
            earliest = torch.where(correct[0], 0.4375, torch.where(correct[1], 0.8125, 1.0))
            totals[3] += (correct[0] | correct[1] | correct[2]).sum()
            totals[4] += earliest.sum(); totals[5] += targets.numel()
    if dist.is_available() and dist.is_initialized(): dist.all_reduce(totals)
    return {"stage2_accuracy": (totals[0] / totals[5]).item(), "stage3_accuracy": (totals[1] / totals[5]).item(),
            "stage4_accuracy": (totals[2] / totals[5]).item(), "oracle_accuracy": (totals[3] / totals[5]).item(),
            "oracle_mean_cost_fraction": (totals[4] / totals[5]).item()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text()); ctx = initialize_distributed("auto", config["distributed"]["world_size"])
    try:
        torch.manual_seed(config["seed"] + ctx.rank)
        train, validation = build_tinyimagenet_loaders(config["data_dir"], config["batch_size"], config["workers"], ctx.enabled, ctx.rank, ctx.world_size)
        model = EarlyExitResNet50(config["num_classes"], small_images=True).to(ctx.device)
        if ctx.enabled: model = DistributedDataParallel(model, device_ids=[ctx.local_rank])
        raw = model.module if ctx.enabled else model; training = config["training"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
        output = Path(args.output_dir); history = []
        if ctx.is_main: output.mkdir(parents=True, exist_ok=True)
        for epoch in range(training["epochs"]):
            if hasattr(train.sampler, "set_epoch"): train.sampler.set_epoch(epoch)
            model.train(); correct = total = 0
            for images, targets in train:
                images, targets = images.to(ctx.device), targets.to(ctx.device); optimizer.zero_grad(set_to_none=True)
                outputs = model(images); weights = training["loss_weights"]
                loss = sum(weight * F.cross_entropy(outputs[name], targets) for weight, name in zip(weights, ("stage2", "stage3", "stage4")))
                loss.backward(); optimizer.step(); correct += outputs["stage4"].argmax(1).eq(targets).sum().item(); total += targets.numel()
            metrics = evaluate(raw, validation, ctx.device)
            if ctx.is_main:
                record = {"epoch": epoch + 1, "train_stage4_accuracy": correct / total, **metrics}; history.append(record)
                (output / "history.json").write_text(json.dumps(history, indent=2) + "\n"); torch.save({"epoch": epoch + 1, "model": raw.state_dict(), "config": config}, output / "latest.pt"); print(json.dumps(record))
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
