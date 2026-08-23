"""Screen training-only semantic feedback for static channel selection."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from attentionv3.data import build_tinyimagenet_loaders
from attentionv3.models.semantic_feedback_pruning import SemanticFeedbackPrunedResNet50
from attentionv3.training import cleanup_distributed, initialize_distributed


def evaluate(model, loader, device):
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for images, targets in loader:
            logits, _ = model(images.to(device)); targets = targets.to(device)
            totals += torch.tensor([logits.argmax(1).eq(targets).sum(), targets.numel()], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized(): dist.all_reduce(totals)
    return (totals[0] / totals[1]).item()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text()); ctx = initialize_distributed("auto", config["distributed"]["world_size"])
    try:
        torch.manual_seed(config["seed"] + ctx.rank)
        train, validation = build_tinyimagenet_loaders(config["data_dir"], config["batch_size"], config["workers"], ctx.enabled, ctx.rank, ctx.world_size)
        model = SemanticFeedbackPrunedResNet50(config["num_classes"], **config["model"], small_images=True).to(ctx.device)
        if ctx.enabled: model = DistributedDataParallel(model, device_ids=[ctx.local_rank], find_unused_parameters=True)
        raw = model.module if ctx.enabled else model; training = config["training"]; optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
        output, history = Path(args.output_dir), []
        if ctx.is_main: output.mkdir(parents=True, exist_ok=True)
        for epoch in range(training["epochs"]):
            if hasattr(train.sampler, "set_epoch"): train.sampler.set_epoch(epoch)
            model.train(); correct = total = 0; feedback_magnitude = 0.0
            for images, targets in train:
                images, targets = images.to(ctx.device), targets.to(ctx.device); optimizer.zero_grad(set_to_none=True); logits, info = model(images)
                F.cross_entropy(logits, targets).backward(); optimizer.step(); correct += logits.argmax(1).eq(targets).sum().item(); total += targets.numel()
                if info["feedback_scores"] is not None: feedback_magnitude += info["feedback_scores"].detach().abs().mean().item() * targets.numel()
            deployment_accuracy = evaluate(raw, validation, ctx.device)
            if ctx.is_main:
                record = {"epoch": epoch + 1, "train_accuracy": correct / total, "deployment_validation_accuracy": deployment_accuracy, "feedback_score_abs_mean": feedback_magnitude / total}
                history.append(record); (output / "history.json").write_text(json.dumps(history, indent=2) + "\n"); torch.save({"epoch": epoch + 1, "model": raw.state_dict(), "config": config}, output / "latest.pt"); print(json.dumps(record))
    finally: cleanup_distributed(ctx)


if __name__ == "__main__": main()
