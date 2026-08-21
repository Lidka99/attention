"""Train a fixed stage-3 ResNet baseline on Tiny ImageNet."""
import argparse, json
from pathlib import Path
import torch, torch.distributed as dist, yaml
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from attentionv3.data import build_tinyimagenet_loaders
from attentionv3.models.static_depth_resnet import StaticDepthResNet50
from attentionv3.training import cleanup_distributed, initialize_distributed

def evaluate(model, loader, device):
    model.eval(); totals = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device)); targets = targets.to(device); totals += torch.tensor([logits.argmax(1).eq(targets).sum(), targets.numel()], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized(): dist.all_reduce(totals)
    return (totals[0] / totals[1]).item()

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True); args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text()); ctx = initialize_distributed("auto", config["distributed"]["world_size"])
    try:
        torch.manual_seed(config["seed"] + ctx.rank); train, validation = build_tinyimagenet_loaders(config["data_dir"], config["batch_size"], config["workers"], ctx.enabled, ctx.rank, ctx.world_size)
        model = StaticDepthResNet50(config["num_classes"], 3, True).to(ctx.device)
        if ctx.enabled: model = DistributedDataParallel(model, device_ids=[ctx.local_rank])
        raw = model.module if ctx.enabled else model; t = config["training"]; optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"]); output = Path(args.output_dir); history = []
        if ctx.is_main: output.mkdir(parents=True, exist_ok=True)
        for epoch in range(t["epochs"]):
            if hasattr(train.sampler, "set_epoch"): train.sampler.set_epoch(epoch)
            model.train(); correct = total = 0
            for images, targets in train:
                images, targets = images.to(ctx.device), targets.to(ctx.device); optimizer.zero_grad(set_to_none=True); logits = model(images); F.cross_entropy(logits, targets).backward(); optimizer.step(); correct += logits.argmax(1).eq(targets).sum().item(); total += targets.numel()
            validation_accuracy = evaluate(raw, validation, ctx.device)
            if ctx.is_main:
                record = {"epoch": epoch + 1, "train_accuracy": correct / total, "validation_accuracy": validation_accuracy}; history.append(record); (output / "history.json").write_text(json.dumps(history, indent=2) + "\n"); print(json.dumps(record))
    finally: cleanup_distributed(ctx)
if __name__ == "__main__": main()
