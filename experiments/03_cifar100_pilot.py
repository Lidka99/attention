"""Four-GPU CIFAR-100 smoke pilot for the UCLA training pipeline."""
import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchvision.models import resnet50

from attentionv3.data import build_cifar100_loaders
from attentionv3.models import GlobalUCLAResNet50, UCLAResNet50
from attentionv3.training import (BudgetCurriculum, UCLALoss, apply_curriculum, cleanup_distributed,
                                  evaluate, initialize_distributed, train_one_epoch)


def cifar_resnet50(config):
    a = config["attention"]
    if a.get("mode") == "none":
        backbone = resnet50(weights=None, num_classes=config["num_classes"])
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        class Baseline(nn.Module):
            def __init__(self, network):
                super().__init__()
                self.network = network

            def forward(self, images):
                return self.network(images), []

        return Baseline(backbone)
    if a.get("global_budget", False):
        model = GlobalUCLAResNet50(config["num_classes"], a["groups_per_stage"], a["hidden"],
                                   a["budget"], a["mode"])
        model.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.backbone.maxpool = torch.nn.Identity()
        return model
    model = UCLAResNet50(config["num_classes"], a["groups_per_stage"], a["hidden"], a["budget"],
                         a["max_budget"], a["uncertainty_weight"], a["adaptive_extra"], mode=a["mode"])
    model.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.backbone.maxpool = torch.nn.Identity()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar100_ucla.yaml")
    parser.add_argument("--output-dir", default="results/cifar100/seed42/ucla")
    parser.add_argument("--seed", type=int, default=None, help="Override the seed stored in the YAML config.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    ctx = initialize_distributed("auto", config["distributed"]["world_size"])
    try:
        torch.manual_seed(config["seed"] + ctx.rank)
        train, test = build_cifar100_loaders(config["data_dir"], config["batch_size"], config["workers"],
                                              ctx.enabled, ctx.rank, ctx.world_size)
        model = cifar_resnet50(config).to(ctx.device)
        if ctx.enabled:
            model = DistributedDataParallel(model, device_ids=[ctx.local_rank], find_unused_parameters=True)
        a, t = config["attention"], config["training"]
        criterion = UCLALoss(a.get("budget", 1.0), a.get("groups_per_stage", 1),
                             brier_weight=t["brier_weight"], budget_weight=t["budget_weight"],
                             uncertainty_target=t.get("uncertainty_target", "hard_error"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        schedule = BudgetCurriculum(t["warmup_epochs"], t["pilot_epochs"], t["fine_tune_epochs"],
                                    a.get("budget", 1.0), a.get("max_budget", 1.0),
                                    t["brier_weight"], t["budget_weight"])
        output = Path(args.output_dir)
        if ctx.is_main: output.mkdir(parents=True, exist_ok=True)
        history = []
        for epoch in range(schedule.total_epochs):
            state = schedule.state_for_epoch(epoch); apply_curriculum(model, criterion, state)
            if hasattr(train.sampler, "set_epoch"): train.sampler.set_epoch(epoch)
            train_metrics = train_one_epoch(model, None, train, optimizer, criterion, ctx.device)
            test_metrics = evaluate(model, test, criterion, ctx.device)
            if ctx.is_main:
                history.append({"epoch": epoch + 1, "phase": state.to_dict(), "train": train_metrics.__dict__, "validation": test_metrics.__dict__})
                (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                torch.save({"model": (model.module if ctx.enabled else model).state_dict(), "config": config}, output / "latest.pt")
                print(f"epoch={epoch + 1} val_top1={test_metrics.accuracy:.4f} keep={test_metrics.mean_keep_ratio:.4f}")
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__": main()
