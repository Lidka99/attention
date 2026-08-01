"""Run a reproducible ImageNet-100 UCLA pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchvision import datasets
from torchvision.models import resnet50

from attentionv3.data import build_imagenet100_loaders, load_or_create_manifest
from attentionv3.models import UCLAResNet50
from attentionv3.training import (BudgetCurriculum, UCLALoss, apply_curriculum, barrier,
                                    cleanup_distributed, evaluate, initialize_distributed, train_one_epoch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/imagenet100_resnet50.yaml")
    parser.add_argument("--output-dir", default="results/imagenet100_pilot")
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_teacher(path: str, num_classes: int, device: torch.device) -> nn.Module:
    teacher = resnet50(weights=None, num_classes=num_classes)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    teacher.load_state_dict(state_dict)
    return teacher.to(device).eval()


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    distributed_config = config.get("distributed", {})
    context = initialize_distributed(args.device, distributed_config.get("world_size"))
    try:
        output_dir = Path(args.output_dir)
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            train_source = datasets.ImageFolder(str(Path(config["data_dir"]) / "train"))
            load_or_create_manifest(train_source.classes,
                                    output_dir / f"imagenet100_seed{config['seed']}_manifest.json",
                                    config["seed"], config["num_classes"])
        barrier(context)
        torch.manual_seed(config["seed"] + context.rank)
        train_loader, val_loader, manifest = build_imagenet100_loaders(
            train_dir=Path(config["data_dir"]) / "train",
            val_dir=Path(config["data_dir"]) / "val",
            manifest_path=output_dir / f"imagenet100_seed{config['seed']}_manifest.json",
            seed=config["seed"], batch_size=config["batch_size"], workers=config["workers"],
            image_size=config["image_size"], num_classes=config["num_classes"],
            distributed=context.enabled, rank=context.rank, world_size=context.world_size,
        )
        attention = config["attention"]
        student = UCLAResNet50(num_classes=config["num_classes"], groups=attention["groups_per_stage"],
                               hidden=attention["hidden"], budget=attention["budget"],
                               max_budget=attention["max_budget"],
                               uncertainty_weight=attention["uncertainty_weight"],
                               adaptive_extra=attention["adaptive_extra"],
                               temperature=attention.get("temperature", 0.5),
                               mode=attention.get("mode", "ucla")).to(context.device)
        if context.enabled:
            # Static and utility-only ablations intentionally leave some heads
            # unused; DDP must account for their absent gradients.
            student = DistributedDataParallel(student, device_ids=[context.local_rank],
                                              find_unused_parameters=True)
        teacher = load_teacher(args.teacher_checkpoint, config["num_classes"], context.device) if args.teacher_checkpoint else None
        training = config["training"]
        criterion = UCLALoss(target_budget=attention["budget"], groups=attention["groups_per_stage"],
                             distillation_weight=training["distillation_weight"],
                             brier_weight=training["brier_weight"], budget_weight=training["budget_weight"])
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
        curriculum = BudgetCurriculum(training["warmup_epochs"], training["pilot_epochs"],
                                      training["fine_tune_epochs"], attention["budget"],
                                      attention["max_budget"], training["brier_weight"],
                                      training["budget_weight"])
        history = {"manifest": manifest, "world_size": context.world_size, "epochs": []}
        for epoch in range(curriculum.total_epochs):
            state = curriculum.state_for_epoch(epoch)
            apply_curriculum(student, criterion, state)
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            train = train_one_epoch(student, teacher, train_loader, optimizer, criterion, context.device)
            validation = evaluate(student, val_loader, criterion, context.device)
            if context.is_main:
                history["epochs"].append({"epoch": epoch + 1, "phase": state.to_dict(),
                                          "train": train.__dict__, "validation": validation.__dict__})
                (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                model_to_save = student.module if context.enabled else student
                torch.save({"epoch": epoch + 1, "model": model_to_save.state_dict(),
                            "optimizer": optimizer.state_dict(), "config": config, "manifest": manifest},
                           output_dir / "latest.pt")
                print(f"epoch={epoch + 1} val_top1={validation.accuracy:.4f} keep={validation.mean_keep_ratio:.4f}")
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
