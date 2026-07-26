"""Run a reproducible ImageNet-100 UCLA pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torchvision.models import resnet50

from attentionv3.data import build_imagenet100_loaders
from attentionv3.models import UCLAResNet50
from attentionv3.training import UCLALoss, evaluate, train_one_epoch


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
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])

    train_loader, val_loader, manifest = build_imagenet100_loaders(
        train_dir=Path(config["data_dir"]) / "train",
        val_dir=Path(config["data_dir"]) / "val",
        manifest_path=output_dir / f"imagenet100_seed{config['seed']}_manifest.json",
        seed=config["seed"], batch_size=config["batch_size"], workers=config["workers"],
        image_size=config["image_size"], num_classes=config["num_classes"],
    )
    attention = config["attention"]
    student = UCLAResNet50(num_classes=config["num_classes"], groups=attention["groups_per_stage"],
                           hidden=attention["hidden"], budget=attention["budget"],
                           max_budget=attention["max_budget"],
                           uncertainty_weight=attention["uncertainty_weight"],
                           adaptive_extra=attention["adaptive_extra"],
                           temperature=attention.get("temperature", 0.5)).to(device)
    teacher = load_teacher(args.teacher_checkpoint, config["num_classes"], device) if args.teacher_checkpoint else None
    training = config["training"]
    criterion = UCLALoss(target_budget=attention["budget"], groups=attention["groups_per_stage"],
                         distillation_weight=training["distillation_weight"],
                         brier_weight=training["brier_weight"], budget_weight=training["budget_weight"])
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    history = {"manifest": manifest, "epochs": []}
    for epoch in range(training["warmup_epochs"] + training["pilot_epochs"]):
        train = train_one_epoch(student, teacher, train_loader, optimizer, criterion, device)
        validation = evaluate(student, val_loader, criterion, device)
        history["epochs"].append({"epoch": epoch + 1, "train": train.__dict__, "validation": validation.__dict__})
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        torch.save({"epoch": epoch + 1, "model": student.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": config, "manifest": manifest}, output_dir / "latest.pt")
        print(f"epoch={epoch + 1} val_top1={validation.accuracy:.4f} keep={validation.mean_keep_ratio:.4f}")


if __name__ == "__main__":
    main()
