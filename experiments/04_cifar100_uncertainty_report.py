"""Report CIFAR-100 classification and controller-uncertainty diagnostics."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from attentionv3.data.cifar100 import cifar100_transforms
from attentionv3.evaluation import compute_metrics


def load_pilot_module():
    path = Path(__file__).with_name("03_cifar100_pilot.py")
    spec = importlib.util.spec_from_file_location("cifar_pilot", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    logits, targets, uncertainty = [], [], []
    for images, labels in loader:
        output, diagnostics = model(images.to(device))
        logits.append(output.cpu())
        targets.append(labels.cpu())
        if diagnostics:
            uncertainty.append(torch.stack([item.uncertainty for item in diagnostics]).mean(dim=0).cpu())
    return torch.cat(logits), torch.cat(targets), (torch.cat(uncertainty) if uncertainty else None)


def pearson(x, y):
    x, y = x.float(), y.float()
    return ((x - x.mean()) * (y - y.mean())).mean().div((x.std(unbiased=False) * y.std(unbiased=False)).clamp_min(1e-8)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    _, test_transform = cifar100_transforms()
    dataset = CIFAR100(config["data_dir"], train=False, download=False, transform=test_transform)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["workers"], pin_memory=True)
    model = load_pilot_module().cifar_resnet50(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    logits, targets, uncertainty = collect(model, loader, device)
    report = {"classification": compute_metrics(logits, targets), "samples": int(targets.numel())}
    if uncertainty is not None:
        errors = logits.argmax(dim=1).ne(targets).float()
        report["controller_uncertainty"] = {
            "mean": uncertainty.mean().item(),
            "brier_error": torch.mean((uncertainty - errors).square()).item(),
            "pearson_error": pearson(uncertainty, errors),
            "mean_on_correct": uncertainty[errors == 0].mean().item(),
            "mean_on_error": uncertainty[errors == 1].mean().item(),
        }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "uncertainty_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
