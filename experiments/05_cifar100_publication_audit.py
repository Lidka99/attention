"""Held-out, paired audit of completed CIFAR-100 v3 checkpoints."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from attentionv3.data.cifar100 import cifar100_transforms
from attentionv3.evaluation import benchmark_latency, bootstrap_mean_ci, paired_accuracy_delta


def pilot_module():
    path = Path(__file__).with_name("03_cifar100_pilot.py")
    spec = importlib.util.spec_from_file_location("cifar_pilot", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def predictions(model, loader, device):
    model.eval(); pred = []; target = []
    for images, labels in loader:
        logits, _ = model(images.to(device))
        pred.append(logits.argmax(1).cpu()); target.append(labels.cpu())
    return torch.cat(pred), torch.cat(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=CONFIG,CHECKPOINT")
    parser.add_argument("--output", default="results/cifar100/publication_audit_seed42.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, transform = cifar100_transforms()
    data = CIFAR100("/home/users/s224574/attention/attention_v2/data", train=False, download=False, transform=transform)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    report = {"protocol": {"dataset": "CIFAR-100 test", "samples": len(data), "bootstrap_repeats": args.repeats, "device": str(device)}, "models": {}}
    all_predictions, targets = {}, None
    for spec in args.run:
        label, paths = spec.split("=", 1); config_path, checkpoint_path = paths.split(",", 1)
        config = yaml.safe_load(Path(config_path).read_text())
        model = pilot_module().cifar_resnet50(config).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"])
        pred, target = predictions(model, loader, device); targets = target; correct = pred.eq(target).float()
        all_predictions[label] = pred
        report["models"][label] = {"accuracy": correct.mean().item(), "bootstrap_95ci": list(bootstrap_mean_ci(correct, args.repeats)),
                                   "per_class_accuracy": {str(k): correct[target == k].mean().item() for k in range(100)},
                                   "parameters": sum(p.numel() for p in model.parameters()),
                                   "latency": benchmark_latency(model, torch.randn(1, 3, 32, 32, device=device))}
    if "dense" in all_predictions:
        report["paired_vs_dense"] = {label: paired_accuracy_delta(all_predictions["dense"], pred, targets, args.repeats)
                                      for label, pred in all_predictions.items() if label != "dense"}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
