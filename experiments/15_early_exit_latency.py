"""Benchmark real prefix latency for adaptive-depth exits."""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from attentionv3.models.early_exit_resnet import EarlyExitResNet50
from attentionv3.models.early_exit_runtime import forward_to_exit


def load_model(config, checkpoint, device):
    model = EarlyExitResNet50(config["num_classes"], small_images=True)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    return model.to(device).eval()


def benchmark(model, images, stage, warmup, repeats):
    with torch.no_grad():
        for _ in range(warmup):
            forward_to_exit(model, images, stage)
        if images.is_cuda:
            torch.cuda.synchronize()
        values = []
        for _ in range(repeats):
            start = time.perf_counter(); forward_to_exit(model, images, stage)
            if images.is_cuda:
                torch.cuda.synchronize()
            values.append((time.perf_counter() - start) * 1000)
    tensor = torch.tensor(values)
    return {"median_ms": tensor.median().item(), "p95_ms": tensor.quantile(0.95).item()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--warmup", type=int, default=30); parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text()); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.checkpoint, device); images = torch.randn(args.batch_size, 3, 64, 64, device=device)
    report = {"device": str(device), "batch_size": args.batch_size, "warmup": args.warmup, "repeats": args.repeats,
              "stage2": benchmark(model, images, 2, args.warmup, args.repeats), "stage3": benchmark(model, images, 3, args.warmup, args.repeats), "stage4": benchmark(model, images, 4, args.warmup, args.repeats)}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
