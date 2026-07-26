"""Post-training calibration and reproducible evaluation metrics."""

from __future__ import annotations

import time
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TemperatureScaler(nn.Module):
    """Single scalar temperature fit only on a dedicated calibration split."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        self.log_temperature = nn.Parameter(torch.tensor(initial_temperature).log())

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature.clamp_min(1e-6)

    def fit(self, logits: Tensor, targets: Tensor, max_iter: int = 50) -> float:
        """Fit temperature by NLL minimisation without changing the classifier."""
        if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
            raise ValueError("expected logits [n, classes] and targets [n]")
        logits, targets = logits.detach(), targets.detach()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self(logits), targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.detach().item()


@torch.no_grad()
def collect_logits(model: nn.Module, loader: Iterable[object],
                   device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
    """Collect logits once, allowing calibration and reports to share outputs."""
    model.eval()
    device = torch.device(device)
    logits, targets = [], []
    for images, labels in loader:
        output = model(images.to(device))
        if isinstance(output, tuple):
            output = output[0]
        logits.append(output.cpu())
        targets.append(labels.cpu())
    if not logits:
        raise ValueError("loader yielded no batches")
    return torch.cat(logits), torch.cat(targets)


def expected_calibration_error(logits: Tensor, targets: Tensor, bins: int = 15) -> float:
    """Top-label ECE with equally spaced confidence bins."""
    if bins < 1:
        raise ValueError("bins must be positive")
    probabilities = logits.softmax(dim=1)
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(targets).float()
    ece = logits.new_zeros(())
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & ((confidence < high) if index < bins - 1 else (confidence <= high))
        if mask.any():
            ece += mask.float().mean() * (confidence[mask].mean() - correct[mask].mean()).abs()
    return ece.item()


def compute_metrics(logits: Tensor, targets: Tensor, bins: int = 15) -> dict[str, float]:
    """Return JSON-ready classification and calibration metrics."""
    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("expected logits [n, classes] and targets [n]")
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(targets, num_classes=logits.shape[1]).to(probabilities.dtype)
    top1 = logits.argmax(dim=1).eq(targets).float().mean().item()
    top5_width = min(5, logits.shape[1])
    top5 = logits.topk(top5_width, dim=1).indices.eq(targets[:, None]).any(dim=1).float().mean().item()
    return {
        "top1": top1,
        "top5": top5,
        "nll": F.cross_entropy(logits, targets).item(),
        "brier_multiclass": (probabilities - one_hot).square().sum(dim=1).mean().item(),
        "ece": expected_calibration_error(logits, targets, bins),
        "samples": int(targets.numel()),
    }


def benchmark_latency(model: nn.Module, sample: Tensor, warmup: int = 10,
                      runs: int = 50) -> dict[str, float]:
    """Measure batch-1 inference latency; reports median and p95 in ms."""
    if warmup < 0 or runs < 1:
        raise ValueError("warmup must be non-negative and runs positive")
    model.eval()
    device = sample.device

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        synchronize()
        values = []
        for _ in range(runs):
            start = time.perf_counter()
            model(sample)
            synchronize()
            values.append((time.perf_counter() - start) * 1000)
    timings = torch.tensor(values)
    return {
        "median_ms": timings.median().item(),
        "p95_ms": torch.quantile(timings, 0.95).item(),
        "mean_ms": timings.mean().item(),
        "runs": runs,
    }
