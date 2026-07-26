"""Small, explicit training loop for the ImageNet UCLA pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .losses import UCLALoss


@dataclass
class EpochMetrics:
    """Aggregated quantities recorded for a train or validation epoch."""

    loss: float
    accuracy: float
    classification: float
    distillation: float
    brier: float
    budget: float
    mean_keep_ratio: float
    samples: int


def _unpack_batch(batch: object) -> tuple[Tensor, Tensor]:
    """Accept the standard ``(images, targets)`` pair from a DataLoader."""
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("each batch must be an (images, targets) pair")
    images, targets = batch
    if not isinstance(images, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("images and targets must be tensors")
    return images, targets


@torch.no_grad()
def evaluate(model: nn.Module, loader: Iterable[object], criterion: UCLALoss,
             device: torch.device | str = "cpu") -> EpochMetrics:
    """Evaluate accuracy and the same loss terms used in training."""
    model.eval()
    return _run_epoch(model, loader, criterion, None, device)


def train_one_epoch(model: nn.Module, teacher: nn.Module | None,
                    loader: Iterable[object], optimizer: torch.optim.Optimizer,
                    criterion: UCLALoss,
                    device: torch.device | str = "cpu") -> EpochMetrics:
    """Train one epoch; the teacher is always frozen and evaluated."""
    model.train()
    if teacher is not None:
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    return _run_epoch(model, loader, criterion, optimizer, device, teacher)


def _run_epoch(model: nn.Module, loader: Iterable[object], criterion: UCLALoss,
               optimizer: torch.optim.Optimizer | None,
               device: torch.device | str,
               teacher: nn.Module | None = None) -> EpochMetrics:
    device = torch.device(device)
    totals = {name: 0.0 for name in (
        "loss", "classification", "distillation", "brier", "budget", "correct", "keep_ratio"
    )}
    samples = 0

    for batch in loader:
        images, targets = _unpack_batch(batch)
        images, targets = images.to(device), targets.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        logits, diagnostics = model(images)
        with torch.no_grad():
            teacher_logits = teacher(images) if teacher is not None else None
            if isinstance(teacher_logits, tuple):
                teacher_logits = teacher_logits[0]
        breakdown = criterion(logits, targets, diagnostics, teacher_logits)

        if optimizer is not None:
            breakdown.total.backward()
            optimizer.step()

        batch_size = targets.shape[0]
        samples += batch_size
        totals["loss"] += breakdown.total.detach().item() * batch_size
        totals["classification"] += breakdown.classification.detach().item() * batch_size
        totals["distillation"] += breakdown.distillation.detach().item() * batch_size
        totals["brier"] += breakdown.brier.detach().item() * batch_size
        totals["budget"] += breakdown.budget.detach().item() * batch_size
        totals["correct"] += (logits.detach().argmax(dim=1) == targets).sum().item()
        stage_ratios = [item.keep_count.float().mean().item() / criterion.groups for item in diagnostics]
        totals["keep_ratio"] += (sum(stage_ratios) / len(stage_ratios)) * batch_size

    if samples == 0:
        raise ValueError("loader yielded no batches")
    if dist.is_available() and dist.is_initialized():
        names = ("loss", "classification", "distillation", "brier", "budget", "correct", "keep_ratio")
        reduced = torch.tensor([totals[name] for name in names] + [samples], dtype=torch.float64, device=device)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        for index, name in enumerate(names):
            totals[name] = reduced[index].item()
        samples = int(reduced[-1].item())
    return EpochMetrics(
        loss=totals["loss"] / samples,
        accuracy=totals["correct"] / samples,
        classification=totals["classification"] / samples,
        distillation=totals["distillation"] / samples,
        brier=totals["brier"] / samples,
        budget=totals["budget"] / samples,
        mean_keep_ratio=totals["keep_ratio"] / samples,
        samples=samples,
    )
