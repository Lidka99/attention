"""Bootstrap confidence intervals for held-out model comparisons."""

import torch
from torch import Tensor


def bootstrap_mean_ci(values: Tensor, repeats: int = 1000, seed: int = 12345) -> tuple[float, float]:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("values must be a non-empty vector")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(values.numel(), (repeats, values.numel()), generator=generator)
    estimates = values[indices].float().mean(dim=1)
    return torch.quantile(estimates, 0.025).item(), torch.quantile(estimates, 0.975).item()


def paired_accuracy_delta(reference_predictions: Tensor, candidate_predictions: Tensor,
                          targets: Tensor, repeats: int = 1000, seed: int = 12345) -> dict[str, float | list[float]]:
    if reference_predictions.shape != candidate_predictions.shape or targets.shape != reference_predictions.shape:
        raise ValueError("predictions and targets must have equal one-dimensional shapes")
    delta = candidate_predictions.eq(targets).float() - reference_predictions.eq(targets).float()
    low, high = bootstrap_mean_ci(delta, repeats, seed)
    return {"accuracy_delta": delta.mean().item(), "bootstrap_95ci": [low, high],
            "reference_correct_candidate_wrong": int((reference_predictions.eq(targets) & candidate_predictions.ne(targets)).sum()),
            "reference_wrong_candidate_correct": int((reference_predictions.ne(targets) & candidate_predictions.eq(targets)).sum())}
