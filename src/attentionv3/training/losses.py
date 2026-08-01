"""Training objectives for uncertainty-calibrated adaptive inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from attentionv3.budget_controller import BudgetOutput


@dataclass
class UCLALossBreakdown:
    """Individual terms are returned for transparent experiment logging."""

    total: Tensor
    classification: Tensor
    distillation: Tensor
    brier: Tensor
    budget: Tensor


def distillation_loss(student_logits: Tensor, teacher_logits: Tensor,
                      temperature: float = 2.0) -> Tensor:
    """Temperature-scaled KL distillation loss with a detached teacher."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=1)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * temperature**2


def brier_error_loss(logits: Tensor, targets: Tensor,
                     diagnostics: Sequence[BudgetOutput], target: str = "hard_error") -> Tensor:
    """Train uncertainty as the probability that the final prediction is wrong.

    The correctness target is detached because it is a label for the
    uncertainty head, not a differentiable replacement for cross-entropy.
    Uncertainty is averaged across stages to avoid privileging one layer.
    """
    if not diagnostics:
        return logits.new_zeros(())
    if target == "hard_error":
        predicted = logits.detach().argmax(dim=1)
        error_target = (predicted != targets).float()
    elif target == "soft_error":
        error_target = 1.0 - logits.detach().softmax(dim=1).gather(1, targets[:, None]).squeeze(1)
    else:
        raise ValueError("target must be hard_error or soft_error")
    uncertainty = torch.stack([item.uncertainty for item in diagnostics]).mean(dim=0)
    return F.mse_loss(uncertainty, error_target)


def budget_loss(diagnostics: Sequence[BudgetOutput], target_budget: float,
                groups: int) -> Tensor:
    """Penalise deviation of mean active-group ratio from the target budget."""
    if not 0 < target_budget <= 1:
        raise ValueError("target_budget must be in (0, 1]")
    if groups < 1:
        raise ValueError("groups must be positive")
    if not diagnostics:
        return torch.tensor(0.0)
    ratios = torch.stack([item.keep_count.float().mean() / groups for item in diagnostics])
    return (ratios.mean() - target_budget).square()


class UCLALoss(nn.Module):
    """Combine task, teacher, uncertainty and compute-budget objectives."""

    def __init__(self, target_budget: float = 0.65, groups: int = 16,
                 distillation_weight: float = 0.5, brier_weight: float = 0.1,
                 budget_weight: float = 0.01, temperature: float = 2.0,
                 uncertainty_target: str = "hard_error") -> None:
        super().__init__()
        self.target_budget = target_budget
        self.groups = groups
        self.distillation_weight = distillation_weight
        self.brier_weight = brier_weight
        self.budget_weight = budget_weight
        self.temperature = temperature
        self.uncertainty_target = uncertainty_target

    def forward(self, logits: Tensor, targets: Tensor,
                diagnostics: Sequence[BudgetOutput],
                teacher_logits: Tensor | None = None) -> UCLALossBreakdown:
        classification = F.cross_entropy(logits, targets)
        distillation = (distillation_loss(logits, teacher_logits, self.temperature)
                        if teacher_logits is not None else logits.new_zeros(()))
        brier = brier_error_loss(logits, targets, diagnostics, self.uncertainty_target)
        budget = budget_loss(diagnostics, self.target_budget, self.groups).to(logits.device)
        total = (classification + self.distillation_weight * distillation
                 + self.brier_weight * brier + self.budget_weight * budget)
        return UCLALossBreakdown(total, classification, distillation, brier, budget)
