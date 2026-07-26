"""First executable prototype of the uncertainty-aware budget controller."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class BudgetOutput:
    """Gate and diagnostics returned by the controller."""

    gate: Tensor
    score: Tensor
    keep_count: Tensor
    uncertainty: Tensor


class BudgetController(nn.Module):
    """Select channel groups using utility and predicted uncertainty.

    Groups have equal cost in this first prototype. A confident sample gets
    the base budget; an uncertain sample may spend a bounded extra budget.
    During training, a straight-through estimator keeps the hard forward
    decision while using a sigmoid relaxation for the backward pass.
    """

    def __init__(self, groups: int, budget: float = 0.65, max_budget: float = 0.90,
                 uncertainty_weight: float = 0.5, adaptive_extra: float = 0.25,
                 temperature: float = 0.5) -> None:
        super().__init__()
        if groups < 1:
            raise ValueError("groups must be positive")
        if not 0 < budget <= max_budget <= 1:
            raise ValueError("require 0 < budget <= max_budget <= 1")
        if uncertainty_weight < 0 or adaptive_extra < 0:
            raise ValueError("uncertainty parameters must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.groups = groups
        self.budget = budget
        self.max_budget = max_budget
        self.uncertainty_weight = uncertainty_weight
        self.adaptive_extra = adaptive_extra
        self.temperature = temperature

    def forward(self, utility: Tensor, uncertainty_logits: Tensor) -> BudgetOutput:
        """Return a binary per-sample gate and diagnostics."""
        if utility.shape != uncertainty_logits.shape:
            raise ValueError("utility and uncertainty_logits must have the same shape")
        if utility.ndim != 2 or utility.shape[1] != self.groups:
            raise ValueError(f"expected [batch, {self.groups}] inputs")

        # Higher values mean that the utility estimate is less trustworthy.
        uncertainty = torch.sigmoid(uncertainty_logits)
        # The zero-weight ablation is the utility-only baseline.
        score = utility - self.uncertainty_weight * uncertainty

        base_k = max(1, round(self.groups * self.budget))
        max_k = max(base_k, round(self.groups * self.max_budget))
        sample_uncertainty = uncertainty.mean(dim=1)
        extra = torch.round(sample_uncertainty * self.groups * self.adaptive_extra)
        keep_count = (base_k + extra).clamp(max=max_k).long()

        # Hard top-k is explicit so soft-mask sparsity is never reported as
        # actual inference savings. During training, the sigmoid below is
        # used only for the backward pass through a straight-through gate.
        gate = torch.zeros_like(score)
        thresholds = []
        for row, k in enumerate(keep_count.tolist()):
            topk = torch.topk(score[row], k=k, dim=0)
            gate[row, topk.indices] = 1.0
            thresholds.append(topk.values[-1])

        if self.training:
            threshold = torch.stack(thresholds).unsqueeze(1)
            soft_gate = torch.sigmoid((score - threshold) / self.temperature)
            # Forward value is exactly hard top-k; backward follows the
            # smooth relaxation so both prediction heads can be trained.
            gate = gate + soft_gate - soft_gate.detach()

        return BudgetOutput(gate, score, keep_count, sample_uncertainty)
