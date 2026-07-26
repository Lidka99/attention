"""CNN adapter for the uncertainty-calibrated budget controller."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .budget_controller import BudgetController, BudgetOutput


class UCLAChannelAttention(nn.Module):
    """Predict utility and uncertainty for groups of convolutional channels."""

    def __init__(self, channels: int, groups: int, hidden: int = 64,
                 budget: float = 0.65, max_budget: float = 0.90,
                 uncertainty_weight: float = 0.5, adaptive_extra: float = 0.25,
                 temperature: float = 0.5, mode: str = "ucla") -> None:
        super().__init__()
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        if mode not in {"ucla", "utility_only", "static"}:
            raise ValueError("mode must be ucla, utility_only, or static")
        self.channels = channels
        self.groups = groups
        self.channels_per_group = channels // groups
        self.mode = mode
        self.utility_head = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, groups)
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, groups)
        )
        # In ablations, uncertainty must not leak through either the score
        # or the sample-adaptive extra budget.
        controller_uncertainty_weight = uncertainty_weight if mode == "ucla" else 0.0
        controller_adaptive_extra = adaptive_extra if mode == "ucla" else 0.0
        self.controller = BudgetController(
            groups, budget, max_budget, controller_uncertainty_weight,
            controller_adaptive_extra, temperature
        )
        self.static_utility = nn.Parameter(torch.zeros(groups)) if mode == "static" else None

    def forward(self, x: Tensor) -> tuple[Tensor, BudgetOutput]:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected [batch, {self.channels}, height, width] input")
        # Global average pooling gives both heads an image-dependent summary.
        pooled = x.mean(dim=(2, 3))
        if self.mode == "static":
            # A learned vector is shared by all images: this is the static
            # channel-selection baseline, not dynamic attention.
            utility = self.static_utility.unsqueeze(0).expand(x.shape[0], -1)
            uncertainty_logits = utility.new_zeros(utility.shape)
        else:
            utility = self.utility_head(pooled)
            uncertainty_logits = (self.uncertainty_head(pooled) if self.mode == "ucla"
                                  else torch.zeros_like(utility))
        decision = self.controller(utility, uncertainty_logits)
        channel_gate = decision.gate.repeat_interleave(self.channels_per_group, dim=1)
        return x * channel_gate[:, :, None, None], decision
