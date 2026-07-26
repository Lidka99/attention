"""CNN adapter for the uncertainty-calibrated budget controller."""

from __future__ import annotations

from torch import Tensor, nn

from .budget_controller import BudgetController, BudgetOutput


class UCLAChannelAttention(nn.Module):
    """Predict utility and uncertainty for groups of convolutional channels."""

    def __init__(self, channels: int, groups: int, hidden: int = 64,
                 budget: float = 0.65, max_budget: float = 0.90,
                 uncertainty_weight: float = 0.5, adaptive_extra: float = 0.25,
                 temperature: float = 0.5) -> None:
        super().__init__()
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        self.channels = channels
        self.groups = groups
        self.channels_per_group = channels // groups
        self.utility_head = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, groups)
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, groups)
        )
        self.controller = BudgetController(
            groups, budget, max_budget, uncertainty_weight, adaptive_extra, temperature
        )

    def forward(self, x: Tensor) -> tuple[Tensor, BudgetOutput]:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected [batch, {self.channels}, height, width] input")
        # Global average pooling gives both heads an image-dependent summary.
        pooled = x.mean(dim=(2, 3))
        utility = self.utility_head(pooled)
        uncertainty_logits = self.uncertainty_head(pooled)
        decision = self.controller(utility, uncertainty_logits)
        channel_gate = decision.gate.repeat_interleave(self.channels_per_group, dim=1)
        return x * channel_gate[:, :, None, None], decision
