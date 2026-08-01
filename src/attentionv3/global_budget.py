"""Global, per-image allocation of channel-group budgets across CNN stages."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class GlobalBudgetOutput:
    """Hard group masks whose total count is fixed for every image."""

    gate: Tensor  # [batch, stages, groups]
    keep_count: Tensor  # [batch, stages]
    stage_uncertainty: Tensor  # [batch, stages]


class GlobalBudgetAllocator:
    """Allocate one fixed budget across stages, favouring uncertain stages.

    Each stage keeps at least one group. Remaining groups are assigned one at
    a time to the currently most uncertain eligible stage, which makes the
    total budget exact rather than merely bounded independently per layer.
    """

    def __init__(self, stages: int, groups: int, budget: float) -> None:
        if stages < 1 or groups < 1:
            raise ValueError("stages and groups must be positive")
        if not 0 < budget <= 1:
            raise ValueError("budget must be in (0, 1]")
        self.stages, self.groups = stages, groups
        self.total_keep = max(stages, round(stages * groups * budget))

    def __call__(self, utility: Tensor, uncertainty_logits: Tensor) -> GlobalBudgetOutput:
        if utility.shape != uncertainty_logits.shape or utility.ndim != 3:
            raise ValueError("utility and uncertainty_logits must both be [batch, stages, groups]")
        batch, stages, groups = utility.shape
        if stages != self.stages or groups != self.groups:
            raise ValueError("inputs do not match allocator dimensions")
        uncertainty = torch.sigmoid(uncertainty_logits)
        stage_uncertainty = uncertainty.mean(dim=2)
        keep = torch.ones(batch, stages, dtype=torch.long, device=utility.device)
        for row in range(batch):
            for _ in range(self.total_keep - stages):
                eligible = keep[row] < groups
                scores = stage_uncertainty[row].masked_fill(~eligible, float("-inf"))
                keep[row, scores.argmax()] += 1
        gate = torch.zeros_like(utility)
        for row in range(batch):
            for stage in range(stages):
                selected = utility[row, stage].topk(int(keep[row, stage])).indices
                gate[row, stage, selected] = 1.0
        return GlobalBudgetOutput(gate, keep, stage_uncertainty)
