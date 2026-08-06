"""ResNet-50 with a stage-wise value-of-compute policy and exact global budget."""

import torch
from torch import Tensor, nn
from torchvision.models import ResNet, resnet50

from attentionv3.budget_controller import BudgetOutput


class GlobalValueBudgetResNet50(nn.Module):
    """Allocate channel groups to stages using learned marginal-compute value.

    ``stage_values`` are logits learned from counterfactual ablation targets.
    They are deliberately separate from the legacy uncertainty head.
    """

    stage_channels = (256, 512, 1024, 2048)

    def __init__(self, num_classes: int = 1000, groups: int = 16, hidden: int = 64,
                 budget: float = 0.625, min_groups_per_stage: int = 1,
                 allocation: str = "value", quota_temperature: float = 1.0,
                 backbone: ResNet | None = None) -> None:
        super().__init__()
        if not 0 < budget <= 1:
            raise ValueError("budget must be in (0, 1]")
        if not 1 <= min_groups_per_stage <= groups:
            raise ValueError("min_groups_per_stage must be between one and groups")
        if allocation not in {"value", "utility"}:
            raise ValueError("allocation must be value or utility")
        if quota_temperature <= 0:
            raise ValueError("quota_temperature must be positive")
        self.groups, self.min_groups_per_stage = groups, min_groups_per_stage
        self.allocation = allocation
        self.quota_temperature = quota_temperature
        self.total_keep = max(4, round(4 * groups * budget))
        if self.total_keep < 4 * min_groups_per_stage:
            raise ValueError("budget cannot satisfy min_groups_per_stage")
        self.backbone = backbone if backbone is not None else resnet50(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.policy = nn.Sequential(nn.Linear(64, hidden), nn.ReLU(inplace=True),
                                    nn.Linear(hidden, 4 * groups + 4))

    def _keep_counts(self, stage_values: Tensor, override: Tensor | None) -> Tensor:
        batch = stage_values.shape[0]
        if override is not None:
            if override.shape != (batch, 4) or override.dtype not in (torch.int32, torch.int64):
                raise ValueError("stage_keep_override must be integer [batch, 4]")
            if (override < self.min_groups_per_stage).any() or (override > self.groups).any() or not torch.all(override.sum(1) == self.total_keep):
                raise ValueError("override must respect per-stage limits and exact total budget")
            return override
        keep = torch.full((batch, 4), self.min_groups_per_stage, dtype=torch.long, device=stage_values.device)
        for row in range(batch):
            available = self.total_keep - 4 * self.min_groups_per_stage
            quota = torch.softmax(stage_values[row] / self.quota_temperature, dim=0) * available
            extra = quota.floor().long().clamp(max=self.groups - self.min_groups_per_stage)
            keep[row] += extra
            for _ in range(self.total_keep - int(keep[row].sum())):
                eligible = keep[row] < self.groups
                remainder = (quota - extra).masked_fill(~eligible, float("-inf"))
                chosen = remainder.argmax(); keep[row, chosen] += 1
        return keep

    def forward(self, x: Tensor, stage_keep_override: Tensor | None = None):
        x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x); x = self.backbone.maxpool(x)
        policy = self.policy(x.mean(dim=(2, 3)))
        utility, stage_values = policy[:, :4 * self.groups].reshape(x.shape[0], 4, self.groups), policy[:, 4 * self.groups:]
        allocation_score = stage_values if self.allocation == "value" else utility.mean(dim=2)
        keep = self._keep_counts(allocation_score, stage_keep_override)
        diagnostics = []
        for index, stage in enumerate((self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4)):
            x = stage(x)
            values, selected = utility[:, index].topk(self.groups, dim=1)
            gate = torch.zeros_like(utility[:, index])
            for row in range(x.shape[0]):
                gate[row, selected[row, :keep[row, index]]] = 1.0
            if self.training:
                threshold = values.gather(1, (keep[:, index] - 1).unsqueeze(1))
                soft = torch.sigmoid((utility[:, index] - threshold) / 0.5)
                gate = gate + soft - soft.detach()
            channel_gate = gate.repeat_interleave(x.shape[1] // self.groups, dim=1)
            x = x * channel_gate[:, :, None, None]
            diagnostics.append(BudgetOutput(gate, utility[:, index], keep[:, index], stage_values[:, index]))
        return self.backbone.fc(torch.flatten(self.backbone.avgpool(x), 1)), diagnostics
