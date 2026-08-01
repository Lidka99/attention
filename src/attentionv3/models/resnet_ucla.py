"""ResNet-50 integration for the UCLA channel controller."""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn
from torchvision.models import ResNet, resnet50

from attentionv3.budget_controller import BudgetOutput
from attentionv3.channel_attention import UCLAChannelAttention
from attentionv3.global_budget import GlobalBudgetAllocator


class UCLAResNet50(nn.Module):
    """ImageNet ResNet-50 with sample-adaptive attention after each stage.

    The torchvision backbone is kept intact. Attention is inserted after
    ``layer1`` through ``layer4`` so residual dimensions and pretrained
    checkpoint compatibility remain easy to inspect. The returned diagnostics
    are intentionally exposed for later calibration and Pareto analysis.
    """

    stage_channels = (256, 512, 1024, 2048)

    def __init__(self, num_classes: int = 1000, groups: int = 16, hidden: int = 64,
                 budget: float = 0.65, max_budget: float = 0.90,
                 uncertainty_weight: float = 0.5, adaptive_extra: float = 0.25,
                 temperature: float = 0.5, mode: str = "ucla",
                 backbone: ResNet | None = None) -> None:
        super().__init__()
        self.backbone = backbone if backbone is not None else resnet50(weights=None)
        self.attention = nn.ModuleList([
            UCLAChannelAttention(channels, groups, hidden, budget, max_budget,
                                 uncertainty_weight, adaptive_extra, temperature, mode)
            for channels in self.stage_channels
        ])
        # Keep the classifier explicit so changing num_classes never mutates
        # torchvision's backbone unexpectedly.
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, List[BudgetOutput]]:
        diagnostics: List[BudgetOutput] = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        for index, stage in enumerate((self.backbone.layer1, self.backbone.layer2,
                                       self.backbone.layer3, self.backbone.layer4)):
            x = stage(x)
            x, decision = self.attention[index](x)
            diagnostics.append(decision)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.backbone.fc(x), diagnostics


class GlobalUCLAResNet50(nn.Module):
    """ResNet-50 with one exact channel-group budget shared by all stages."""

    stage_channels = (256, 512, 1024, 2048)

    def __init__(self, num_classes: int = 1000, groups: int = 16, hidden: int = 64,
                 budget: float = 0.625, mode: str = "ucla", backbone: ResNet | None = None) -> None:
        super().__init__()
        if mode not in {"ucla", "utility_only"}:
            raise ValueError("mode must be ucla or utility_only")
        self.groups, self.mode = groups, mode
        self.backbone = backbone if backbone is not None else resnet50(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        # The stem is available before every stage, enabling a one-pass policy.
        self.policy = nn.Sequential(nn.Linear(64, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 8 * groups))
        self.allocator = GlobalBudgetAllocator(4, groups, budget,
                                               allocation="uncertainty" if mode == "ucla" else "utility")

    def forward(self, x: Tensor) -> tuple[Tensor, List[BudgetOutput]]:
        x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x); x = self.backbone.maxpool(x)
        policy = self.policy(x.mean(dim=(2, 3))).reshape(x.shape[0], 2, 4, self.groups)
        utility, uncertainty_logits = policy[:, 0], policy[:, 1]
        if self.mode == "utility_only":
            uncertainty_logits = torch.zeros_like(utility)
        allocation = self.allocator(utility, uncertainty_logits, straight_through=self.training)
        diagnostics: List[BudgetOutput] = []
        for index, stage in enumerate((self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4)):
            x = stage(x)
            channel_gate = allocation.gate[:, index].repeat_interleave(x.shape[1] // self.groups, dim=1)
            x = x * channel_gate[:, :, None, None]
            diagnostics.append(BudgetOutput(allocation.gate[:, index], utility[:, index], allocation.keep_count[:, index],
                                            allocation.stage_uncertainty[:, index]))
        x = self.backbone.avgpool(x)
        return self.backbone.fc(torch.flatten(x, 1)), diagnostics
