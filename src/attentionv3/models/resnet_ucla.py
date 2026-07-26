"""ResNet-50 integration for the UCLA channel controller."""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn
from torchvision.models import ResNet, resnet50

from attentionv3.budget_controller import BudgetOutput
from attentionv3.channel_attention import UCLAChannelAttention


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
                 temperature: float = 0.5, backbone: ResNet | None = None) -> None:
        super().__init__()
        self.backbone = backbone if backbone is not None else resnet50(weights=None)
        self.attention = nn.ModuleList([
            UCLAChannelAttention(channels, groups, hidden, budget, max_budget,
                                 uncertainty_weight, adaptive_extra, temperature)
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
