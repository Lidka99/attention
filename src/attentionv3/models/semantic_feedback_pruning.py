"""Training-only semantic feedback for deployable structured channel pruning.

The feedback path is intentionally absent from ``deploy`` inference.  It may
shape a channel-selection score during optimisation, but validation uses only
the learned, sample-independent scores.  This makes the experiment falsifiable
as a static structured-pruning method rather than another dynamic gate.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import resnet50


class SemanticFeedbackPrunedResNet50(nn.Module):
    """ResNet-50 with a static stage-1 group mask and optional deep feedback.

    A first, unmasked pass produces a detached stage-4 semantic summary.  Its
    small feedback head can alter stage-1 scores only while training.  The
    second pass performs classification through a hard, grouped mask.  At
    evaluation the first pass and feedback head are skipped entirely.
    """

    stage1_channels = 256

    def __init__(self, num_classes: int, groups: int = 16, keep_ratio: float = 0.625,
                 feedback_scale: float = 1.0, small_images: bool = False) -> None:
        super().__init__()
        if self.stage1_channels % groups != 0:
            raise ValueError("stage-1 channels must divide evenly into groups")
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        backbone = resnet50(weights=None)
        if small_images:
            backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            backbone.maxpool = nn.Identity()
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        self.pool = backbone.avgpool
        self.classifier = nn.Linear(backbone.fc.in_features, num_classes)
        self.groups, self.keep_ratio, self.feedback_scale = groups, keep_ratio, feedback_scale
        self.static_scores = nn.Parameter(torch.zeros(groups))
        self.feedback_head = nn.Sequential(
            nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, groups)
        )

    @property
    def keep_groups(self) -> int:
        return max(1, round(self.groups * self.keep_ratio))

    def _features_to_stage4(self, images: Tensor) -> Tensor:
        x = self.layer1(self.stem(images))
        x = self.layer2(x); x = self.layer3(x); return self.layer4(x)

    def _hard_gate(self, scores: Tensor) -> Tensor:
        top = scores.topk(self.keep_groups, dim=1).indices
        hard = torch.zeros_like(scores).scatter_(1, top, 1.0)
        if self.training:
            threshold = scores.gather(1, top[:, -1:])
            soft = torch.sigmoid((scores - threshold) / 0.5)
            hard = hard + soft - soft.detach()
        return hard.repeat_interleave(self.stage1_channels // self.groups, dim=1)

    def forward(self, images: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        batch = images.shape[0]
        static = self.static_scores.unsqueeze(0).expand(batch, -1)
        feedback_scores = None
        if self.training and self.feedback_scale:
            # This pass is a training-time teacher only: it cannot leak into
            # eval/deployment latency or make per-image inference decisions.
            with torch.no_grad():
                semantic = self.pool(self._features_to_stage4(images)).flatten(1)
            feedback_scores = self.feedback_head(semantic)
            scores = static + self.feedback_scale * feedback_scores
        else:
            scores = static
        x = self.layer1(self.stem(images))
        gate = self._hard_gate(scores)
        x = x * gate[:, :, None, None]
        x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        logits = self.classifier(self.pool(x).flatten(1))
        diagnostics = {"static_scores": static, "gate": gate, "feedback_scores": feedback_scores}
        return logits, diagnostics
