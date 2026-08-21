"""Truncated ResNet-50 baseline for fixed-depth comparisons."""

from torch import Tensor, nn
from torchvision.models import resnet50


class StaticDepthResNet50(nn.Module):
    def __init__(self, num_classes: int, exit_stage: int = 3, small_images: bool = False) -> None:
        super().__init__()
        if exit_stage not in (2, 3, 4):
            raise ValueError("exit_stage must be 2, 3 or 4")
        backbone = resnet50(weights=None)
        if small_images:
            backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); backbone.maxpool = nn.Identity()
        self.exit_stage = exit_stage; self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.layer4 = backbone.layer4 if exit_stage == 4 else None
        channels = {2: 512, 3: 1024, 4: 2048}[exit_stage]
        self.pool, self.head = nn.AdaptiveAvgPool2d((1, 1)), nn.Linear(channels, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        x = self.layer2(self.layer1(self.stem(images)))
        if self.exit_stage >= 3: x = self.layer3(x)
        if self.exit_stage == 4: x = self.layer4(x)
        return self.head(self.pool(x).flatten(1))
