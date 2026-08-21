"""ResNet-50 with classifiers after stages 2, 3 and 4 for adaptive depth."""

from torch import Tensor, nn
from torchvision.models import resnet50


class EarlyExitResNet50(nn.Module):
    """Expose progressively deeper classifiers without channel gating."""

    def __init__(self, num_classes: int, small_images: bool = False) -> None:
        super().__init__()
        backbone = resnet50(weights=None)
        if small_images:
            backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            backbone.maxpool = nn.Identity()
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head2, self.head3 = nn.Linear(512, num_classes), nn.Linear(1024, num_classes)
        self.head4 = nn.Linear(2048, num_classes)

    def _head(self, features: Tensor, head: nn.Linear) -> Tensor:
        return head(self.pool(features).flatten(1))

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        x = self.layer1(self.stem(images))
        x = self.layer2(x); stage2 = self._head(x, self.head2)
        x = self.layer3(x); stage3 = self._head(x, self.head3)
        x = self.layer4(x); stage4 = self._head(x, self.head4)
        return {"stage2": stage2, "stage3": stage3, "stage4": stage4}
