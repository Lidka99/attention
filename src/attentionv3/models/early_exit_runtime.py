"""Runtime helpers that execute only the required early-exit prefix."""

from torch import Tensor

from .early_exit_resnet import EarlyExitResNet50


def forward_to_exit(model: EarlyExitResNet50, images: Tensor, exit_stage: int) -> Tensor:
    """Return one exit's logits without evaluating later ResNet stages."""
    if exit_stage not in (2, 3, 4):
        raise ValueError("exit_stage must be 2, 3 or 4")
    x = model.layer2(model.layer1(model.stem(images)))
    if exit_stage == 2:
        return model._head(x, model.head2)
    x = model.layer3(x)
    if exit_stage == 3:
        return model._head(x, model.head3)
    return model._head(model.layer4(x), model.head4)
