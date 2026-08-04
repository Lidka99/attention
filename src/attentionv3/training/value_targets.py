"""Counterfactual supervision for stage-wise value-of-compute policies."""

import torch
from torch import Tensor
from torch.nn import functional as F


def counterfactual_stage_value_targets(base_loss: Tensor, ablated_loss: Tensor,
                                       temperature: float = 1.0) -> Tensor:
    """Return a distribution over stages from per-sample ablation regret.

    ``ablated_loss[:, stage]`` is the classification loss after constraining a
    single stage. A larger increase relative to ``base_loss`` means that one
    more unit of compute is more valuable in that stage for that image.
    """
    if base_loss.ndim != 1 or ablated_loss.ndim != 2 or base_loss.shape[0] != ablated_loss.shape[0]:
        raise ValueError("base_loss must be [batch] and ablated_loss [batch, stages]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    regret = (ablated_loss - base_loss[:, None]).clamp_min(0)
    return F.softmax(regret / temperature, dim=1)


def stage_value_loss(value_logits: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy from policy stage values to counterfactual targets."""
    if value_logits.shape != targets.shape or value_logits.ndim != 2:
        raise ValueError("value_logits and targets must both be [batch, stages]")
    if not torch.allclose(targets.sum(dim=1), torch.ones_like(targets[:, 0]), atol=1e-5):
        raise ValueError("each target row must sum to one")
    return -(targets.detach() * F.log_softmax(value_logits, dim=1)).sum(dim=1).mean()
