"""Action-space targets for oracle-distilled budget routing."""

import torch
from torch import Tensor
from torch.nn import functional as F


def transfer_actions(stages: int = 4) -> tuple[tuple[int, int], ...]:
    """Ordered directed transfers; action zero is reserved for ``noop``."""
    if stages < 2:
        raise ValueError("at least two stages are required")
    return tuple((donor, recipient) for donor in range(stages) for recipient in range(stages)
                 if donor != recipient)


def legal_action_mask(keep: Tensor, min_groups: int, groups: int,
                      transfer_groups: int) -> Tensor:
    """Return [batch, 1 + stages * (stages - 1)] legality including noop."""
    if keep.ndim != 2:
        raise ValueError("keep must be [batch, stages]")
    if transfer_groups <= 0:
        raise ValueError("transfer_groups must be positive")
    mask = torch.ones((keep.shape[0], 1), dtype=torch.bool, device=keep.device)
    transfers = [((keep[:, donor] - transfer_groups >= min_groups)
                  & (keep[:, recipient] + transfer_groups <= groups))
                 for donor, recipient in transfer_actions(keep.shape[1])]
    return torch.cat((mask, torch.stack(transfers, dim=1)), dim=1)


def oracle_action_labels(losses: Tensor, legal: Tensor) -> Tensor:
    """Choose the legal action with lowest per-example classification loss."""
    if losses.shape != legal.shape or losses.ndim != 2:
        raise ValueError("losses and legal must share [batch, actions] shape")
    if not legal[:, 0].all():
        raise ValueError("noop must be legal for every example")
    return losses.masked_fill(~legal, float("inf")).argmin(dim=1)


def masked_action_loss(logits: Tensor, labels: Tensor, legal: Tensor) -> Tensor:
    """Cross-entropy that prevents the student from selecting illegal actions."""
    if logits.shape != legal.shape or labels.shape != logits.shape[:1]:
        raise ValueError("expected logits/legal [batch, actions] and labels [batch]")
    if not legal[:, 0].all():
        raise ValueError("noop must be legal for every example")
    if not legal.gather(1, labels[:, None]).all():
        raise ValueError("oracle labels must be legal")
    return F.cross_entropy(logits.masked_fill(~legal, torch.finfo(logits.dtype).min), labels)
