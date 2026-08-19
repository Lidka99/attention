"""Utilities for auditing the attainable benefit of local budget transfers."""

import torch
from torch import Tensor


def local_transfer_candidates(keep: Tensor, min_groups: int, groups: int,
                              transfer_groups: int) -> list[tuple[int, int, Tensor, Tensor]]:
    """Return every directed legal local transfer, leaving invalid rows unchanged."""
    if keep.ndim != 2:
        raise ValueError("keep must be [batch, stages]")
    if transfer_groups <= 0:
        raise ValueError("transfer_groups must be positive")
    candidates = []
    for donor in range(keep.shape[1]):
        for recipient in range(keep.shape[1]):
            if donor == recipient:
                continue
            valid = ((keep[:, donor] - transfer_groups >= min_groups)
                     & (keep[:, recipient] + transfer_groups <= groups))
            override = keep.clone()
            override[valid, donor] -= transfer_groups
            override[valid, recipient] += transfer_groups
            candidates.append((donor, recipient, valid, override))
    return candidates
