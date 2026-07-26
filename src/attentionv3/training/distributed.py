"""Minimal DistributedDataParallel support for multi-GPU training."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def initialize_distributed(requested_device: str = "auto",
                           required_world_size: int | None = None) -> DistributedContext:
    """Initialise DDP from torchrun environment variables when requested."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if required_world_size is not None and world_size != required_world_size:
        raise RuntimeError(
            f"this experiment requires {required_world_size} processes; run with "
            f"torchrun --nproc_per_node={required_world_size}"
        )
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA GPUs")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return DistributedContext(rank, world_size, local_rank, torch.device(f"cuda:{local_rank}"))
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device)
    return DistributedContext(rank, world_size, local_rank, device)


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        dist.barrier()


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()
