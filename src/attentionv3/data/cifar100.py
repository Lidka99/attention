"""CIFAR-100 loaders used for inexpensive UCLA pipeline validation."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


def cifar100_transforms():
    train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                            (0.2675, 0.2565, 0.2761))])
    test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                           (0.2675, 0.2565, 0.2761))])
    return train, test


def build_cifar100_loaders(data_dir: str | Path, batch_size: int, workers: int = 4,
                           distributed: bool = False, rank: int = 0, world_size: int = 1):
    """Create CIFAR-100 loaders; data must be downloaded before DDP starts."""
    train_transform, test_transform = cifar100_transforms()
    train = datasets.CIFAR100(str(data_dir), train=True, download=False, transform=train_transform)
    test = datasets.CIFAR100(str(data_dir), train=False, download=False, transform=test_transform)
    train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    test_sampler = DistributedSampler(test, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
    options = {"batch_size": batch_size, "num_workers": workers, "pin_memory": True,
               "persistent_workers": workers > 0}
    return (DataLoader(train, shuffle=train_sampler is None, sampler=train_sampler, **options),
            DataLoader(test, shuffle=False, sampler=test_sampler, **options))


def cifar100_train_validation_indices(size: int = 50_000, validation_size: int = 5_000,
                                      seed: int = 42) -> tuple[list[int], list[int]]:
    """Return a deterministic, disjoint train/validation split of CIFAR-100 train."""
    if not 0 < validation_size < size:
        raise ValueError("validation_size must be between zero and dataset size")
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[validation_size:], order[:validation_size]


def build_cifar100_train_validation_test_loaders(data_dir: str | Path, batch_size: int,
                                                  workers: int = 4, validation_size: int = 5_000,
                                                  split_seed: int = 42, distributed: bool = False,
                                                  rank: int = 0, world_size: int = 1):
    """Create deterministic train/validation/test loaders without monitoring test each epoch."""
    train_transform, test_transform = cifar100_transforms()
    train_base = datasets.CIFAR100(str(data_dir), train=True, download=False, transform=train_transform)
    validation_base = datasets.CIFAR100(str(data_dir), train=True, download=False, transform=test_transform)
    test = datasets.CIFAR100(str(data_dir), train=False, download=False, transform=test_transform)
    train_indices, validation_indices = cifar100_train_validation_indices(len(train_base), validation_size, split_seed)
    train, validation = Subset(train_base, train_indices), Subset(validation_base, validation_indices)
    sampler = lambda dataset, shuffle: DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle) if distributed else None
    train_sampler, validation_sampler, test_sampler = sampler(train, True), sampler(validation, False), sampler(test, False)
    options = {"batch_size": batch_size, "num_workers": workers, "pin_memory": True,
               "persistent_workers": workers > 0}
    return (DataLoader(train, shuffle=train_sampler is None, sampler=train_sampler, **options),
            DataLoader(validation, shuffle=False, sampler=validation_sampler, **options),
            DataLoader(test, shuffle=False, sampler=test_sampler, **options))
