"""CIFAR-100 loaders used for inexpensive UCLA pipeline validation."""

from pathlib import Path

from torch.utils.data import DataLoader
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
