"""Tiny ImageNet data access without duplicating validation images."""

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


class TinyImageNetValidation(Dataset):
    """Read validation labels from Tiny ImageNet's annotation file."""

    def __init__(self, root: str | Path, class_to_idx: dict[str, int], transform=None) -> None:
        root = Path(root)
        annotation = root / "val" / "val_annotations.txt"
        labels = {}
        for line in annotation.read_text().splitlines():
            image, wnid, *_ = line.split("\t")
            labels[image] = class_to_idx[wnid]
        self.samples = [(root / "val" / "images" / image, label) for image, label in sorted(labels.items())]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def tinyimagenet_transforms():
    train = transforms.Compose([transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),
                                                                            (0.229, 0.224, 0.225))])
    validation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),
                                                                                   (0.229, 0.224, 0.225))])
    return train, validation


def build_tinyimagenet_loaders(root: str | Path, batch_size: int, workers: int = 4,
                                distributed: bool = False, rank: int = 0, world_size: int = 1):
    root = Path(root)
    train_transform, validation_transform = tinyimagenet_transforms()
    train = datasets.ImageFolder(root / "train", transform=train_transform)
    validation = TinyImageNetValidation(root, train.class_to_idx, validation_transform)
    train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    validation_sampler = DistributedSampler(validation, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
    options = {"batch_size": batch_size, "num_workers": workers, "pin_memory": True,
               "persistent_workers": workers > 0}
    return (DataLoader(train, shuffle=train_sampler is None, sampler=train_sampler, **options),
            DataLoader(validation, shuffle=False, sampler=validation_sampler, **options))
