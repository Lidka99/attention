"""Deterministic ImageNet-100 subset creation and data loaders."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class ImageFolderSubset(Dataset[tuple[Any, int]]):
    """Filter ImageFolder by class name and remap labels to ``[0, n_classes)``.

    Keeping class names—not original integer labels—in the manifest makes the
    split robust to the ordering of folders on a particular machine.
    """

    def __init__(self, source: Any, class_names: Sequence[str], transform: Any = None) -> None:
        available = set(source.classes)
        missing = sorted(set(class_names) - available)
        if missing:
            raise ValueError(f"source is missing manifest classes: {missing[:3]}")
        self.source = source
        self.classes = list(class_names)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.transform = transform
        self.samples = [
            (path, self.class_to_idx[source.classes[original_label]])
            for path, original_label in source.samples
            if source.classes[original_label] in self.class_to_idx
        ]
        if not self.samples:
            raise ValueError("ImageNet-100 subset contains no samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path, target = self.samples[index]
        image = self.source.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def load_or_create_manifest(train_classes: Sequence[str], manifest_path: str | Path,
                            seed: int, num_classes: int = 100) -> dict[str, Any]:
    """Load a checked-in split manifest or create it once from train classes."""
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        class_names = manifest.get("class_names", [])
        if len(class_names) != num_classes:
            raise ValueError("manifest class count does not match num_classes")
        if not set(class_names).issubset(set(train_classes)):
            raise ValueError("manifest contains classes absent from training data")
        return manifest
    if len(train_classes) < num_classes:
        raise ValueError("not enough classes to create requested subset")

    # Sampling after lexical sorting makes the selection independent of the
    # underlying filesystem enumeration. The selected class list is sorted for
    # stable remapped labels and human-readable diffs.
    population = sorted(train_classes)
    selected = sorted(random.Random(seed).sample(population, num_classes))
    manifest = {"dataset": "ImageNet-100", "seed": seed, "class_names": selected}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def imagenet_transforms(image_size: int = 224):
    """Return standard ImageNet train and validation transformations."""
    train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    validation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return train, validation


def build_imagenet100_loaders(train_dir: str | Path, val_dir: str | Path,
                               manifest_path: str | Path, seed: int, batch_size: int,
                               workers: int = 8, image_size: int = 224,
                               num_classes: int = 100) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Build ImageNet-100 loaders from class-organised ImageFolder directories."""
    train_source = datasets.ImageFolder(str(train_dir))
    val_source = datasets.ImageFolder(str(val_dir))
    manifest = load_or_create_manifest(train_source.classes, manifest_path, seed, num_classes)
    train_transform, val_transform = imagenet_transforms(image_size)
    train_data = ImageFolderSubset(train_source, manifest["class_names"], train_transform)
    val_data = ImageFolderSubset(val_source, manifest["class_names"], val_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers,
                              pin_memory=True, persistent_workers=workers > 0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=True, persistent_workers=workers > 0)
    return train_loader, val_loader, manifest


def stratified_calibration_split(dataset: Dataset, fraction: float, seed: int):
    """Return disjoint calibration/test Subsets with every class represented.

    The dataset must expose remapped ``samples`` like :class:`ImageFolderSubset`.
    Indices are deterministic for a seed and are saved by the calling script.
    """
    from collections import defaultdict
    from torch.utils.data import Subset

    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    if not hasattr(dataset, "samples"):
        raise TypeError("dataset must expose samples for stratified splitting")
    by_class = defaultdict(list)
    for index, (_, target) in enumerate(dataset.samples):
        by_class[target].append(index)
    generator = random.Random(seed)
    calibration, test = [], []
    for target in sorted(by_class):
        indices = list(by_class[target])
        generator.shuffle(indices)
        count = max(1, round(len(indices) * fraction))
        if count >= len(indices):
            raise ValueError("each class needs at least two validation samples")
        calibration.extend(indices[:count])
        test.extend(indices[count:])
    return Subset(dataset, sorted(calibration)), Subset(dataset, sorted(test))
