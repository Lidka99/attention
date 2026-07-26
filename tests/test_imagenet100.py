import tempfile
import unittest
from pathlib import Path

import torch

from attentionv3.data.imagenet100 import ImageFolderSubset, load_or_create_manifest


class DummyImageFolder:
    classes = ["n0003", "n0001", "n0002"]
    samples = [("a", 0), ("b", 1), ("c", 2), ("d", 1)]

    @staticmethod
    def loader(path):
        return torch.tensor([len(path)])


class ImageNet100Tests(unittest.TestCase):
    def test_subset_remaps_targets_by_manifest_class_order(self):
        subset = ImageFolderSubset(DummyImageFolder(), ["n0001", "n0003"])
        self.assertEqual(len(subset), 3)
        self.assertEqual([subset[index][1] for index in range(len(subset))], [1, 0, 0])

    def test_manifest_is_reused_after_first_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            first = load_or_create_manifest(DummyImageFolder.classes, path, seed=3, num_classes=2)
            second = load_or_create_manifest(DummyImageFolder.classes, path, seed=999, num_classes=2)
        self.assertEqual(first, second)
        self.assertEqual(first["class_names"], sorted(first["class_names"]))


if __name__ == "__main__":
    unittest.main()
