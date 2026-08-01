import tempfile
import unittest
from pathlib import Path

from PIL import Image

from attentionv3.data import TinyImageNetValidation


class TinyImageNetTests(unittest.TestCase):
    def test_validation_reads_annotation_labels_without_moving_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "val" / "images"
            images.mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(images / "sample.JPEG")
            (root / "val" / "val_annotations.txt").write_text("sample.JPEG\tn00000001\t0\t0\t4\t4\n")
            dataset = TinyImageNetValidation(root, {"n00000001": 7})
            image, label = dataset[0]
            self.assertEqual(image.size, (4, 4))
            self.assertEqual(label, 7)


if __name__ == "__main__":
    unittest.main()
