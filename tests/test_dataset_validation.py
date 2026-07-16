from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from PIL import Image

from dataset_validation import DatasetValidationError, validate_image_dataset


CLASSES = ("biodegradable", "non_biodegradable")


def create_dataset(root: Path) -> None:
    for split in ("train", "val"):
        for class_name in CLASSES:
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            Image.new("RGB", (4, 4), color="green").save(class_dir / "sample.png")


class DatasetValidationTests(unittest.TestCase):
    def test_accepts_matching_binary_splits_with_readable_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_dataset(root)

            summary = validate_image_dataset(root)

            self.assertEqual(summary.class_names, CLASSES)
            self.assertEqual(summary.image_counts["train/biodegradable"], 1)

    def test_rejects_corrupt_images_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_dataset(root)
            corrupt = root / "train" / "biodegradable" / "broken.jpg"
            corrupt.write_bytes(b"not an image")

            with self.assertRaisesRegex(DatasetValidationError, "broken.jpg"):
                validate_image_dataset(root)

    def test_rejects_mismatched_class_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_dataset(root)
            (root / "val" / "non_biodegradable").rename(root / "val" / "recyclable")

            with self.assertRaisesRegex(DatasetValidationError, "class directories differ"):
                validate_image_dataset(root)


if __name__ == "__main__":
    unittest.main()
