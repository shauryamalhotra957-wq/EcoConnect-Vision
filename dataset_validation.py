"""Preflight validation for the EcoConnect image dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


class DatasetValidationError(ValueError):
    """Raised when the training dataset cannot be consumed safely."""


@dataclass(frozen=True)
class DatasetSummary:
    class_names: tuple[str, ...]
    image_counts: dict[str, int]


def validate_image_dataset(root: str | Path) -> DatasetSummary:
    """Validate binary train/validation folders and every supported image."""

    root = Path(root)
    split_classes: dict[str, tuple[str, ...]] = {}
    image_counts: dict[str, int] = {}
    invalid_images: list[Path] = []

    for split in ("train", "val"):
        split_dir = root / split
        if not split_dir.is_dir():
            raise DatasetValidationError(f"Missing dataset split directory: {split_dir}")

        class_names = tuple(sorted(path.name for path in split_dir.iterdir() if path.is_dir()))
        if len(class_names) != 2:
            raise DatasetValidationError(
                f"Expected exactly two class directories in {split_dir}, found {len(class_names)}"
            )
        split_classes[split] = class_names

        for class_name in class_names:
            class_dir = split_dir / class_name
            images = sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not images:
                raise DatasetValidationError(f"No supported images found in {class_dir}")
            image_counts[f"{split}/{class_name}"] = len(images)

            for image_path in images:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except (OSError, SyntaxError):
                    invalid_images.append(image_path)

    if split_classes["train"] != split_classes["val"]:
        raise DatasetValidationError(
            "Training and validation class directories differ: "
            f"train={split_classes['train']}, val={split_classes['val']}"
        )
    if invalid_images:
        shown = ", ".join(str(path.relative_to(root)) for path in invalid_images[:10])
        remainder = len(invalid_images) - 10
        suffix = f" (and {remainder} more)" if remainder > 0 else ""
        raise DatasetValidationError(f"Unreadable image files: {shown}{suffix}")

    return DatasetSummary(
        class_names=split_classes["train"],
        image_counts=image_counts,
    )
