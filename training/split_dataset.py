"""Create an 80/20 brand split from data/brand_dataset. Skips empty folders."""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    BRAND_DATASET_DIR,
    IMAGE_EXTENSIONS,
    SPLIT_BRAND_DIR,
)

random.seed(42)
VALIDATION_RATIO = 0.20
MINIMUM_IMAGES = 8


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    if not BRAND_DATASET_DIR.exists():
        raise FileNotFoundError(f"Missing {BRAND_DATASET_DIR}")

    brands: dict[str, list[Path]] = {}
    skipped = []
    for folder in sorted(BRAND_DATASET_DIR.iterdir()):
        if not folder.is_dir():
            continue
        images = list_images(folder)
        if len(images) < MINIMUM_IMAGES:
            skipped.append((folder.name, len(images)))
            continue
        brands[folder.name] = images

    print("Brands included:")
    for name, images in brands.items():
        print(f"  {name}: {len(images)}")
    if skipped:
        print("\nSkipped (not enough images yet):")
        for name, count in skipped:
            print(f"  {name}: {count}")

    if len(brands) < 2:
        raise SystemExit("Need at least two brands with enough images.")

    if SPLIT_BRAND_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing split directory: {SPLIT_BRAND_DIR}"
        )

    for brand, images in brands.items():
        random.shuffle(images)
        val_count = max(1, round(len(images) * VALIDATION_RATIO))
        splits = {
            "train": images[val_count:],
            "val": images[:val_count],
        }
        for split_name, group in splits.items():
            destination = SPLIT_BRAND_DIR / split_name / brand
            destination.mkdir(parents=True, exist_ok=True)
            for image in group:
                shutil.copy2(image, destination / image.name)
        print(f"{brand}: {len(splits['train'])} train, {len(splits['val'])} val")

    print(f"\nSplit written to {SPLIT_BRAND_DIR}")


if __name__ == "__main__":
    main()
