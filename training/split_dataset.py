"""Create an 80/20 brand split, grouped so near-duplicate photos stay together.

The imported listing set has several photos of the same model-year car, named
like ``Toyota_Tundra_2017_..._Pickup_GEB.jpg`` / ``..._Pickup_VYX.jpg``. A plain
random split puts some of them in train and some in val, which inflates the
validation score. Here every image is assigned a *group* and whole groups go
to one side of the split:

  * listing photos: the name without its random 3-letter suffix (one car model-year);
  * street crops ``IMG_..._car2``: the source photo ``IMG_...``;
  * anything else: its own group.

Reads data/brand_dataset_clean if it exists (see clean_brand_dataset.py),
otherwise data/brand_dataset. Never deletes an existing split.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    BRAND_CLEAN_DATASET_DIR,
    BRAND_DATASET_DIR,
    EXCLUDED_BRANDS,
    IMAGE_EXTENSIONS,
    SPLIT_BRAND_DIR,
)

VALIDATION_RATIO = 0.20
MINIMUM_IMAGES = 8

LISTING_SUFFIX = re.compile(r"^(?P<group>.+_.+_.+)_[A-Za-z0-9]{3}$")
STREET_CROP = re.compile(r"^(?P<group>.+)_car\d+$")
# Gallery photos saved from one V3 Cars model page show the same car: keep the
# whole page in train or in val ("v3cars_Geely_Galaxy_L7_07" -> Geely_Galaxy_L7).
WEB_GALLERY = re.compile(r"^(?P<group>v3cars_.+)_\d+$")


def group_key(path: Path) -> str:
    stem = path.stem
    for pattern in (WEB_GALLERY, STREET_CROP, LISTING_SUFFIX):
        match = pattern.match(stem)
        if match:
            return match.group("group")
    return stem


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_groups(images: list[Path], rng: random.Random) -> tuple[list[Path], list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        groups[group_key(image)].append(image)
    keys = sorted(groups)
    rng.shuffle(keys)

    target_val = max(1, round(len(images) * VALIDATION_RATIO))
    val: list[Path] = []
    train: list[Path] = []
    for key in keys:
        # Fill validation with whole groups until it reaches ~20%.
        if len(val) < target_val and len(val) + len(groups[key]) <= target_val * 1.5:
            val.extend(groups[key])
        else:
            train.extend(groups[key])
    if not train:  # one giant group: fall back to a plain split
        train, val = images[target_val:], images[:target_val]
    return train, val


def main() -> None:
    default_source = (
        BRAND_CLEAN_DATASET_DIR if BRAND_CLEAN_DATASET_DIR.exists() else BRAND_DATASET_DIR
    )
    parser = argparse.ArgumentParser(description="Grouped 80/20 brand split.")
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--output", type=Path, default=SPLIT_BRAND_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Missing {args.source}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing split directory: {args.output}")
    print("Source:", args.source)

    brands: dict[str, list[Path]] = {}
    skipped = []
    excluded = []
    for folder in sorted(args.source.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name in EXCLUDED_BRANDS:
            excluded.append(folder.name)
            continue
        images = list_images(folder)
        if len(images) < MINIMUM_IMAGES:
            skipped.append((folder.name, len(images)))
            continue
        brands[folder.name] = images

    if excluded:
        print("Excluded brands (config.EXCLUDED_BRANDS):", ", ".join(excluded))
    if skipped:
        print("Skipped (fewer than", MINIMUM_IMAGES, "images):")
        for name, count in skipped:
            print(f"  {name}: {count}")
    if len(brands) < 2:
        raise SystemExit("Need at least two brands with enough images.")

    rng = random.Random(args.seed)
    for brand, images in brands.items():
        train, val = split_groups(images, rng)
        for split_name, group in (("train", train), ("val", val)):
            destination = args.output / split_name / brand
            destination.mkdir(parents=True, exist_ok=True)
            for image in group:
                shutil.copy2(image, destination / image.name)
        print(f"{brand:16} {len(train):4d} train  {len(val):4d} val")

    print(f"\nSplit written to {args.output}")


if __name__ == "__main__":
    main()
