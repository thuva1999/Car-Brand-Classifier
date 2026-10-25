"""Run the car detector on street photos and save crops for labelling.

Crops land in data/crops/unlabelled/. Rename/copy them into:
  data/brand_dataset/<Brand>/
  data/color_dataset/<colour>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import IMAGE_EXTENSIONS, STREET_PHOTO_DIRS  # noqa: E402
from detect import crop_box, detect_cars, load_detector  # noqa: E402


def collect_images(extra_folders: list[Path]) -> list[Path]:
    folders = extra_folders if extra_folders else STREET_PHOTO_DIRS
    images: list[Path] = []
    for folder in folders:
        if not folder.exists():
            continue
        images.extend(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return sorted(images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop cars from street photos.")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Folder of street photos (can be repeated). Defaults to Tashkent folders.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_DIR / "data" / "crops" / "unlabelled"),
        help="Folder for cropped cars.",
    )
    args = parser.parse_args()

    input_folders = [Path(path) for path in args.input]
    images = collect_images(input_folders)
    if not images:
        raise SystemExit("No street photos found. Put phone photos in data/tashkent_test/ or pass --input.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO detector...")
    detector = load_detector()
    saved = 0
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        cars = detect_cars(image, detector)
        for index, car in enumerate(cars):
            crop = crop_box(image, car["box"])
            name = f"{image_path.stem}_car{index + 1}.jpg"
            crop.save(output_dir / name, quality=95)
            saved += 1
        print(f"{image_path.name}: {len(cars)} cars")

    print(f"\nSaved {saved} crops to {output_dir}")
    print("Copy each crop into data/brand_dataset/<Brand>/ and data/color_dataset/<colour>/")


if __name__ == "__main__":
    main()
