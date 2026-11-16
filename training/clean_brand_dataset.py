"""Build a YOLO-cropped, filtered copy of data/brand_dataset for training.

Why: the imported listing photos include interiors, wheels, tail lights and
other close-ups that show no recognisable car, and full press photos with lots
of background. At inference the brand model only ever sees a YOLO crop, so we
train on the same kind of crop.

For each image:
  * run the same detector the live pipeline uses (car / bus / truck);
  * keep the largest vehicle box if it covers at least --min-area of the frame,
    and save that box (plus a small margin) to data/brand_dataset_clean/<Brand>/;
  * otherwise copy the image to data/brand_dataset_review/rejected_by_yolo/<Brand>/
    so a person can look at what was dropped;
  * skip byte-identical duplicates (they would leak between train and val).

The original data/brand_dataset is never modified. Re-running skips images
that were already processed.

    python training/clean_brand_dataset.py
    python training/clean_brand_dataset.py --min-area 0.2 --brands BYD Chery
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    BRAND_CLEAN_DATASET_DIR,
    BRAND_DATASET_DIR,
    BRAND_REJECTED_DIR,
    IMAGE_EXTENSIONS,
)
from detect import load_detector  # noqa: E402

# COCO: 2 car, 5 bus, 7 truck (minivans and SUVs are sometimes "truck"/"bus").
CLEAN_CLASS_IDS = [2, 5, 7]


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def largest_box(result) -> tuple[tuple[float, float, float, float], float] | None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = None
    best_area = 0.0
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
    return best, best_area


def expand_box(box, width: int, height: int, margin: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * margin
    pad_y = (y2 - y1) * margin
    return (
        int(max(0, x1 - pad_x)),
        int(max(0, y1 - pad_y)),
        int(min(width, x2 + pad_x)),
        int(min(height, y2 + pad_y)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, default=BRAND_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=BRAND_CLEAN_DATASET_DIR)
    parser.add_argument("--rejected", type=Path, default=BRAND_REJECTED_DIR)
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.08,
        help="Minimum fraction of the image the largest vehicle box must cover.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence.")
    parser.add_argument("--min-side", type=int, default=64, help="Minimum box side in pixels.")
    parser.add_argument("--margin", type=float, default=0.04, help="Extra border around the box.")
    parser.add_argument("--brands", nargs="*", help="Only process these brand folders.")
    args = parser.parse_args()

    detector = load_detector()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "clean_report.csv"

    seen_hashes: dict[str, Path] = {}
    rows = []
    totals = {"kept": 0, "rejected": 0, "duplicate": 0, "unreadable": 0}

    brand_folders = sorted(p for p in args.source.iterdir() if p.is_dir())
    if args.brands:
        wanted = set(args.brands)
        brand_folders = [p for p in brand_folders if p.name in wanted]

    for folder in brand_folders:
        images = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        kept = rejected = 0
        for image_path in images:
            digest = file_hash(image_path)
            if digest in seen_hashes:
                totals["duplicate"] += 1
                rows.append([folder.name, image_path.name, "duplicate", "", seen_hashes[digest].name])
                continue
            seen_hashes[digest] = image_path

            out_path = args.output / folder.name / (image_path.stem + ".jpg")
            rejected_path = args.rejected / folder.name / image_path.name
            if out_path.exists():
                kept += 1
                totals["kept"] += 1
                continue
            if rejected_path.exists():
                rejected += 1
                totals["rejected"] += 1
                continue

            try:
                image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
            except Exception as error:  # noqa: BLE001 - report and keep going
                totals["unreadable"] += 1
                rows.append([folder.name, image_path.name, "unreadable", "", str(error)])
                continue

            width, height = image.size
            result = detector.predict(
                source=image, conf=args.conf, classes=CLEAN_CLASS_IDS, verbose=False
            )[0]
            found = largest_box(result)
            fraction = found[1] / float(width * height) if found else 0.0

            box_ok = found is not None and min(
                found[0][2] - found[0][0], found[0][3] - found[0][1]
            ) >= args.min_side
            if box_ok and fraction >= args.min_area:
                crop = image.crop(expand_box(found[0], width, height, args.margin))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(out_path, quality=95)
                kept += 1
                totals["kept"] += 1
                rows.append([folder.name, image_path.name, "kept", f"{fraction:.3f}", ""])
            else:
                rejected_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, rejected_path)
                rejected += 1
                totals["rejected"] += 1
                reason = "no vehicle" if not found else "vehicle too small"
                rows.append([folder.name, image_path.name, "rejected", f"{fraction:.3f}", reason])

        print(f"{folder.name:16} kept {kept:4d}  rejected {rejected:4d}  (of {len(images)})")

    with open(report_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["brand", "file", "status", "box_fraction", "note"])
        writer.writerows(rows)

    print("\nTotals:", totals)
    print("Clean crops:", args.output)
    print("Rejected (please skim these):", args.rejected)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
