"""Numbered contact sheets of every car in outputs/live/, for hand-labelling colour.

    python training/make_colour_labelling_sheets.py

Writes to outputs/colour_labelling/:
  sheet_1.jpg, sheet_2.jpg, ...  every car crop, numbered, with the current colour guess
  cars.csv                       one row per number: photo, box, current colour, HSV stats
  cars/<number>.png / .npy       the crop and its YOLO mask, so colour rules can be
                                 re-run on exactly the same pixels after labelling

Fill the `true_colour` column of cars.csv (or list the wrong numbers) to get a
labelled test set taken the same way as the live test: phone photos of a screen.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from color_model import _paint_pixels, hsv_colour_with_confidence, scene_warmth  # noqa: E402
from detect import crop_box, detect_cars, load_detector  # noqa: E402

LIVE_DIR = PROJECT_DIR / "outputs" / "live"
OUT_DIR = PROJECT_DIR / "outputs" / "colour_labelling"
TILE_W, TILE_H, COLUMNS, PER_SHEET = 380, 280, 5, 20


def font(size: int):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def scene_brightness(image: Image.Image, boxes) -> float:
    """Median brightness (HSV V) of everything outside the car boxes."""
    rgb = np.array(image.convert("RGB"))
    background = np.ones(rgb.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in boxes:
        background[y1:y2, x1:x2] = False
    if background.sum() < background.size * 0.05:
        background[:] = True
    value = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 2]
    return float(np.median(value[background]))


def main() -> None:
    (OUT_DIR / "cars").mkdir(parents=True, exist_ok=True)
    detector = load_detector()
    photos = sorted(p for p in LIVE_DIR.glob("*.jpg") if not p.stem.endswith("_pred"))
    rows, tiles = [], []
    for photo in photos:
        image = Image.open(photo).convert("RGB")
        cars = detect_cars(image, detector)
        boxes = [car["box"] for car in cars]
        warm = scene_warmth(image, boxes)
        bright = scene_brightness(image, boxes)
        for index, car in enumerate(cars, start=1):
            number = len(rows) + 1
            crop = crop_box(image, car["box"])
            mask = car.get("mask")
            colour, confidence = hsv_colour_with_confidence(crop, mask, warm)
            bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
            pixels = _paint_pixels(bgr, mask).astype(np.int32)
            value, saturation = pixels[:, 2], pixels[:, 1]
            lit = value >= 60
            crop.save(OUT_DIR / "cars" / f"{number}.png")
            if mask is not None:
                np.save(OUT_DIR / "cars" / f"{number}.npy", mask)
            rows.append({
                "number": number,
                "photo": photo.name,
                "car_in_photo": index,
                "box": " ".join(map(str, car["box"])),
                "predicted_colour": colour,
                "colour_confidence": f"{confidence:.2f}",
                "true_colour": "",
                "median_v": f"{np.median(value):.0f}",
                "median_s_lit": f"{np.median(saturation[lit]) if lit.any() else 0:.0f}",
                "scene_v": f"{bright:.0f}",
                "scene_warm": "" if warm is None else f"{warm:.3f}",
            })
            tiles.append((number, colour, crop))
        print(f"{photo.name}: {len(cars)} car(s)")

    with open(OUT_DIR / "cars.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    big, small = font(30), font(24)
    for sheet_index in range(0, len(tiles), PER_SHEET):
        chunk = tiles[sheet_index:sheet_index + PER_SHEET]
        sheet_rows = (len(chunk) + COLUMNS - 1) // COLUMNS
        sheet = Image.new("RGB", (TILE_W * COLUMNS, TILE_H * sheet_rows), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (number, colour, crop) in enumerate(chunk):
            x, y = (i % COLUMNS) * TILE_W, (i // COLUMNS) * TILE_H
            thumb = crop.copy()
            thumb.thumbnail((TILE_W - 8, TILE_H - 50))
            sheet.paste(thumb, (x + (TILE_W - thumb.width) // 2, y + 46))
            draw.rectangle([x + 4, y + 4, x + TILE_W - 4, y + 42], fill="#ffe14d")
            draw.text((x + 10, y + 7), f"#{number}", fill="black", font=big)
            draw.text((x + 110, y + 11), f"now: {colour}", fill="black", font=small)
            draw.rectangle([x + 2, y + 2, x + TILE_W - 2, y + TILE_H - 2], outline="#999999", width=2)
        path = OUT_DIR / f"sheet_{sheet_index // PER_SHEET + 1}.jpg"
        sheet.save(path, quality=88)
        print("Wrote", path)
    print(f"{len(rows)} cars; details in {OUT_DIR / 'cars.csv'}")


if __name__ == "__main__":
    main()
