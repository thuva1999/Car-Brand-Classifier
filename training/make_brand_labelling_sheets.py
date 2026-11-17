"""Numbered contact sheets of every car in outputs/live/, for hand-labelling brand.

    python training/make_brand_labelling_sheets.py

Cars 1..N keep the numbers they have in outputs/colour_labelling/cars.csv (same
crops), so brand and colour labels line up. Cars from live photos taken after
the colour labelling are detected now and numbered after them.

Writes to outputs/brand_labelling/:
  sheet_1.jpg, sheet_2.jpg, ...  every car crop, numbered, with the model's two best guesses
  cars.csv                       one row per number; fill the `true_brand` column
  cars/<number>.png              the crop, so models can be re-scored on the same pixels

Write the brand folder name (e.g. Chevrolet, GAC Aion), `unknown` if you cannot
tell, or `other` for a brand the model does not have (Toyota, VW, ...).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from brand_model import load_brand_model  # noqa: E402
from detect import crop_box, detect_cars, load_detector  # noqa: E402

LIVE_DIR = PROJECT_DIR / "outputs" / "live"
COLOUR_DIR = PROJECT_DIR / "outputs" / "colour_labelling"
OUT_DIR = PROJECT_DIR / "outputs" / "brand_labelling"
TILE_W, TILE_H, COLUMNS, PER_SHEET = 480, 400, 4, 12


def font(size: int):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    (OUT_DIR / "cars").mkdir(parents=True, exist_ok=True)
    cars = []  # dicts: number, photo, car_in_photo, box, true_colour, crop
    seen_photos = set()
    colour_csv = COLOUR_DIR / "cars.csv"
    if colour_csv.exists():
        for row in csv.DictReader(open(colour_csv, encoding="utf-8")):
            crop_path = COLOUR_DIR / "cars" / f"{row['number']}.png"
            if not crop_path.exists():
                continue
            seen_photos.add(row["photo"])
            cars.append({
                "number": int(row["number"]), "photo": row["photo"],
                "car_in_photo": row["car_in_photo"], "box": row["box"],
                "true_colour": row["true_colour"],
                "crop": Image.open(crop_path).convert("RGB"),
            })
    detector = load_detector()
    number = max((car["number"] for car in cars), default=0)
    photos = sorted(p for p in LIVE_DIR.glob("*.jpg") if not p.stem.endswith("_pred"))
    for photo in photos:
        if photo.name in seen_photos:
            continue
        image = Image.open(photo).convert("RGB")
        for index, car in enumerate(detect_cars(image, detector), start=1):
            number += 1
            cars.append({
                "number": number, "photo": photo.name, "car_in_photo": index,
                "box": " ".join(map(str, car["box"])), "true_colour": "",
                "crop": crop_box(image, car["box"]),
            })

    model, class_names, _, device = load_brand_model()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cars), 16):
            chunk = cars[start:start + 16]
            probabilities = model.probabilities_for_crops(
                [car["crop"] for car in chunk], device, tta=True
            )
            for car, row in zip(chunk, probabilities):
                top = torch.topk(row, 2)
                car["guess"] = class_names[top.indices[0]]
                car["confidence"] = float(top.values[0])
                car["second"] = class_names[top.indices[1]]

    # Keep brands already typed into an earlier cars.csv.
    labelled = {}
    out_csv = OUT_DIR / "cars.csv"
    if out_csv.exists():
        for row in csv.DictReader(open(out_csv, encoding="utf-8")):
            labelled[(row["photo"], row["box"])] = row.get("true_brand", "")

    with open(out_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["number", "photo", "car_in_photo", "box", "true_colour",
                         "predicted_brand", "brand_confidence", "second_guess", "true_brand"])
        for car in cars:
            car["crop"].save(OUT_DIR / "cars" / f"{car['number']}.png")
            writer.writerow([car["number"], car["photo"], car["car_in_photo"], car["box"],
                             car["true_colour"], car["guess"], f"{car['confidence']:.2f}",
                             car["second"], labelled.get((car["photo"], car["box"]), "")])

    big, small = font(34), font(22)
    for sheet_index in range(0, len(cars), PER_SHEET):
        chunk = cars[sheet_index:sheet_index + PER_SHEET]
        sheet_rows = (len(chunk) + COLUMNS - 1) // COLUMNS
        sheet = Image.new("RGB", (TILE_W * COLUMNS, TILE_H * sheet_rows), "white")
        draw = ImageDraw.Draw(sheet)
        for i, car in enumerate(chunk):
            x, y = (i % COLUMNS) * TILE_W, (i // COLUMNS) * TILE_H
            thumb = car["crop"].copy()
            scale = min((TILE_W - 8) / thumb.width, (TILE_H - 56) / thumb.height)
            thumb = thumb.resize((max(1, int(thumb.width * scale)), max(1, int(thumb.height * scale))))
            sheet.paste(thumb, (x + (TILE_W - thumb.width) // 2, y + 52))
            draw.rectangle([x + 4, y + 4, x + TILE_W - 4, y + 48], fill="#ffe14d")
            draw.text((x + 10, y + 6), f"#{car['number']}", fill="black", font=big)
            guess = f"{car['guess']} {car['confidence']:.0%} / {car['second']}"
            draw.text((x + 105, y + 14), guess, fill="black", font=small)
            draw.rectangle([x + 2, y + 2, x + TILE_W - 2, y + TILE_H - 2], outline="#999999", width=2)
        path = OUT_DIR / f"sheet_{sheet_index // PER_SHEET + 1}.jpg"
        sheet.save(path, quality=90)
        print("Wrote", path)
    print(f"{len(cars)} cars; fill true_brand in {out_csv}")


if __name__ == "__main__":
    main()
