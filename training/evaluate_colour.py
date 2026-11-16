"""Score the HSV colour rules on the hand-labelled live cars.

    python training/make_colour_labelling_sheets.py   # once, then fill true_colour
    python training/evaluate_colour.py

Re-runs the current rules on exactly the saved crop + YOLO mask of every car
in outputs/colour_labelling/cars.csv that has a `true_colour`, and prints the
accuracy, a confusion table and every miss. `predicted_colour` in the CSV is
what the rules said when the sheets were made, kept for comparison.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from color_model import hsv_colour_with_confidence  # noqa: E402

LABEL_DIR = PROJECT_DIR / "outputs" / "colour_labelling"


def main() -> None:
    rows = [
        row for row in csv.DictReader(open(LABEL_DIR / "cars.csv", encoding="utf-8"))
        if row["true_colour"]
    ]
    if not rows:
        raise SystemExit("No labelled cars: fill the true_colour column of cars.csv first.")

    now, before = [], []
    for row in rows:
        crop = Image.open(LABEL_DIR / "cars" / f"{row['number']}.png").convert("RGB")
        mask_path = LABEL_DIR / "cars" / f"{row['number']}.npy"
        mask = np.load(mask_path) if mask_path.exists() else None
        warm = float(row["scene_warm"]) if row["scene_warm"] else None
        now.append(hsv_colour_with_confidence(crop, mask, warm)[0])
        before.append(row["predicted_colour"])

    truth = [row["true_colour"] for row in rows]
    total = len(rows)
    for name, predicted in (("when sheets were made", before), ("current rules", now)):
        correct = sum(p == t for p, t in zip(predicted, truth))
        print(f"{name:22}: {correct}/{total} ({correct / total:.0%})")

    colours = sorted(set(truth) | set(now))
    print("\nPer colour (current rules): correct / labelled")
    for colour in colours:
        items = [p for p, t in zip(now, truth) if t == colour]
        if items:
            print(f"  {colour:7} {sum(p == colour for p in items)}/{len(items)}")

    print("\nConfusion, rows = true, columns = predicted:")
    print(" " * 8 + "".join(f"{c[:6]:>7}" for c in colours))
    pairs = Counter(zip(truth, now))
    for t in colours:
        if any(tt == t for tt in truth):
            print(f"{t:8}" + "".join(f"{pairs[(t, p)] or '.':>7}" for p in colours))

    changed = [(row["number"], b, n, t) for row, b, n, t in zip(rows, before, now, truth) if b != n]
    print(f"\nChanged since the sheets were made ({len(changed)}):")
    for number, b, n, t in changed:
        mark = "fixed" if n == t else ("BROKEN" if b == t else "still wrong")
        print(f"  #{number}: {b} -> {n} (true {t}) {mark}")
    misses = [(row["number"], n, t) for row, n, t in zip(rows, now, truth) if n != t]
    print(f"\nStill wrong ({len(misses)}): " + ", ".join(f"#{a} {n}->{t}" for a, n, t in misses))


if __name__ == "__main__":
    main()
