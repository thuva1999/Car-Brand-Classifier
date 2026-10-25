
from pathlib import Path
import random
import shutil

random.seed(42)

SOURCE_DIR = Path("data/brand_dataset")
OUTPUT_DIR = Path("data/split_dataset")
VALIDATION_RATIO = 0.20

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {SOURCE_DIR}")

for brand_dir in SOURCE_DIR.iterdir():
    if not brand_dir.is_dir():
        continue

    images = sorted(
        p for p in brand_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if len(images) < 2:
        print(f"Skipping {brand_dir.name}: not enough images")
        continue

    random.shuffle(images)
    val_count = max(1, round(len(images) * VALIDATION_RATIO))

    splits = {
        "train": images[val_count:],
        "val": images[:val_count],
    }

    for split_name, split_images in splits.items():
        destination = OUTPUT_DIR / split_name / brand_dir.name
        destination.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            shutil.copy2(image_path, destination / image_path.name)

    print(
        f"{brand_dir.name}: "
        f"{len(splits['train'])} train, "
        f"{len(splits['val'])} validation"
    )

print(f"\nDataset split complete: {OUTPUT_DIR}")
