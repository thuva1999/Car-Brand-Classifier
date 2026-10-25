
from datasets import load_dataset
from pathlib import Path
from collections import defaultdict
import re

OUTPUT_DIR = Path("data/brand_dataset")
IMAGES_PER_BRAND = 20

TARGET_BRANDS = {
    "BMW",
    "Toyota",
    "Honda",
    "Audi",
    "Mercedes-Benz",
}

def get_brand(model_name):
    # Match the brand using the beginning of the model name.
    for brand in sorted(TARGET_BRANDS, key=len, reverse=True):
        if model_name.startswith(brand + " ") or model_name == brand:
            return brand
    return None

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Stanford Cars dataset...")

dataset = load_dataset(
    "tanganke/stanford_cars",
    split="train",
    streaming=True,
)

class_names = dataset.features["label"].names
saved_counts = defaultdict(int)

print("Collecting images for:", ", ".join(sorted(TARGET_BRANDS)))

for item in dataset:
    label = item["label"]
    model_name = class_names[label]
    brand = get_brand(model_name)

    if brand is None:
        continue

    if saved_counts[brand] >= IMAGES_PER_BRAND:
        if all(saved_counts[b] >= IMAGES_PER_BRAND for b in TARGET_BRANDS):
            break
        continue

    brand_dir = OUTPUT_DIR / brand
    brand_dir.mkdir(parents=True, exist_ok=True)

    safe_model = re.sub(r"[^A-Za-z0-9]+", "_", model_name)
    image_path = brand_dir / f"{safe_model}_{saved_counts[brand]}.jpg"

    item["image"].convert("RGB").save(image_path)
    saved_counts[brand] += 1

    print(f"{brand}: {saved_counts[brand]}/{IMAGES_PER_BRAND}")

print("\nFinished.")
for brand in sorted(TARGET_BRANDS):
    print(f"{brand}: {saved_counts[brand]} images")

print("Dataset location:", OUTPUT_DIR)
