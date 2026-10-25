"""Brand validation accuracy, confusion matrix, and optional phone-photo dry run."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from brand_model import load_brand_model  # noqa: E402
from config import IMAGE_SIZE, MODELS_DIR, SPLIT_BRAND_DIR  # noqa: E402
from pipeline import CarPipeline, format_answer_line  # noqa: E402


def evaluate_brand() -> None:
    model, class_names, image_transform, device = load_brand_model()
    val_dataset = datasets.ImageFolder(
        SPLIT_BRAND_DIR / "val",
        transform=transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
    if val_dataset.classes != class_names:
        print("Warning: folder names and checkpoint class names differ.")
        print(" folders:", val_dataset.classes)
        print(" model:  ", class_names)

    loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    number = len(class_names)
    matrix = [[0 for _ in range(number)] for _ in range(number)]
    correct = 0
    total = 0
    per_class_correct = [0] * number
    per_class_total = [0] * number

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(1)
            for true_index, pred_index in zip(labels.tolist(), predicted.tolist()):
                matrix[true_index][pred_index] += 1
                per_class_total[true_index] += 1
                if true_index == pred_index:
                    per_class_correct[true_index] += 1
                    correct += 1
                total += 1

    accuracy = correct / max(total, 1)
    print(f"Brand validation accuracy: {accuracy:.1%} ({correct}/{total})")
    print("\nPer-class accuracy:")
    for index, name in enumerate(class_names):
        if per_class_total[index] == 0:
            continue
        print(f"  {name:16} {per_class_correct[index] / per_class_total[index]:.1%}  n={per_class_total[index]}")

    print("\nConfusion matrix rows=true, cols=predicted:")
    header = f"{'':16}" + "".join(f"{name[:7]:>8}" for name in class_names)
    print(header)
    for index, name in enumerate(class_names):
        row = "".join(f"{value:8d}" for value in matrix[index])
        print(f"{name[:16]:16}{row}")

    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "brand_confusion_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred"] + class_names)
        for index, name in enumerate(class_names):
            writer.writerow([name] + matrix[index])
    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "class_names": class_names,
        "per_class_accuracy": {
            name: (per_class_correct[i] / per_class_total[i] if per_class_total[i] else None)
            for i, name in enumerate(class_names)
        },
    }
    with open(output_dir / "brand_eval.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"\nWrote {csv_path}")


def evaluate_street_photos(folder: Path) -> None:
    from config import IMAGE_EXTENSIONS

    images = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print(f"No images in {folder}")
        return

    print(f"\nLoading full pipeline for {len(images)} street photos in {folder}...")
    pipeline = CarPipeline()
    for image_path in images:
        predictions = pipeline.predict_path(image_path)
        print(f"{image_path.name}: {format_answer_line(predictions)}")


def main() -> None:
    evaluate_brand()
    street = PROJECT_DIR / "data" / "tashkent_test"
    if street.exists():
        evaluate_street_photos(street)


if __name__ == "__main__":
    main()
