"""Brand validation accuracy, confusion matrix, and optional phone-photo dry run."""

from __future__ import annotations

import csv
import json
import argparse
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from brand_model import brand_transform, load_brand_model  # noqa: E402
from config import (  # noqa: E402
    BRAND_CANDIDATE_MODEL_PATH,
    BRAND_MODEL_PATH,
    SPLIT_BRAND_DIR,
)
from pipeline import CarPipeline, format_answer_line  # noqa: E402


def normalize_label(label: str) -> str:
    return re.sub(r"[\s-]+", "", label).casefold()


def evaluate_brand(
    model_path: Path = BRAND_MODEL_PATH,
    validation_dir: Path = SPLIT_BRAND_DIR / "val",
) -> None:
    model, class_names, image_transform, device = load_brand_model(model_path=model_path)
    val_dataset = datasets.ImageFolder(
        validation_dir,
        transform=brand_transform(),
    )
    normalized_model_names = [normalize_label(name) for name in class_names]
    if len(set(normalized_model_names)) != len(normalized_model_names):
        raise ValueError(f"Checkpoint contains duplicate normalized class names: {class_names}")
    model_index_by_label = {
        label: index for index, label in enumerate(normalized_model_names)
    }
    if [normalize_label(name) for name in val_dataset.classes] != normalized_model_names:
        print("Warning: validation folder names/order differ from checkpoint class names.")
        print(" folders:", val_dataset.classes)
        print(" model:  ", class_names)

    loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    number = len(class_names)
    matrix = [[0 for _ in range(number)] for _ in range(number)]
    correct = 0
    total = 0
    unsupported = 0
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
                true_label = normalize_label(val_dataset.classes[true_index])
                model_true_index = model_index_by_label.get(true_label)
                if model_true_index is None:
                    unsupported += 1
                    continue
                matrix[model_true_index][pred_index] += 1
                per_class_total[model_true_index] += 1
                if model_true_index == pred_index:
                    per_class_correct[model_true_index] += 1
                    correct += 1
                total += 1

    accuracy = correct / max(total, 1)
    print(f"Brand validation accuracy: {accuracy:.1%} ({correct}/{total})")
    if unsupported:
        print(f"Validation images with labels absent from checkpoint: {unsupported}")
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
    model_suffix = Path(model_path).stem
    csv_path = output_dir / f"brand_confusion_matrix_{model_suffix}.csv"
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
    with open(output_dir / f"brand_eval_{model_suffix}.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"\nWrote {csv_path}")


def evaluate_street_photos(folder: Path, model_path: Path) -> None:
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
    pipeline = CarPipeline(brand_model_path=model_path)
    for image_path in images:
        predictions = pipeline.predict_path(image_path)
        print(f"{image_path.name}: {format_answer_line(predictions)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a car-brand checkpoint.")
    parser.add_argument(
        "--model",
        type=Path,
        default=BRAND_MODEL_PATH,
        help=(
            "Brand checkpoint to evaluate. The candidate checkpoint is available at "
            f"{BRAND_CANDIDATE_MODEL_PATH} after training."
        ),
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=SPLIT_BRAND_DIR / "val",
        help="Labeled validation folder to evaluate against.",
    )
    args = parser.parse_args()
    evaluate_brand(args.model, args.validation_dir)
    street = PROJECT_DIR / "data" / "tashkent_test"
    if street.exists():
        evaluate_street_photos(street, args.model)


if __name__ == "__main__":
    main()
