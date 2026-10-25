"""Train an 8-class colour model on labelled car crops.

Put crops in data/color_dataset/<colour>/*.jpg using the exam names:
white, black, grey, red, blue, green, brown, yellow.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    COLOR_CANDIDATE_MODEL_PATH,
    COLOR_DATASET_DIR,
    COLOUR_NAMES,
    SPLIT_COLOR_DIR,
)
from color_model import color_transform  # noqa: E402

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 8e-4
VALIDATION_RATIO = 0.2


def split_color_dataset() -> None:
    if not COLOR_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Missing {COLOR_DATASET_DIR}. Create folders named {', '.join(COLOUR_NAMES)}."
        )

    if SPLIT_COLOR_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing colour split directory: {SPLIT_COLOR_DIR}"
        )

    random.seed(42)
    from config import IMAGE_EXTENSIONS

    found = False
    for colour in COLOUR_NAMES:
        folder = COLOR_DATASET_DIR / colour
        if not folder.exists():
            continue
        images = [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(images) < 4:
            print(f"Skipping {colour}: {len(images)} images")
            continue
        found = True
        random.shuffle(images)
        val_count = max(1, round(len(images) * VALIDATION_RATIO))
        for split_name, group in (("train", images[val_count:]), ("val", images[:val_count])):
            destination = SPLIT_COLOR_DIR / split_name / colour
            destination.mkdir(parents=True, exist_ok=True)
            for image in group:
                shutil.copy2(image, destination / image.name)
        print(f"{colour}: {len(images) - val_count} train, {val_count} val")

    if not found:
        raise SystemExit("No colour folders had enough labelled crops.")


def main() -> None:
    if COLOR_CANDIDATE_MODEL_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite candidate weights: {COLOR_CANDIDATE_MODEL_PATH}"
        )
    torch.set_num_threads(min(4, torch.get_num_threads()))
    split_color_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = datasets.ImageFolder(
        SPLIT_COLOR_DIR / "train",
        transform=color_transform(training=True),
    )
    val_dataset = datasets.ImageFolder(
        SPLIT_COLOR_DIR / "val",
        transform=color_transform(),
    )
    class_names = train_dataset.classes
    print("Colours:", class_names)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    model = model.to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    best_val = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_correct = train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / max(val_total, 1)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train {(train_correct / max(train_total, 1)):.1%} | "
            f"val {val_accuracy:.1%}"
        )
        if val_accuracy >= best_val:
            best_val = val_accuracy
            COLOR_CANDIDATE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "architecture": "mobilenet_v3_small_color",
                },
                COLOR_CANDIDATE_MODEL_PATH,
            )
            print("  saved best colour checkpoint")

    print(f"Best colour validation accuracy: {best_val:.1%}")


if __name__ == "__main__":
    main()
