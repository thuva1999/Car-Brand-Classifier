"""Fine-tune MobileNetV3-Small on Tashkent brand folders."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from brand_model import brand_transform  # noqa: E402
from config import (  # noqa: E402
    BRAND_CANDIDATE_MODEL_PATH,
    IMAGE_SIZE,
    SPLIT_BRAND_DIR,
)

BATCH_SIZE = 16
EPOCHS = 12
HEAD_LEARNING_RATE = 1e-3
BODY_LEARNING_RATE = 3e-5
UNFREEZE_AFTER_EPOCH = 3


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(number_of_classes: int, device: torch.device) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, number_of_classes)
    return model.to(device)


def freeze_features(model: nn.Module, freeze: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = not freeze


def class_counts(dataset: datasets.ImageFolder) -> list[int]:
    counter = Counter(label for _, label in dataset.samples)
    return [counter[index] for index in range(len(dataset.classes))]


def make_loader(dataset: datasets.ImageFolder, batch_size: int, train: bool) -> DataLoader:
    if train:
        counts = class_counts(dataset)
        weights_per_class = [1.0 / max(count, 1) for count in counts]
        sample_weights = [weights_per_class[label] for _, label in dataset.samples]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    loss_total = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            loss_total += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return loss_total / max(total, 1), correct / max(total, 1)


def main() -> None:
    if BRAND_CANDIDATE_MODEL_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite candidate weights: {BRAND_CANDIDATE_MODEL_PATH}"
        )

    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = get_device()
    print("Device:", device)
    print("Torch CPU threads:", torch.get_num_threads())

    train_dataset = datasets.ImageFolder(
        SPLIT_BRAND_DIR / "train",
        transform=brand_transform(training=True),
    )
    val_dataset = datasets.ImageFolder(
        SPLIT_BRAND_DIR / "val",
        transform=brand_transform(),
    )
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Train and val class folders do not match.")
    if not train_dataset or not val_dataset:
        raise ValueError(
            f"Brand split must contain training and validation images: {SPLIT_BRAND_DIR}"
        )

    class_names = train_dataset.classes
    print("Brands:", class_names)
    print("Train images:", len(train_dataset))
    print("Val images:", len(val_dataset))
    print("Per-class train counts:", dict(zip(class_names, class_counts(train_dataset))))
    if any(count == 0 for count in class_counts(train_dataset)):
        raise ValueError("Every included brand must have at least one training image.")

    train_loader = make_loader(train_dataset, BATCH_SIZE, train=True)
    val_loader = make_loader(val_dataset, BATCH_SIZE, train=False)

    model = build_model(len(class_names), device)
    freeze_features(model, freeze=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=HEAD_LEARNING_RATE)

    BRAND_CANDIDATE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0.0
    unfrozen = False

    for epoch in range(EPOCHS):
        if epoch + 1 == UNFREEZE_AFTER_EPOCH and not unfrozen:
            freeze_features(model, freeze=False)
            optimizer = torch.optim.Adam(
                [
                    {"params": model.features.parameters(), "lr": BODY_LEARNING_RATE},
                    {"params": model.classifier.parameters(), "lr": HEAD_LEARNING_RATE / 5},
                ]
            )
            unfrozen = True
            print("Unfroze MobileNetV3-Small feature extractor.")

        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.1%} | "
            f"val loss {val_loss:.4f} acc {val_accuracy:.1%}"
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "architecture": "mobilenet_v3_small",
                "image_size": IMAGE_SIZE,
                "val_accuracy": val_accuracy,
            }
            torch.save(checkpoint, BRAND_CANDIDATE_MODEL_PATH)
            print("  saved best checkpoint")

    print(f"\nBest validation accuracy: {best_val_accuracy:.1%}")
    print("Candidate model saved in:", BRAND_CANDIDATE_MODEL_PATH)


if __name__ == "__main__":
    main()
