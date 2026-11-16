"""Fine-tune an ImageNet backbone on the Tashkent brand split.

    python training/train_brand_classifier.py
    python training/train_brand_classifier.py --arch mobilenet_v3_large --image-size 288

Recipe: pad-to-square crops (no stretching), stronger augmentation, AdamW with
warmup + cosine decay, label smoothing, capped oversampling of small brands,
and checkpoint selection by *balanced* accuracy so small brands count as much
as the big imported ones.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from brand_model import (  # noqa: E402
    PREPROCESS_PAD,
    SUPPORTED_ARCHITECTURES,
    brand_transform,
    build_brand_model,
)
from config import (  # noqa: E402
    BRAND_CANDIDATE_MODEL_PATH,
    SPLIT_BRAND_DIR,
)

DEFAULT_ARCH = "mobilenet_v3_small"
DEFAULT_IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 30
WARMUP_EPOCHS = 2
HEAD_ONLY_EPOCHS = 1
LEARNING_RATE = 1e-3          # head
BACKBONE_LR_FACTOR = 0.3      # backbone lr = LEARNING_RATE * factor
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
# A small brand is repeated at most this many times more often than its real
# count per epoch; uncapped oversampling makes the model memorise 16 images.
MAX_OVERSAMPLE = 3.0


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_counts(dataset: datasets.ImageFolder) -> list[int]:
    counter = Counter(label for _, label in dataset.samples)
    return [counter[index] for index in range(len(dataset.classes))]


def make_train_loader(dataset, batch_size: int, workers: int) -> DataLoader:
    counts = class_counts(dataset)
    largest = max(counts)
    # Target samples per class per epoch: balanced, but never more than
    # MAX_OVERSAMPLE x the real number of images.
    targets = [min(largest, count * MAX_OVERSAMPLE) for count in counts]
    weight_per_class = [target / count for target, count in zip(targets, counts)]
    sample_weights = [weight_per_class[label] for _, label in dataset.samples]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=int(sum(targets)), replacement=True
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=workers > 0,
        drop_last=True,
    )


def backbone_and_head(model: nn.Module) -> tuple[list, list]:
    head = list(model.classifier.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone, head


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    backbone, _ = backbone_and_head(model)
    for parameter in backbone:
        parameter.requires_grad = trainable


def lr_lambda_factory(total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return lr_lambda


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device) -> tuple[float, float]:
    model.train()
    loss_total = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        loss_total += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_total / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, number_of_classes: int):
    model.eval()
    loss_total = correct = total = 0
    per_class_correct = [0] * number_of_classes
    per_class_total = [0] * number_of_classes
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss_total += criterion(outputs, labels).item() * images.size(0)
        predicted = outputs.argmax(1)
        for true, pred in zip(labels.tolist(), predicted.tolist()):
            per_class_total[true] += 1
            if true == pred:
                per_class_correct[true] += 1
                correct += 1
            total += 1
    recalls = [c / t for c, t in zip(per_class_correct, per_class_total) if t]
    balanced = sum(recalls) / max(len(recalls), 1)
    return loss_total / max(total, 1), correct / max(total, 1), balanced


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the brand classifier.")
    parser.add_argument("--arch", choices=SUPPORTED_ARCHITECTURES, default=DEFAULT_ARCH)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_BRAND_DIR)
    parser.add_argument("--output", type=Path, default=BRAND_CANDIDATE_MODEL_PATH)
    parser.add_argument("--workers", type=int, default=2, help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate weights: {args.output}")

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = get_device()
    print(f"Device: {device} | arch {args.arch} | image size {args.image_size}")

    train_dataset = datasets.ImageFolder(
        args.split_dir / "train",
        transform=brand_transform(True, args.image_size, PREPROCESS_PAD),
    )
    val_dataset = datasets.ImageFolder(
        args.split_dir / "val",
        transform=brand_transform(False, args.image_size, PREPROCESS_PAD),
    )
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Train and val class folders do not match.")
    counts = class_counts(train_dataset)
    if any(count == 0 for count in counts):
        raise ValueError("Every included brand must have at least one training image.")

    class_names = train_dataset.classes
    print("Train images:", len(train_dataset), "| Val images:", len(val_dataset))
    print("Per-class train counts:", dict(zip(class_names, counts)))

    train_loader = make_train_loader(train_dataset, args.batch_size, args.workers)
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    model = build_brand_model(len(class_names), args.arch, pretrained=True).to(device)
    backbone, head = backbone_and_head(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": args.lr * BACKBONE_LR_FACTOR},
            {"params": head, "lr": args.lr},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda_factory(args.epochs * steps_per_epoch, WARMUP_EPOCHS * steps_per_epoch),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    plain_criterion = nn.CrossEntropyLoss()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_balanced = -1.0
    set_backbone_trainable(model, False)

    for epoch in range(args.epochs):
        if epoch == HEAD_ONLY_EPOCHS:
            set_backbone_trainable(model, True)
            print("Unfroze backbone.")
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_acc, val_balanced = evaluate(
            model, val_loader, plain_criterion, device, len(class_names)
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} ({time.time() - start:.0f}s) | "
            f"train loss {train_loss:.3f} acc {train_acc:.1%} | "
            f"val loss {val_loss:.3f} acc {val_acc:.1%} balanced {val_balanced:.1%}"
        )

        if val_balanced > best_balanced:
            best_balanced = val_balanced
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "architecture": args.arch,
                    "image_size": args.image_size,
                    "preprocess": PREPROCESS_PAD,
                    "val_accuracy": val_acc,
                    "val_balanced_accuracy": val_balanced,
                    "epoch": epoch + 1,
                },
                args.output,
            )
            print("  saved best checkpoint")

    print(f"\nBest balanced validation accuracy: {best_balanced:.1%}")
    print("Candidate model saved in:", args.output)


if __name__ == "__main__":
    main()
