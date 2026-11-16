"""Train a brand classifier on frozen DINOv2 ViT-S/14 features (CPU friendly).

    python training/train_brand_dinov2.py
    python training/train_brand_dinov2.py --aug-passes 4 --output models/x.pth

Fine-tuning a vision transformer on a laptop CPU takes many hours. DINOv2's
frozen features already separate car makes well, so instead:

1. Run the backbone once over every crop and cache the features
   (train: the clean crop, its mirror, and --aug-passes augmented copies;
   val: the clean crop and its mirror, matching flip-averaging at inference).
2. Train only the linear head on the cached features (seconds), trying a few
   weight-decay values and keeping the best *balanced* validation accuracy.
3. Save a normal checkpoint (architecture "dinov2_vits14") that
   load_brand_model / serve.py / infer.py / evaluate.py load unchanged.

The feature cache lives in outputs/ and is reused when the split and settings
have not changed, so re-running with other head settings is instant.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from brand_model import (  # noqa: E402
    DINOV2_ARCH,
    PREPROCESS_PAD,
    brand_transform,
    build_brand_model,
)
from config import (  # noqa: E402
    BRAND_CANDIDATE_MODEL_PATH,
    SPLIT_BRAND_DIR,
)

IMAGE_SIZE = 224
HEAD_EPOCHS = 300
HEAD_LR = 1e-3
WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2, 5e-2)
LABEL_SMOOTHING = 0.1


@torch.no_grad()
def extract(model, dataset, workers: int, flip: bool = False, label: str = "") -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=workers)
    chunks = []
    start = time.time()
    done = 0
    for images, _ in loader:
        if flip:
            images = torch.flip(images, dims=[3])
        chunks.append(model.features(images))
        done += images.size(0)
        print(f"\r  {label}: {done}/{len(dataset)} ({time.time() - start:.0f}s)", end="", flush=True)
    print()
    return torch.cat(chunks)


def cached_features(model, split_dir: Path, aug_passes: int, workers: int) -> dict:
    cache = PROJECT_DIR / "outputs" / f"dinov2_features_{split_dir.name}_{IMAGE_SIZE}_aug{aug_passes}.pt"
    train_clean = datasets.ImageFolder(
        split_dir / "train", transform=brand_transform(False, IMAGE_SIZE, PREPROCESS_PAD)
    )
    train_aug = datasets.ImageFolder(
        split_dir / "train", transform=brand_transform(True, IMAGE_SIZE, PREPROCESS_PAD)
    )
    val = datasets.ImageFolder(
        split_dir / "val", transform=brand_transform(False, IMAGE_SIZE, PREPROCESS_PAD)
    )
    if train_clean.classes != val.classes:
        raise ValueError("Train and val class folders do not match.")
    signature = [path for path, _ in train_clean.samples] + [path for path, _ in val.samples]

    if cache.exists():
        data = torch.load(cache, weights_only=False)
        if data.get("signature") == signature:
            print(f"Using cached features: {cache}")
            return data
        print("Split changed since the cache was made; re-extracting.")

    print(f"Extracting DINOv2 features ({len(train_clean)} train, {len(val)} val images)...")
    train_features = [
        extract(model, train_clean, workers, label="train clean"),
        extract(model, train_clean, workers, flip=True, label="train mirrored"),
    ]
    for index in range(aug_passes):
        torch.manual_seed(1000 + index)
        train_features.append(extract(model, train_aug, workers, label=f"train augmented {index + 1}"))
    train_labels = torch.tensor([label for _, label in train_clean.samples])
    data = {
        "signature": signature,
        "class_names": train_clean.classes,
        "train_x": torch.cat(train_features),
        "train_y": train_labels.repeat(len(train_features)),
        "val_x": extract(model, val, workers, label="val clean"),
        "val_x_flip": extract(model, val, workers, flip=True, label="val mirrored"),
        "val_y": torch.tensor([label for _, label in val.samples]),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache)
    print(f"Saved feature cache: {cache}")
    return data


def balanced_accuracy(predicted: torch.Tensor, labels: torch.Tensor, number: int) -> tuple[float, float]:
    recalls = []
    for index in range(number):
        mask = labels == index
        if mask.any():
            recalls.append((predicted[mask] == index).float().mean().item())
    return (predicted == labels).float().mean().item(), sum(recalls) / len(recalls)


def train_head(data: dict, weight_decay: float, seed: int):
    torch.manual_seed(seed)
    x, y = data["train_x"], data["train_y"]
    number = len(data["class_names"])
    counts = torch.bincount(y, minlength=number).float()
    # Class-balanced loss: small brands count as much as big ones.
    class_weights = (counts.sum() / (number * counts)).clamp(max=10.0)
    head = nn.Linear(x.shape[1], number)
    optimizer = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, HEAD_EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    best = (-1.0, 0.0, None, 0)
    for epoch in range(HEAD_EPOCHS):
        head.train()
        order = torch.randperm(len(x))
        for start in range(0, len(x), 256):
            batch = order[start:start + 256]
            optimizer.zero_grad(set_to_none=True)
            criterion(head(x[batch]), y[batch]).backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            head.eval()
            with torch.no_grad():
                probabilities = (
                    torch.softmax(head(data["val_x"]), 1) + torch.softmax(head(data["val_x_flip"]), 1)
                ) / 2
            accuracy, balanced = balanced_accuracy(probabilities.argmax(1), data["val_y"], number)
            if balanced > best[0]:
                best = (balanced, accuracy, {k: v.clone() for k, v in head.state_dict().items()}, epoch + 1)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DINOv2 brand classifier.")
    parser.add_argument("--split-dir", type=Path, default=SPLIT_BRAND_DIR)
    parser.add_argument("--output", type=Path, default=BRAND_CANDIDATE_MODEL_PATH)
    parser.add_argument("--aug-passes", type=int, default=1,
                        help="Extra randomly augmented copies of each training crop.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate weights: {args.output}")

    torch.set_num_threads(min(8, torch.get_num_threads()))
    model = build_brand_model(1, DINOV2_ARCH, pretrained=True, image_size=IMAGE_SIZE).eval()
    data = cached_features(model, args.split_dir, args.aug_passes, args.workers)
    class_names = data["class_names"]
    clean_train_counts = Counter(data["train_y"][: len(data["train_y"]) // (args.aug_passes + 2)].tolist())
    print("Train counts:", {name: clean_train_counts[i] for i, name in enumerate(class_names)})

    best = None
    for weight_decay in WEIGHT_DECAYS:
        balanced, accuracy, state, epoch = train_head(data, weight_decay, args.seed)
        print(f"weight decay {weight_decay:g}: val acc {accuracy:.1%} balanced {balanced:.1%} (epoch {epoch})")
        if best is None or balanced > best[0]:
            best = (balanced, accuracy, state, epoch, weight_decay)
    balanced, accuracy, head_state, epoch, weight_decay = best

    final = build_brand_model(len(class_names), DINOV2_ARCH, pretrained=True, image_size=IMAGE_SIZE)
    final.classifier[0].load_state_dict(head_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": final.state_dict(),
            "class_names": class_names,
            "architecture": DINOV2_ARCH,
            "image_size": IMAGE_SIZE,
            "preprocess": PREPROCESS_PAD,
            "val_accuracy": accuracy,
            "val_balanced_accuracy": balanced,
            "epoch": epoch,
            "head_weight_decay": weight_decay,
            "backbone": "frozen DINOv2 ViT-S/14 (linear probe)",
        },
        args.output,
    )
    print(f"\nBest: val acc {accuracy:.1%}, balanced {balanced:.1%} (weight decay {weight_decay:g})")
    print("Candidate model saved in:", args.output)


if __name__ == "__main__":
    main()
