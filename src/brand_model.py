"""Load and run the team's MobileNetV3-Small brand classifier."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from config import (
    BRAND_MODEL_PATH,
    BRAND_UNKNOWN_THRESHOLD,
    IMAGE_SIZE,
)


def build_brand_model(number_of_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, number_of_classes)
    return model


def brand_transform(training: bool = False) -> transforms.Compose:
    operations = []
    if training:
        operations.extend(
            [
                transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.25,
                    hue=0.02,
                ),
                transforms.RandomRotation(12),
            ]
        )
    else:
        operations.append(transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    if training:
        operations.append(transforms.RandomErasing(p=0.15))
    return transforms.Compose(operations)


def load_brand_model(model_path: Path | None = None, device: torch.device | None = None):
    path = Path(model_path) if model_path else BRAND_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Brand model not found: {path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        class_names = checkpoint.get("class_names")
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint),
        )
    else:
        class_names = None
        state_dict = checkpoint

    if not isinstance(class_names, list) or not class_names:
        raise ValueError(
            f"Brand checkpoint must contain its ordered 'class_names': {path}"
        )
    output_features = state_dict["classifier.3.weight"].shape[0]
    if output_features != len(class_names):
        raise ValueError(
            f"Brand checkpoint has {output_features} output classes but "
            f"{len(class_names)} class names: {path}"
        )

    model = build_brand_model(len(class_names))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names, brand_transform(), device


def predict_brand(
    crop: Image.Image,
    model,
    class_names: list[str],
    image_transform,
    device,
    unknown_threshold: float = BRAND_UNKNOWN_THRESHOLD,
) -> tuple[str, float]:
    image_tensor = image_transform(crop.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)

    confidence_value = float(confidence.item())
    brand = class_names[int(predicted_index.item())]
    if confidence_value < unknown_threshold:
        return "unknown", confidence_value
    return brand, confidence_value
