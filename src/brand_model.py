"""Load and run the team's MobileNetV3-Small brand classifier."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from config import (
    BRAND_MODEL_PATH,
    BRAND_NAMES_PATH,
    BRAND_UNKNOWN_THRESHOLD,
    IMAGE_SIZE,
)


def build_brand_model(number_of_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, number_of_classes)
    return model


def brand_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_brand_model(model_path: Path | None = None, device: torch.device | None = None):
    path = Path(model_path) if model_path else BRAND_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Brand model not found: {path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
    elif BRAND_NAMES_PATH.exists():
        with open(BRAND_NAMES_PATH, encoding="utf-8") as file:
            class_names = json.load(file)
    else:
        raise FileNotFoundError("Could not find brand class names.")

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint),
        )
    else:
        state_dict = checkpoint

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
