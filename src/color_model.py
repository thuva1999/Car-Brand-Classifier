"""Colour prediction: trained classifier if available, otherwise HSV voting on the crop."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from config import (
    COLOR_MODEL_CONFIDENCE,
    COLOR_MODEL_PATH,
    COLOR_NAMES_PATH,
    COLOUR_ALIASES,
    COLOUR_NAMES,
    IMAGE_SIZE,
)


def build_color_model(number_of_classes: int | None = None) -> nn.Module:
    if number_of_classes is None:
        number_of_classes = len(COLOUR_NAMES)
    model = models.mobilenet_v3_small(weights=None)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, number_of_classes)
    return model


def color_transform(training: bool = False) -> transforms.Compose:
    operations = []
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.15,
                ),
            ]
        )
    operations.extend(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(operations)


def load_color_model(model_path: Path | None = None, device: torch.device | None = None):
    path = Path(model_path) if model_path else COLOR_MODEL_PATH
    if not path.exists():
        return None, COLOUR_NAMES, color_transform(), device

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
    elif COLOR_NAMES_PATH.exists():
        import json

        with open(COLOR_NAMES_PATH, encoding="utf-8") as file:
            class_names = json.load(file)
    else:
        class_names = COLOUR_NAMES

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint),
        )
    else:
        state_dict = checkpoint

    model = build_color_model(len(class_names))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names, color_transform(), device


def _body_roi(bgr: np.ndarray) -> np.ndarray:
    height, width = bgr.shape[:2]
    y1, y2 = int(height * 0.32), int(height * 0.76)
    x1, x2 = int(width * 0.15), int(width * 0.85)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return bgr
    return crop


def hsv_colour(crop: Image.Image) -> str:
    """Map HSV paint pixels in the central body region to an exam colour.

    Maroon maps to red, orange to yellow, silver to grey, and beige/gold to
    brown so predictions remain within the exam's eight scored classes.
    """

    bgr = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2BGR)
    body = _body_roi(bgr)
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)

    hue = hsv[:, :, 0].astype(np.int32).ravel()
    saturation = hsv[:, :, 1].astype(np.int32).ravel()
    value = hsv[:, :, 2].astype(np.int32).ravel()
    usable = value >= 20
    if not np.any(usable):
        usable = np.ones_like(value, dtype=bool)

    hue = hue[usable]
    saturation = saturation[usable]
    value = value[usable]

    def bucket(h: int, s: int, v: int) -> str:
        if s < 35:
            if v < 90:
                return "black"
            if v >= 215:
                return "white"
            if 10 <= h <= 36 and s >= 12 and v > 130:
                return "beige"
            if v > 165:
                return "silver"
            return "grey"
        if h <= 8 or h >= 170:
            return "maroon" if v < 120 else "red"
        if 9 <= h <= 36 and s >= 100:
            if h <= 19:
                return "brown" if v < 210 else "orange"
            if h <= 24:
                return "gold"
            return "yellow" if v >= 150 else "gold"
        if 9 <= h <= 36:
            return "gold" if s >= 35 else "beige"
        if v < 40:
            return "black"
        if h <= 85:
            return "green"
        if h <= 135:
            return "blue"
        if h <= 169:
            return "purple"
        return "maroon"

    votes = []
    for h, s, v in zip(hue, saturation, value):
        name = bucket(int(h), int(s), int(v))
        votes.append(COLOUR_ALIASES.get(name, name))
    counts = {name: 0 for name in COLOUR_NAMES}
    for name in votes:
        if name in counts:
            counts[name] += 1
    return max(counts, key=counts.get)


def predict_color(
    crop: Image.Image,
    model=None,
    class_names: list[str] | None = None,
    image_transform=None,
    device=None,
) -> tuple[str, float]:
    if model is None:
        return hsv_colour(crop), 0.0

    image_tensor = image_transform(crop.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)

    colour = class_names[int(predicted_index.item())].lower()
    if colour == "gray":
        colour = "grey"
    confidence_value = float(confidence.item())
    if confidence_value < COLOR_MODEL_CONFIDENCE:
        return hsv_colour(crop), 0.0
    return colour, confidence_value
