"""Colour prediction: trained classifier if available, otherwise HSV voting on the crop."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from config import COLOR_MODEL_PATH, COLOR_NAMES_PATH, COLOUR_NAMES, IMAGE_SIZE


def build_color_model(number_of_classes: int = 8) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, number_of_classes)
    return model


def color_transform() -> transforms.Compose:
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
    y1, y2 = int(height * 0.30), int(height * 0.82)
    x1, x2 = int(width * 0.12), int(width * 0.88)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return bgr
    return crop


def hsv_colour(crop: Image.Image) -> str:
    """Name one of the eight exam colours from the car body region.

    The method is not learned: it votes over HSV pixels after dropping
    windshield-like and glare pixels. Lighting still causes mistakes,
    which is why a trained colour model is preferred once crops are labelled.
    """

    bgr = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2BGR)
    body = _body_roi(bgr)
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)

    hue = hsv[:, :, 0].astype(np.int32).ravel()
    saturation = hsv[:, :, 1].astype(np.int32).ravel()
    value = hsv[:, :, 2].astype(np.int32).ravel()

    # Drop extreme shadows and specular highlights.
    usable = (value >= 25) & (value <= 245)
    if not np.any(usable):
        usable = np.ones_like(value, dtype=bool)

    hue = hue[usable]
    saturation = saturation[usable]
    value = value[usable]

    chromatic = saturation >= 45
    if np.mean(chromatic) < 0.18:
        median_value = float(np.median(value))
        median_saturation = float(np.median(saturation))
        if median_value < 55:
            return "black"
        if median_value > 175 and median_saturation < 40:
            return "white"
        if median_value < 95:
            return "black"
        return "grey"

    def bucket(h: int, s: int, v: int) -> str:
        if s < 40:
            if v < 55:
                return "black"
            if v > 175:
                return "white"
            return "grey"
        if v < 40:
            return "black"
        if h <= 8 or h >= 170:
            return "red"
        if h <= 18:
            return "brown"
        if h <= 34:
            return "yellow"
        if h <= 85:
            return "green"
        if h <= 135:
            return "blue"
        return "brown"

    votes = [bucket(int(h), int(s), int(v)) for h, s, v in zip(hue, saturation, value)]
    counts = {name: 0 for name in COLOUR_NAMES}
    for name in votes:
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
        return hsv_colour(crop), 1.0

    image_tensor = image_transform(crop.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)

    colour = class_names[int(predicted_index.item())].lower()
    if colour == "gray":
        colour = "grey"
    return colour, float(confidence.item())
