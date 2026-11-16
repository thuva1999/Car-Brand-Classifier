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


def _body_band(height: int, width: int) -> tuple[slice, slice]:
    # Doors/bonnet band: skip roof and windscreen above, wheels and road below.
    rows = slice(int(height * 0.30), int(height * 0.70))
    cols = slice(int(width * 0.15), int(width * 0.85))
    return rows, cols


def _paint_pixels(bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """HSV pixels (N x 3) that most likely show this car's paint.

    Without a mask: the fixed body band. With a car mask: the band restricted
    to this car's outline (eroded so edge pixels bleeding into road or a
    neighbouring car are dropped). Falls back to the whole eroded mask, then
    to the plain band, if the overlap is too small to trust.
    """

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    rows, cols = _body_band(height, width)
    band = np.zeros((height, width), dtype=bool)
    band[rows, cols] = True
    band_hsv = hsv[band] if band.any() else hsv.reshape(-1, 3)

    if mask is None or mask.shape != (height, width) or not mask.any():
        return band_hsv

    radius = max(1, int(min(height, width) * 0.03))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    car = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    if not car.any():
        car = mask

    minimum = max(100, int(band.sum() * 0.15))
    for selection in (car & band, car):
        if selection.sum() >= minimum:
            return hsv[selection]
    return band_hsv


def _warm_fraction(hsv_pixels: np.ndarray) -> float:
    """Share of well-lit pixels with a warm (orange/beige) tint."""

    hue = hsv_pixels[:, 0].astype(np.int32)
    saturation = hsv_pixels[:, 1].astype(np.int32)
    value = hsv_pixels[:, 2].astype(np.int32)
    well_lit = (value >= 60) & (value <= 235)
    warm = well_lit & (saturation >= 30) & (hue >= 11) & (hue <= 30)
    return float(np.count_nonzero(warm)) / max(len(hsv_pixels), 1)


def scene_warmth(
    image: Image.Image, boxes: list[tuple[int, int, int, int]]
) -> float | None:
    """Warm fraction of the photo's background (everything outside car boxes).

    Golden-hour sun or a warm camera setting tints white paint cream, which
    looks like beige. If the road and buildings are just as warm, the warmth
    comes from the light, not the paint. None when cars fill the photo.
    """

    rgb = np.array(image.convert("RGB"))
    background = np.ones(rgb.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in boxes:
        background[y1:y2, x1:x2] = False
    if background.sum() < background.size * 0.05:
        return None
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    return _warm_fraction(hsv[background])


def hsv_colour(
    crop: Image.Image,
    mask: np.ndarray | None = None,
    scene_warm: float | None = None,
) -> str:
    """Pick one of the eight exam colours from the car's paint pixels."""

    return hsv_colour_with_confidence(crop, mask, scene_warm)[0]


def _rule_confidence(margin: float) -> float:
    """Map how far a measurement is past its decision threshold (0 = right on
    the line, 1 = comfortably past it) to a 0.50-0.95 confidence. Capped at
    0.95 because these are hand-tuned rules, not a calibrated probability."""

    return round(0.50 + 0.45 * float(np.clip(margin, 0.0, 1.0)), 2)


def hsv_colour_with_confidence(
    crop: Image.Image,
    mask: np.ndarray | None = None,
    scene_warm: float | None = None,
) -> tuple[str, float]:
    """Pick one of the eight exam colours and say how clear-cut the call was.

    ``mask`` (optional) is a boolean array the size of the crop, True on this
    car's own pixels (from YOLO segmentation). It stops a neighbouring car,
    road or wall inside the box from voting.

    Tuned on Tashkent street photos, where the camera adds a bluish cast:
    grey/white/black paint and sky reflections show up as hue ~104 with
    saturation 30-60. So:

    1. Red, yellow, green or blue must cover >= 15% of the paint pixels
       with well-lit, saturated pixels. Blue needs S >= 100, because the cast
       sits in the blue hue range; red and green are cast-free, so S >= 60 is
       enough (dull lime/olive paint in shade). Yellow needs S >= 140 so beige
       is not mistaken for it. Blue does not count on a dark body (darkest
       quarter V < 80) unless vivid: that is black paint under a blue cast.
       Red does not count when the red pixels are very bright (lit tail
       lights on a rear view).
    2. Beige/gold/brown paint is warm but dull, so it is detected separately:
       warm hue (11-30) with S >= 30 covering >= 25% of the paint pixels,
       and at least 0.15 more than the background (``scene_warm``, from
       ``scene_warmth``) so warm sunlight on white paint is not called brown.
    3. Otherwise the car is neutral and the median brightness decides
       black / grey / white. Black paint reflects a lot, so its median V is
       typically 20-65, not near 0; reflective black photographed off a
       screen can have a median up to ~125 but keeps a dark core (darkest
       quarter V <= 56), so that is black too. A dark car (V < 110) whose lit pixels are
       strongly saturated (S >= 75) is dark-coloured paint such as greenish
       black, not grey: black. Shaded white (V 150-175) is told apart from
       silver by its lower saturation.
    """

    bgr = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2BGR)
    pixels = _paint_pixels(bgr, mask).astype(np.int32)
    if len(pixels) == 0:
        raise ValueError("No usable body pixels for HSV colour voting.")

    hue, saturation, value = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    total = len(pixels)

    well_lit = (value >= 60) & (value <= 235)
    saturated = well_lit & (saturation >= 60)
    strong = well_lit & (saturation >= 100)
    colour_masks = {
        "red": saturated & ((hue <= 10) | (hue >= 170)),
        "yellow": well_lit & (hue >= 20) & (hue <= 35) & (saturation >= 140) & (value >= 145),
        "green": saturated & (hue >= 36) & (hue <= 84),
        "blue": strong & (hue >= 85) & (hue <= 169),
    }
    strong_counts = {name: int(np.count_nonzero(m)) for name, m in colour_masks.items()}
    # Darkest quarter of the body. Glossy black paint photographed off a
    # screen is covered in bright reflections (trees, sky, watermarks), which
    # lift the median, but its dark core survives: p25 <= ~56. Grey and white
    # bodies stay brighter, apart from wheel arches. Measured on 106
    # hand-labelled live cars (outputs/colour_labelling, 16 Nov).
    dark_quarter = float(np.percentile(value, 25))
    median_value = float(np.median(value))

    best = max(strong_counts, key=strong_counts.get)
    best_fraction = strong_counts[best] / total
    if best_fraction >= 0.15 and best == "blue" and dark_quarter < 80:
        # Dark body with blue-tinted reflections: black paint under the
        # screen's blue cast, not blue paint. Real blue bodies are lighter
        # (p25 >= ~90) unless the blue is vivid (S >= 185).
        if float(np.median(saturation[colour_masks["blue"]])) < 185:
            best_fraction = 0.0
    if best_fraction >= 0.15 and best == "red":
        # Lit tail lights on a rear view are bright red (V ~200); red paint
        # sits far lower (V ~90-140).
        if float(np.median(value[colour_masks["red"]])) >= 180:
            best_fraction = 0.0
    if best_fraction >= 0.15:
        return best, _rule_confidence((best_fraction - 0.15) / 0.35)

    warm_fraction = _warm_fraction(pixels)
    warmer_than_scene = (
        1.0 if scene_warm is None else (warm_fraction - scene_warm - 0.15) / 0.15
    )
    if warm_fraction >= 0.25 and warmer_than_scene >= 0:
        margin = min((warm_fraction - 0.25) / 0.35, warmer_than_scene)
        return "brown", _rule_confidence(margin)
    if warm_fraction >= 0.25:
        # Warm because of the light: judge it as a neutral car, and do not
        # count the tint against the neutral answer below.
        warm_fraction = 0.0

    # Neutral: less sure when a colour almost won the votes above.
    colour_closeness = 1.0 - max(best_fraction / 0.15, warm_fraction / 0.25)

    if median_value < 65:
        margin = (65 - median_value) / 40
        return "black", _rule_confidence(min(margin, colour_closeness))
    if dark_quarter <= 56 and median_value <= 125:
        # Reflective black: dark core, reflections lift the median.
        margin = min((56 - dark_quarter) / 30, (125 - median_value) / 40)
        return "black", _rule_confidence(min(margin, colour_closeness))
    # Very dark green/blue/teal paint (e.g. "greenish black") is too dark for
    # the colour votes above but far too saturated to be grey paint (grey with
    # the camera's blue cast sits at S ~30-55). Call it black.
    lit_body = value >= 60
    body_saturation = float(np.median(saturation[lit_body])) if lit_body.any() else 0.0
    if median_value < 110 and body_saturation >= 75:
        margin = min((110 - median_value) / 40, (body_saturation - 75) / 40)
        return "black", _rule_confidence(min(margin, colour_closeness))
    if median_value >= 175:
        margin = (median_value - 175) / 40
        return "white", _rule_confidence(min(margin, colour_closeness))
    # White paint in shade drops to V ~150-175, overlapping silver. Silver /
    # grey metallic picks up the bluish sky cast (body S ~40-55), while white
    # paint stays almost colourless (S ~10-30; a screen photo of a white
    # Daewoo Nexia measured S 30, V 161). Ignore glass/tyres (V < 60).
    if median_value >= 150 and body_saturation < 32:
        margin = min((median_value - 150) / 25, (32 - body_saturation) / 15)
        return "white", _rule_confidence(min(margin, colour_closeness))

    margin = min(median_value - 65, 175 - median_value) / 40
    if median_value >= 150:
        margin = min(margin, (body_saturation - 32) / 15)
    return "grey", _rule_confidence(min(margin, colour_closeness))


def predict_color(
    crop: Image.Image,
    model=None,
    class_names: list[str] | None = None,
    image_transform=None,
    device=None,
    mask: np.ndarray | None = None,
    scene_warm: float | None = None,
) -> tuple[str, float]:
    if model is None:
        return hsv_colour_with_confidence(crop, mask, scene_warm)

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
        return hsv_colour_with_confidence(crop, mask, scene_warm)
    return colour, confidence_value
