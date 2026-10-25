"""Phone photo -> list of (brand, colour) pairs, fully offline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from brand_model import load_brand_model, predict_brand
from color_model import load_color_model, predict_color
from config import PROJECT_DIR
from detect import crop_box, detect_cars, load_detector


class CarPipeline:
    def __init__(
        self,
        brand_model_path: Path | None = None,
        load_vehicle_detector: bool = True,
    ):
        self.detector = load_detector() if load_vehicle_detector else None
        (
            self.brand_model,
            self.brand_names,
            self.brand_transform,
            self.device,
        ) = load_brand_model(model_path=brand_model_path)
        (
            self.color_model,
            self.color_names,
            self.color_transform,
            color_device,
        ) = load_color_model(device=self.device)
        if color_device is not None:
            self.device = color_device

    def predict_image(self, image: Image.Image) -> list[dict]:
        if self.detector is None:
            raise RuntimeError("Vehicle detection was disabled for this pipeline.")
        cars = detect_cars(image, self.detector)
        predictions = []
        for car in cars:
            crop = crop_box(image, car["box"])
            predictions.append(
                self._predict_crop(crop, car["box"], car["detect_confidence"])
            )
        return predictions

    def predict_crop(self, crop: Image.Image) -> list[dict]:
        """Classify an already-cropped single car without invoking YOLO."""

        return [
            self._predict_crop(
                crop,
                (0, 0, crop.width, crop.height),
                1.0,
            )
        ]

    def _predict_crop(
        self,
        crop: Image.Image,
        box: tuple[int, int, int, int],
        detect_confidence: float,
    ) -> dict:
        brand, brand_confidence = predict_brand(
            crop,
            self.brand_model,
            self.brand_names,
            self.brand_transform,
            self.device,
        )
        colour, colour_confidence = predict_color(
            crop,
            self.color_model,
            self.color_names,
            self.color_transform,
            self.device,
        )
        return {
            "brand": brand,
            "colour": colour,
            "brand_confidence": brand_confidence,
            "colour_confidence": colour_confidence,
            "detect_confidence": detect_confidence,
            "box": box,
        }

    def predict_path(self, image_path: Path) -> list[dict]:
        image = Image.open(image_path).convert("RGB")
        return self.predict_image(image)


def format_answer_line(predictions: list[dict]) -> str:
    if not predictions:
        return "(no classifiable cars)"
    return "; ".join(f"{item['brand']} {item['colour']}" for item in predictions)


def annotate_image(image: Image.Image, predictions: list[dict]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    for item in predictions:
        x1, y1, x2, y2 = item["box"]
        label = f"{item['brand']} {item['colour']}"
        draw.rectangle((x1, y1, x2, y2), outline=(0, 200, 80), width=3)
        text_top = max(0, y1 - 16)
        draw.rectangle((x1, text_top, x1 + 8 * len(label), text_top + 16), fill=(0, 200, 80))
        draw.text((x1 + 2, text_top + 2), label, fill=(0, 0, 0), font=font)
    return canvas


def save_annotated(image: Image.Image, predictions: list[dict], output_path: Path) -> None:
    annotated = annotate_image(image, predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)


def bgr_from_pil(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def resolve_image_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_file():
        return path

    project_path = path if path.is_absolute() else PROJECT_DIR / path
    if project_path.is_file():
        return project_path

    # Allow a dataset image to be specified by filename alone.
    if path.name == str(path) and path.name:
        data_dir = PROJECT_DIR / "data"
        matches = [candidate for candidate in data_dir.rglob(path.name) if candidate.is_file()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            candidates = "\n".join(f"  {candidate}" for candidate in matches[:10])
            raise ValueError(
                f"More than one dataset image is named {path.name!r}. "
                f"Use its full path. Matches:\n{candidates}"
            )

    raise FileNotFoundError(
        f"Image not found: {raw_path}\n"
        f"Checked the current directory, project directory, and dataset folders under "
        f"{PROJECT_DIR / 'data'}."
    )
