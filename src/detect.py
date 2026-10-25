"""Find cars in a street photo with a COCO-pretrained YOLOv8n detector.

The detector is used only to crop vehicles. Brand and colour come from
models we train ourselves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    DETECT_CONFIDENCE,
    MIN_BOX_RELATIVE_AREA,
    MIN_BOX_SIDE_PX,
    VEHICLE_CLASS_IDS,
    YOLO_WEIGHTS_PATH,
)


def load_detector(weights_path: Path | None = None):
    try:
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
    except ImportError as error:
        raise ImportError(
            "ultralytics is required for car detection. "
            "Install it with: pip install ultralytics"
        ) from error

    SETTINGS["sync"] = False
    path = Path(weights_path) if weights_path else YOLO_WEIGHTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return YOLO(str(path))

    # First run may download yolov8n.pt; keep a copy in models/ for the live test.
    model = YOLO("yolov8n.pt")
    downloaded = Path(getattr(model, "ckpt_path", "") or "yolov8n.pt")
    if downloaded.exists() and downloaded.resolve() != path.resolve():
        try:
            path.write_bytes(downloaded.read_bytes())
        except OSError:
            pass
    if path.exists():
        return YOLO(str(path))
    return model


def _keep_box(x1, y1, x2, y2, image_width, image_height) -> bool:
    width = x2 - x1
    height = y2 - y1
    if width < MIN_BOX_SIDE_PX or height < MIN_BOX_SIDE_PX:
        return False
    area = width * height
    relative_area = area / float(image_width * image_height)
    return relative_area >= MIN_BOX_RELATIVE_AREA


def detect_cars(image: Image.Image, detector) -> list[dict]:
    """Return kept vehicle boxes in pixel coordinates (xyxy)."""

    image_width, image_height = image.size
    results = detector.predict(
        source=np.array(image.convert("RGB")),
        conf=DETECT_CONFIDENCE,
        classes=VEHICLE_CLASS_IDS,
        verbose=False,
    )

    cars = []
    if not results:
        return cars

    boxes = results[0].boxes
    if boxes is None:
        return cars

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = int(max(0, min(image_width - 1, x1)))
        y1 = int(max(0, min(image_height - 1, y1)))
        x2 = int(max(0, min(image_width, x2)))
        y2 = int(max(0, min(image_height, y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        if not _keep_box(x1, y1, x2, y2, image_width, image_height):
            continue
        cars.append(
            {
                "box": (x1, y1, x2, y2),
                "detect_confidence": float(box.conf[0].item()),
            }
        )

    cars.sort(key=lambda item: item["detect_confidence"], reverse=True)
    return cars


def crop_box(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = box
    return image.crop((x1, y1, x2, y2))
