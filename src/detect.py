"""Find cars in a street photo with a COCO-pretrained YOLO detector.

YOLO11n-seg is preferred: besides the box it outlines each car, so colour is
read from that car's own pixels instead of neighbours, road and background.
The detector is used only to locate vehicles. Brand and colour come from
models we train ourselves.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import (
    DETECT_CONFIDENCE,
    DETECT_NMS_IOU,
    MIN_BOX_RELATIVE_AREA,
    MIN_AREA_VS_LARGEST,
    MIN_BOX_SIDE_PX,
    VEHICLE_CLASS_IDS,
    YOLO_SEG_WEIGHTS_PATH,
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
    path = Path(weights_path) if weights_path else YOLO_SEG_WEIGHTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return YOLO(str(path))

    # First run may download the weights; keep a copy in models/ for the live test.
    model = YOLO(path.name)
    downloaded = Path(getattr(model, "ckpt_path", "") or path.name)
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


def _crop_mask(polygon, box: tuple[int, int, int, int]) -> np.ndarray | None:
    """Rasterise a car outline (image coords) into a boolean mask of the crop."""

    x1, y1, x2, y2 = box
    if polygon is None or len(polygon) < 3:
        return None
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    points = np.round(np.asarray(polygon) - (x1, y1)).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    if not mask.any():
        return None
    return mask.astype(bool)


def detect_cars(image: Image.Image, detector) -> list[dict]:
    """Return kept vehicles: box (xyxy pixels), confidence and, with a
    segmentation detector, ``mask`` - a boolean array the size of the crop
    that is True on this car's own pixels (``None`` for box-only detectors).
    """

    image_width, image_height = image.size
    results = detector.predict(
        # Pass the PIL image itself: ultralytics treats a NumPy array as BGR,
        # so np.array(rgb_image) showed the detector a red/blue-swapped photo
        # and it missed or under-scored cars.
        source=image.convert("RGB"),
        conf=DETECT_CONFIDENCE,
        iou=DETECT_NMS_IOU,
        classes=VEHICLE_CLASS_IDS,
        # One vehicle is often boxed as both "car" and "truck"; merge across
        # classes so it is reported once, not twice.
        agnostic_nms=True,
        verbose=False,
    )

    cars = []
    if not results:
        return cars

    boxes = results[0].boxes
    if boxes is None:
        return cars

    masks = results[0].masks
    polygons = masks.xy if masks is not None else [None] * len(boxes)

    for box, polygon in zip(boxes, polygons):
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
                "mask": _crop_mask(polygon, (x1, y1, x2, y2)),
            }
        )

    if cars:
        def area(item):
            x1, y1, x2, y2 = item["box"]
            return (x2 - x1) * (y2 - y1)

        largest = max(area(item) for item in cars)
        cars = [item for item in cars if area(item) >= MIN_AREA_VS_LARGEST * largest]
    cars.sort(key=lambda item: item["detect_confidence"], reverse=True)
    return cars


def crop_box(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = box
    return image.crop((x1, y1, x2, y2))
