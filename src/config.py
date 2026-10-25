"""Shared paths and live-test thresholds for the Tashkent car pipeline."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"

BRAND_DATASET_DIR = DATA_DIR / "brand_dataset"
COLOR_DATASET_DIR = DATA_DIR / "color_dataset"
SPLIT_BRAND_DIR = DATA_DIR / "split_dataset_new"
SPLIT_COLOR_DIR = DATA_DIR / "split_color"
STREET_PHOTO_DIRS = [
    DATA_DIR / "tashkent_test",
    DATA_DIR / "raw" / "Tashkent" / "images",
    DATA_DIR / "raw" / "Tashkent",
]
LABELS_CSV = DATA_DIR / "labels" / "cars.csv"

BRAND_MODEL_PATH = MODELS_DIR / "car_brand_classifier.pth"
COLOR_MODEL_PATH = MODELS_DIR / "car_color_classifier.pth"
BRAND_NAMES_PATH = MODELS_DIR / "class_names.json"
COLOR_NAMES_PATH = MODELS_DIR / "color_names.json"
YOLO_WEIGHTS_PATH = MODELS_DIR / "yolov8n.pt"

IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# COCO ids: 2 = car, 7 = truck (some SUVs are labelled truck).
VEHICLE_CLASS_IDS = [2, 7]

# Tuned to prefer missing a dubious car over inventing one.
# Extra leftover predictions cost 1 live-test point each.
DETECT_CONFIDENCE = 0.35
MIN_BOX_RELATIVE_AREA = 0.008
MIN_BOX_SIDE_PX = 40
BRAND_UNKNOWN_THRESHOLD = 0.35

COLOUR_NAMES = [
    "white",
    "black",
    "grey",
    "red",
    "blue",
    "green",
    "brown",
    "yellow",
]
