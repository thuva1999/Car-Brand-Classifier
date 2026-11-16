"""Shared paths and live-test thresholds for the Tashkent car pipeline."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"

BRAND_DATASET_DIR = DATA_DIR / "brand_dataset"
# YOLO-cropped, filtered copy of BRAND_DATASET_DIR (training/clean_brand_dataset.py).
BRAND_CLEAN_DATASET_DIR = DATA_DIR / "brand_dataset_clean"
BRAND_REJECTED_DIR = DATA_DIR / "brand_dataset_review" / "rejected_by_yolo"
COLOR_DATASET_DIR = DATA_DIR / "color_dataset"
SPLIT_BRAND_CANDIDATE_DIR = DATA_DIR / "split_dataset_candidate_v2"
SPLIT_BRAND_DIR = SPLIT_BRAND_CANDIDATE_DIR
SPLIT_COLOR_CANDIDATE_DIR = DATA_DIR / "split_color_candidate"
SPLIT_COLOR_DIR = SPLIT_COLOR_CANDIDATE_DIR
STREET_PHOTO_DIRS = [
    DATA_DIR / "tashkent_test",
    DATA_DIR / "raw" / "Tashkent" / "images",
    DATA_DIR / "raw" / "Tashkent",
]
LABELS_CSV = DATA_DIR / "labels" / "cars.csv"

# Active brand model: candidate_v2, 24 brands (adds Deepal, Daewoo), trained on
# split_dataset_candidate_v2 (modern BYD/Changan/Geely/GAC/Chery/Deepal/Hongqi
# gallery photos added). 52.4% (balanced 50.0%) on its own 1,077-image val.
# Same-image comparison with candidate_v1:
#   new val images neither model saw: 51.3% vs 8.0%
#   phone photos of screens (24 cars): 12/24 vs 5/24
#   older brands, val for both models: 54.1% vs 62.3%  (BMW/Kia/Hyundai/Lexus lost)
#
# Active brand model: ensemble of candidate_v3 (frozen DINOv2 ViT-S/14 + linear
# head, training/train_brand_dinov2.py) and candidate_v2 (MobileNetV3-Small).
# Both run on every car; brand probabilities are averaged 40% DINOv2 / 60%
# MobileNet. On split_dataset_candidate_v2 val (1,057 images):
#   v2 alone 51.5% (balanced 49.0%), v3 alone 46.5% (53.5%),
#   ensemble 56.1% (57.1%). On the 549 val images no model trained on:
#   v2 51.0% (48.6%), v3 42.4% (53.4%), ensemble 54.8% (55.3%).
# Any DINOv2 weight from 0.3 to 0.6 beat both single models.
# A list of checkpoints runs as an ensemble with BRAND_ENSEMBLE_WEIGHTS (same
# order); a single path runs that model alone.
BRAND_MODEL_PATH = [
    MODELS_DIR / "car_brand_classifier_candidate_v3.pth",
    MODELS_DIR / "car_brand_classifier_candidate_v2.pth",
]
BRAND_ENSEMBLE_WEIGHTS = [0.4, 0.6]
# Where the next training run saves (kept separate from the active model).
BRAND_CANDIDATE_MODEL_PATH = MODELS_DIR / "car_brand_classifier_candidate_v4.pth"
# Pretrained DINOv2 ViT-S/14 weights (timm format) for the "dinov2_vits14" backbone.
DINOV2_WEIGHTS_PATH = MODELS_DIR / "dinov2_vits14_224.pth"
COLOR_MODEL_PATH = MODELS_DIR / "car_color_classifier.pth"
COLOR_CANDIDATE_MODEL_PATH = MODELS_DIR / "car_color_classifier_candidate.pth"
COLOR_NAMES_PATH = MODELS_DIR / "color_names.json"
# YOLO11n-seg detector: car boxes plus a per-car outline, so colour is read
# only from that car's own pixels. Must be in models/ before the offline test.
YOLO_SEG_WEIGHTS_PATH = MODELS_DIR / "yolo11n-seg.pt"

# Brands removed from the project: never split, trained or predicted. Their
# image folders were deleted from data/brand_dataset (still in git history).
EXCLUDED_BRANDS = [
    "Audi", "Ford", "Jeep", "Mitsubishi", "Tesla", "Toyota", "Volkswagen",
]

IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# COCO ids: 2 = car, 7 = truck (some SUVs are labelled truck).
VEHICLE_CLASS_IDS = [2, 7]

# Tuned to prefer missing a dubious car over inventing one.
# Extra leftover predictions cost 1 live-test point each.
DETECT_CONFIDENCE = 0.35
MIN_BOX_RELATIVE_AREA = 0.008
MIN_BOX_SIDE_PX = 40
# Drop boxes smaller than this share of the biggest car in the photo: on the
# 14 Nov phone tests these were background cars and web thumbnails (1-6% of
# the frame), while real second cars were 35-40% of the main car.
MIN_AREA_VS_LARGEST = 0.20
BRAND_UNKNOWN_THRESHOLD = 0.00
COLOR_MODEL_CONFIDENCE = 0.70

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

# Map common paint names to the exam's eight colour labels.
COLOUR_ALIASES = {
    "maroon": "red",
    "orange": "yellow",
    "silver": "grey",
    "beige": "brown",
    "gold": "brown",
    "purple": "blue",
}
