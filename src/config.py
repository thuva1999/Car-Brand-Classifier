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
SPLIT_BRAND_CANDIDATE_DIR = DATA_DIR / "split_dataset_candidate_v7"
SPLIT_BRAND_DIR = SPLIT_BRAND_CANDIDATE_DIR
SPLIT_COLOR_CANDIDATE_DIR = DATA_DIR / "split_color_candidate"
SPLIT_COLOR_DIR = SPLIT_COLOR_CANDIDATE_DIR
STREET_PHOTO_DIRS = [
    DATA_DIR / "tashkent_test",
    DATA_DIR / "raw" / "Tashkent" / "images",
    DATA_DIR / "raw" / "Tashkent",
]

# Active brand model (retrained 17 Nov, 25 brands): ensemble of
# candidate_v1_dinov2 (frozen DINOv2 ViT-S/14 + linear head,
# training/train_brand_dinov2.py) and candidate_v1_mobilenet
# (MobileNetV3-Small), both trained on split_dataset_candidate_v7
# (4,308 train / 1,128 val; adds Tashkent-market Geely, GAC Aion, Kia,
# Hyundai and Changan listings, close-up crops removed). Both run on every
# car; brand probabilities are averaged 60% DINOv2 / 40% MobileNet.
# On the v7 val: DINOv2 alone 47.9% (balanced 58.8%), MobileNet alone
# best balanced 51.3%.
# Against the previous ensemble (candidate_v5 + candidate_v6) on live phone
# photos of a screen (never trained on):
#   103 hand-labelled cars (17 Nov):  new 62/103 (60%)  vs  old 60/103 (58%)
#   58 phone-feedback cars:           new 24/58 (41%)   vs  old 18/58 (31%)
# Gains: Geely 0/14 -> 4/14, GAC Aion 0/13 -> 4/13, Kia 5/17 -> 9/17.
# Losses: Chevrolet 20/22 -> 16/22, Dongfeng 8/11 -> 5/11.
# DINOv2 is weighted higher because it holds up better on phone photos of a
# screen (the live test): with the previous models, on 7 hand-checked live
# cars DINOv2 alone was right on Leapmotor, Jetour and a Chery sedan where
# MobileNet was not.
# A list of checkpoints runs as an ensemble with BRAND_ENSEMBLE_WEIGHTS (same
# order); a single path runs that model alone.
BRAND_MODEL_PATH = [
    MODELS_DIR / "car_brand_classifier_candidate_v1_dinov2.pth",
    MODELS_DIR / "car_brand_classifier_candidate_v1_mobilenet.pth",
]
BRAND_ENSEMBLE_WEIGHTS = [0.6, 0.4]
# Where the next training run saves (kept separate from the active model).
BRAND_CANDIDATE_MODEL_PATH = MODELS_DIR / "car_brand_classifier_candidate_v7.pth"
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
# 0.35 dropped a Hyundai Palisade half hidden behind a Mercedes (0.349) and
# a Kia rear view seen through tree glare (0.31). On all 104 live photos
# (17 Nov), 0.25 with NMS IoU 0.5 adds only those two and nothing wrong;
# 0.20 added web-page boxes and duplicates. IoU 0.5 (YOLO default 0.7)
# merges the two overlapping boxes YOLO puts on a partly hidden car.
DETECT_CONFIDENCE = 0.25
DETECT_NMS_IOU = 0.5
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
