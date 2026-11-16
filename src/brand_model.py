"""Load and run the team's brand classifier (MobileNetV3 / EfficientNet / DINOv2 backbones)."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from config import (
    BRAND_ENSEMBLE_WEIGHTS,
    BRAND_MODEL_PATH,
    BRAND_UNKNOWN_THRESHOLD,
    DINOV2_WEIGHTS_PATH,
    IMAGE_SIZE,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# "stretch": old checkpoints were trained on crops squashed to a square.
# "pad": new checkpoints keep the car's aspect ratio and pad to a square.
PREPROCESS_STRETCH = "stretch"
PREPROCESS_PAD = "pad"

SUPPORTED_ARCHITECTURES = (
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "dinov2_vits14",
)
DINOV2_ARCH = "dinov2_vits14"
DINOV2_TIMM_NAME = "vit_small_patch14_dinov2"


class DinoBrandModel(nn.Module):
    """DINOv2 ViT-S/14 backbone + linear head on [CLS token, mean patch token]."""

    def __init__(self, number_of_classes: int, image_size: int = 224) -> None:
        super().__init__()
        import timm  # only needed for this backbone

        self.backbone = timm.create_model(
            DINOV2_TIMM_NAME, pretrained=False, num_classes=0, img_size=image_size
        )
        self.feature_dim = self.backbone.embed_dim * 2
        self.classifier = nn.Sequential(nn.Linear(self.feature_dim, number_of_classes))

    def load_pretrained_backbone(self, weights_path: Path = DINOV2_WEIGHTS_PATH) -> None:
        from timm.layers import resample_abs_pos_embed

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        grid = self.backbone.patch_embed.grid_size
        if state_dict["pos_embed"].shape != self.backbone.pos_embed.shape:
            state_dict["pos_embed"] = resample_abs_pos_embed(
                state_dict["pos_embed"], new_size=grid, num_prefix_tokens=1
            )
        self.backbone.load_state_dict(state_dict)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(images)
        return torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


class SquarePad:
    """Pad a PIL image to a square with a neutral grey border (keeps aspect ratio)."""

    def __init__(self, fill: tuple[int, int, int] = (124, 116, 104)) -> None:
        # ImageNet mean in 0-255, so the pad normalises to roughly zero.
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        if width == height:
            return image
        canvas = Image.new("RGB", (side, side), self.fill)
        canvas.paste(image, ((side - width) // 2, (side - height) // 2))
        return canvas

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fill={self.fill})"


def _classifier_head(model: nn.Module, architecture: str) -> tuple[nn.Sequential, int]:
    if architecture.startswith("mobilenet_v3"):
        return model.classifier, 3
    if architecture.startswith("efficientnet"):
        return model.classifier, 1
    raise ValueError(f"Unsupported brand architecture: {architecture}")


def build_brand_model(
    number_of_classes: int,
    architecture: str = "mobilenet_v3_small",
    pretrained: bool = False,
    image_size: int = 224,
) -> nn.Module:
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported brand architecture {architecture!r}; "
            f"choose one of {SUPPORTED_ARCHITECTURES}"
        )
    if architecture == DINOV2_ARCH:
        model = DinoBrandModel(number_of_classes, image_size=image_size)
        if pretrained:
            model.load_pretrained_backbone()
        return model
    weights = "DEFAULT" if pretrained else None
    model = getattr(models, architecture)(weights=weights)
    head, index = _classifier_head(model, architecture)
    head[index] = nn.Linear(head[index].in_features, number_of_classes)
    return model


def brand_transform(
    training: bool = False,
    image_size: int = IMAGE_SIZE,
    preprocess: str = PREPROCESS_STRETCH,
) -> transforms.Compose:
    operations: list = []
    if preprocess == PREPROCESS_PAD:
        operations.append(SquarePad())
    elif preprocess != PREPROCESS_STRETCH:
        raise ValueError(f"Unknown brand preprocess mode: {preprocess!r}")

    if training:
        operations.extend(
            [
                transforms.Resize((image_size + image_size // 8,) * 2),
                transforms.RandomResizedCrop(
                    image_size, scale=(0.65, 1.0), ratio=(0.85, 1.18)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomPerspective(0.2, p=1.0)], p=0.3),
                transforms.RandomRotation(8),
                transforms.ColorJitter(
                    brightness=0.35,
                    contrast=0.35,
                    saturation=0.3,
                    hue=0.03,
                ),
                # Phone photos of a projector are often soft: blur some samples.
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 2.0))], p=0.25
                ),
                transforms.RandomGrayscale(p=0.05),
            ]
        )
    else:
        operations.append(transforms.Resize((image_size, image_size)))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    if training:
        operations.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)))
    return transforms.Compose(operations)


def brand_probabilities(model: nn.Module, images: torch.Tensor, tta: bool = True) -> torch.Tensor:
    """Softmax probabilities, optionally averaged with the mirrored image."""

    probabilities = torch.softmax(model(images), dim=1)
    if tta:
        mirrored = torch.flip(images, dims=[3])
        probabilities = (probabilities + torch.softmax(model(mirrored), dim=1)) / 2
    return probabilities


def _output_features(state_dict: dict) -> int:
    linear_weights = [
        key for key, value in state_dict.items()
        if key.startswith("classifier.") and key.endswith(".weight") and value.ndim == 2
    ]
    if not linear_weights:
        raise ValueError("Brand checkpoint has no classifier weights.")
    last = max(linear_weights, key=lambda key: int(key.split(".")[1]))
    return state_dict[last].shape[0]


def _normalize_label(label: str) -> str:
    return re.sub(r"[\s-]+", "", label).casefold()


class BrandEnsemble:
    """Several brand checkpoints that vote together on every crop.

    Each member keeps its own preprocessing (image size, pad/stretch), so the
    ensemble works on PIL crops rather than on one shared tensor. Members'
    probabilities (each flip-averaged) are combined with the given weights
    over the union of their brand names; a brand a member does not know gets
    zero from that member.
    """

    def __init__(self, members: list[tuple], weights: list[float], paths: list[Path]) -> None:
        if len(weights) != len(members):
            raise ValueError("Need one ensemble weight per brand model.")
        total = float(sum(weights))
        self.weights = [float(weight) / total for weight in weights]
        self.members = members  # (model, class_names, transform)
        self.paths = paths
        self.class_names: list[str] = []
        seen: dict[str, int] = {}
        for _, names, _ in members:
            for name in names:
                key = _normalize_label(name)
                if key not in seen:
                    seen[key] = len(self.class_names)
                    self.class_names.append(name)
        # For each member: the ensemble index of each of its output columns.
        self.index_maps = [
            torch.tensor([seen[_normalize_label(name)] for name in names])
            for _, names, _ in members
        ]

    def eval(self) -> "BrandEnsemble":
        for model, _, _ in self.members:
            model.eval()
        return self

    @torch.no_grad()
    def probabilities_for_crops(
        self, crops: list[Image.Image], device: torch.device, tta: bool = True
    ) -> torch.Tensor:
        combined = torch.zeros(len(crops), len(self.class_names))
        for (model, _, transform), index_map, weight in zip(
            self.members, self.index_maps, self.weights
        ):
            images = torch.stack([transform(crop.convert("RGB")) for crop in crops]).to(device)
            probabilities = brand_probabilities(model, images, tta=tta).cpu()
            combined[:, index_map] += weight * probabilities
        return combined

    def __repr__(self) -> str:
        parts = ", ".join(f"{p.name} x{w:.2f}" for p, w in zip(self.paths, self.weights))
        return f"BrandEnsemble({parts})"


def load_brand_model(
    model_path: Path | list[Path] | tuple[Path, ...] | None = None,
    device: torch.device | None = None,
    weights: list[float] | None = None,
):
    """Load one brand checkpoint, or a BrandEnsemble when given several paths.

    Returns (model, class_names, image_transform, device). For an ensemble the
    transform is None: each member applies its own, via predict_brand.
    """
    if model_path is None:
        model_path = BRAND_MODEL_PATH
    if isinstance(model_path, (list, tuple)):
        paths = [Path(path) for path in model_path]
        if len(paths) == 1:
            return _load_single_brand_model(paths[0], device)
        if weights is None:
            configured = BRAND_MODEL_PATH if isinstance(BRAND_MODEL_PATH, (list, tuple)) else []
            same_as_config = [Path(p) for p in configured] == paths
            weights = list(BRAND_ENSEMBLE_WEIGHTS) if same_as_config else [1.0] * len(paths)
        members = []
        for path in paths:
            model, names, transform, device = _load_single_brand_model(path, device)
            members.append((model, names, transform))
        ensemble = BrandEnsemble(members, weights, paths)
        return ensemble, ensemble.class_names, None, device
    return _load_single_brand_model(Path(model_path), device)


def _load_single_brand_model(path: Path, device: torch.device | None = None):
    if not path.exists():
        raise FileNotFoundError(f"Brand model not found: {path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        class_names = checkpoint.get("class_names")
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint),
        )
        architecture = checkpoint.get("architecture", "mobilenet_v3_small")
        image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
        preprocess = checkpoint.get("preprocess", PREPROCESS_STRETCH)
    else:
        class_names = None
        state_dict = checkpoint
        architecture = "mobilenet_v3_small"
        image_size = IMAGE_SIZE
        preprocess = PREPROCESS_STRETCH

    if not isinstance(class_names, list) or not class_names:
        raise ValueError(
            f"Brand checkpoint must contain its ordered 'class_names': {path}"
        )
    output_features = _output_features(state_dict)
    if output_features != len(class_names):
        raise ValueError(
            f"Brand checkpoint has {output_features} output classes but "
            f"{len(class_names)} class names: {path}"
        )

    model = build_brand_model(
        len(class_names), architecture=architecture, image_size=image_size
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    image_transform = brand_transform(image_size=image_size, preprocess=preprocess)
    return model, class_names, image_transform, device


def predict_brand(
    crop: Image.Image,
    model,
    class_names: list[str],
    image_transform,
    device,
    unknown_threshold: float = BRAND_UNKNOWN_THRESHOLD,
    tta: bool = True,
) -> tuple[str, float]:
    with torch.no_grad():
        if isinstance(model, BrandEnsemble):
            probabilities = model.probabilities_for_crops([crop], device, tta=tta)[0]
        else:
            image_tensor = image_transform(crop.convert("RGB")).unsqueeze(0).to(device)
            probabilities = brand_probabilities(model, image_tensor, tta=tta)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)

    confidence_value = float(confidence.item())
    brand = class_names[int(predicted_index.item())]
    if confidence_value < unknown_threshold:
        return "unknown", confidence_value
    return brand, confidence_value
