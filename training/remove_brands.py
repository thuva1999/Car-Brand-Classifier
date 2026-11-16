"""Make a copy of a trained brand checkpoint that can no longer predict some brands.

Removes the rows for the given brands from the final classifier layer, so the
copy chooses only among the remaining brands. Every other weight is unchanged,
so this is not retraining: the remaining brands keep exactly the scores they
had, and images of a removed brand get the closest remaining brand instead.
The source checkpoint is never modified.

Examples:
    python training/remove_brands.py models/car_brand_classifier_candidate_v4.pth
    python training/remove_brands.py models/a.pth --brands Audi Ford --output models/b.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from config import EXCLUDED_BRANDS  # noqa: E402


def remove_brands(source: Path, output: Path, brands: list[str]) -> list[str]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "class_names" not in checkpoint:
        raise ValueError(f"{source} has no ordered 'class_names'; cannot remove brands safely.")

    state_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
    state_dict = dict(checkpoint[state_key])
    class_names = list(checkpoint["class_names"])

    linear_weights = [
        key for key, value in state_dict.items()
        if key.startswith("classifier.") and key.endswith(".weight") and value.ndim == 2
    ]
    weight_key = max(linear_weights, key=lambda key: int(key.split(".")[1]))
    bias_key = weight_key[: -len("weight")] + "bias"

    keep = [index for index, name in enumerate(class_names) if name not in brands]
    removed = [name for name in class_names if name in brands]
    if len(keep) < 2:
        raise ValueError("Fewer than two brands would remain.")

    state_dict[weight_key] = state_dict[weight_key][keep].clone()
    state_dict[bias_key] = state_dict[bias_key][keep].clone()

    new_checkpoint = dict(checkpoint)
    new_checkpoint[state_key] = state_dict
    new_checkpoint["class_names"] = [class_names[index] for index in keep]
    new_checkpoint["removed_brands"] = removed
    new_checkpoint["derived_from"] = Path(source).name
    # Validation scores of the source no longer apply to the reduced model.
    for key in ("val_accuracy", "val_balanced_accuracy"):
        new_checkpoint.pop(key, None)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_checkpoint, output)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("--brands", nargs="+", default=EXCLUDED_BRANDS)
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: <source name>_reduced.pth next to the source.",
    )
    args = parser.parse_args()
    output = args.output or args.source.with_name(f"{args.source.stem}_reduced.pth")
    removed = remove_brands(args.source, output, args.brands)
    print(f"{args.source.name} -> {output.name}: removed {', '.join(removed) or 'nothing'}")


if __name__ == "__main__":
    main()
