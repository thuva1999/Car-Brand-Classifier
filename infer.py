"""Live-test command: classify every car in a phone photo.

Examples:
    python infer.py path/to/photo.jpg
    python infer.py path/to/photo.jpg --save-annotated
    python infer.py --camera 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2  # noqa: E402
from PIL import Image  # noqa: E402

from pipeline import (  # noqa: E402
    CarPipeline,
    annotate_image,
    format_answer_line,
    resolve_image_path,
)


def print_predictions(image_name: str, predictions: list[dict]) -> None:
    if not predictions:
        print("No cars detected.")
        return

    for item in predictions:
        print(f"{item['brand']} — {item['colour']}")

def run_image(pipeline: CarPipeline, image_path: Path, save_annotated: bool) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    predictions = pipeline.predict_image(image)
    print_predictions(image_path.name, predictions)

    if save_annotated:
        output_path = PROJECT_DIR / "outputs" / f"{image_path.stem}_pred.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotate_image(image, predictions).save(output_path)
        print(f"Annotated image saved to: {output_path}")


def run_camera(pipeline: CarPipeline, camera_index: int, save_annotated: bool) -> None:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera_index}. "
            "On Windows try --camera 0 or 1 after connecting the phone as a webcam."
        )

    print("Camera ready. SPACE = classify this frame. Q = quit.")
    window_name = "Tashkent car classifier"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Camera frame not available.")
            break

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if key == 32:  # space
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            predictions = pipeline.predict_image(image)
            print_predictions("camera", predictions)
            annotated = annotate_image(image, predictions)
            cv2.imshow(window_name, cv2.cvtColor(np_from_pil(annotated), cv2.COLOR_RGB2BGR))
            if save_annotated:
                output_path = PROJECT_DIR / "outputs" / "camera_pred.jpg"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                annotated.save(output_path)
                print(f"Annotated image saved to: {output_path}")

    capture.release()
    cv2.destroyAllWindows()


def np_from_pil(image: Image.Image):
    import numpy as np

    return np.array(image.convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect cars and predict brand + colour locally."
    )
    parser.add_argument("image", nargs="?", help="Path to a phone photo.")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Open this camera index instead of a file (phone-as-webcam).",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save a boxed preview under outputs/.",
    )
    args = parser.parse_args()

    if args.image is None and args.camera is None:
        parser.error("Provide an image path or --camera 0")

    print("Loading models (offline after the first YOLO download)...")
    pipeline = CarPipeline()
    print(
        f"Brand classes ({len(pipeline.brand_names)}): "
        + ", ".join(pipeline.brand_names)
    )
    colour_source = "trained classifier" if pipeline.color_model is not None else "HSV on car crops"
    print(f"Colour method: {colour_source}")

    if args.camera is not None:
        run_camera(pipeline, args.camera, args.save_annotated)
        return

    run_image(pipeline, resolve_image_path(args.image), args.save_annotated)


if __name__ == "__main__":
    main()
