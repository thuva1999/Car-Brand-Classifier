# Tashkent car brand and colour classifier

CSAI-3210 midterm pipeline. A phone photo of a Tashkent street goes in. The laptop, **offline**, prints one `(brand, colour)` pair per classifiable car.

Architecture (keep this design; do not share it with other teams):

1. **YOLOv8n** pretrained on COCO finds cars and trucks (detector only).
2. Each crop goes through **MobileNetV3-Small** fine-tuned on our brand folders.
3. Colour is predicted on the same crop: a trained 8-class MobileNetV3-Small when its confidence is at least 0.70; otherwise **HSV voting** on the central car-body region (allowed by the brief).
4. Small or low-confidence boxes are dropped. Low brand confidence becomes `unknown` so colour can still score.

## Live test (test day)

No internet. Use the `midterm` tag. One photo, under two minutes.

```bat
cd "Car Brand Classifier"
venv\Scripts\activate
python infer.py path\to\photo.jpg
```

The default inference checkpoint is the validated `car_brand_classifier_candidate_v2.pth`. The previous model remains available as `car_brand_classifier.pth`.

To select a specific brand checkpoint:

```bat
python infer.py path\to\photo.jpg --brand-model models\car_brand_classifier.pth
```

For a single image that is already cropped to one car, skip YOLO explicitly:

```bat
python infer.py data\brand_dataset\BYD\image.png --crop
```

Normal street photos should not use `--crop`; YOLO is required to find and crop each vehicle.

You can also pass a photo's filename by itself if it is stored somewhere under
`data\`; if that filename occurs more than once, pass the full path instead.

Phone as webcam (SPACE classifies the frame, Q quits):

```bat
python infer.py --camera 0
```

Write the **Answer sheet** line on the form, for example:

```text
BYD white; Chevrolet grey
```

If the model is not sure of the badge it prints `unknown` plus the colour.

## Clean laptop setup

Python 3.10–3.12 is safest. This repo has also been used with a local 3.14 CPU Torch install.

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python infer.py data\tashkent_test\IMG_20261023_164608.jpg
```

The first detector run may download `yolov8n.pt`. Copy it into `models\yolov8n.pt` **before** test day so the classroom has no network.

Weights:

- `models/car_brand_classifier_candidate_v2.pth` — active brand model
- `models/car_brand_classifier.pth` — previous brand model, retained for comparison
- `models/yolov8n.pt` — required on test day
- `models/car_color_classifier.pth` — optional; HSV is used without it

## Data

Put **cropped cars** (or nearly full-frame cars) here:

```text
data/brand_dataset/<BrandName>/*.jpg
data/color_dataset/white|black|grey|red|blue|green|brown|yellow/*.jpg
```

Brand folder names must match the badge in English: `Chevrolet`, `Daewoo`, `BYD`, `Jetour`, `Mercedes-Benz`, `Li Auto`, …

Chinese-brand folders already exist under `data/brand_dataset/` (`BYD`, `Chery`, `Changan`, `Geely`, `Leapmotor`, …) but they were empty when the pipeline was wired. **Fill them.** Half of the live test is Chinese brands. Photograph dealer streets and new residential car parks, many angles, day and evening, and a few photos of a car on a screen (the live test is a phone pointed at a projector).

Public sets (Stanford Cars, CompCars) are a supplement only — cite them in the report. Do not train only on those.

Street photos for cropping:

```text
data/tashkent_test/     phone photos
data/raw/Tashkent/      extra street photos
```

Label helper — detector crops every car it finds:

```bat
python training\crop_street_photos.py --input data\tashkent_test
```

Crops land in `data/crops/unlabelled/`. Sort them into brand and colour folders.

Then split and train:

```bat
python training\split_dataset.py
python training\train_brand_classifier.py
python training\train_color_classifier.py
python training\evaluate.py
```

`split_dataset.py` skips a brand with fewer than 8 images and writes a separate candidate split under `data/split_dataset_candidate/`; it never deletes an existing split. `train_brand_classifier.py` writes the next model to `models/car_brand_classifier_candidate_v3.pth`, keeping both existing models unchanged. `evaluate.py --model models\car_brand_classifier_candidate_v3.pth` prints validation accuracy, a confusion matrix, and a dry run on `data/tashkent_test`.

The training and inference brand pipelines share their resize, RGB tensor conversion, and ImageNet normalization. Class order is stored in each checkpoint and loaded from that checkpoint. Validation images are scored by normalized class name, not by assuming the folder indices match the model's output indices.

The street-photo dry run is not an accuracy measurement: `data/tashkent_test` currently has no per-car ground-truth labels. For domain-matched training, run `crop_street_photos.py`, manually identify every crop, then put labelled crops into `data/brand_dataset/<Brand>/`. Never train on unlabelled crops. Report the source of every imported dataset. The brand unknown threshold is set to 0.25 based on the existing validation confidence/coverage results; it changes how many guesses are shown, not the classifier's underlying accuracy. Detector confidence remains at 0.35 because the test set has no box annotations to verify whether lowering it adds true cars or false/duplicate detections.

## Colour labels and HSV fallback

Keep the eight exam labels in `src/config.py`: white, black, grey, red, blue, green, brown, yellow. Common paint names map into these scored classes: silver to grey, maroon to red, orange to yellow, and beige/gold to brown. The fallback votes on a central body region, excluding the outer 15% horizontally and the upper window/lower wheel regions. A trained colour model is used only when its confidence is at least 0.70; otherwise the HSV method is used. The threshold and HSV approximation should be described in the report.

## Design choices you must be able to explain

- **Why YOLO + classifier, not one whole-image network?** Test photos have one or many cars. Extra leftover predictions cost 1 point, so we filter by box size and detector confidence instead of guessing a car in every frame.
- **Why MobileNetV3-Small?** Small enough for a CPU laptop, ImageNet pretrained, and already our brand backbone. Features are frozen for a few epochs, then unfrozen at a lower learning rate.
- **Why `unknown`?** A wrong brand scores 0 for brand; `unknown` also scores 0 for brand but still allows the colour share, and it avoids a confident wrong badge.
- **Why HSV colour until crops are labelled?** The brief allows a non-learned colour method if we describe it. HSV on the **crop** is much less noisy than HSV on the whole street. Replace it by training `train_color_classifier.py` once `data/color_dataset` is filled.

## Repository layout

```text
infer.py                     live-test entry point
src/config.py                paths and thresholds
src/detect.py                YOLOv8n car finder
src/brand_model.py           MobileNetV3-Small brand head
src/color_model.py           colour classifier or HSV
src/pipeline.py              ties the three stages together
training/split_dataset.py
training/train_brand_classifier.py
training/train_color_classifier.py
training/evaluate.py
training/crop_street_photos.py
models/                      trained weights
data/brand_dataset/          labelled brand crops
```

Stanford Cars download (optional supplement): `python src/prepare_dataset.py`
