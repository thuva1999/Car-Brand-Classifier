# Tashkent car brand and colour classifier

CSAI-3210 midterm pipeline. A phone photo of a Tashkent street goes in. The laptop, **offline**, prints one `(brand, colour)` pair per classifiable car.

Architecture:

1. **YOLO11n-seg** pretrained on COCO finds cars and trucks and outlines each one (detector only). Overlapping "car" and "truck" boxes for the same vehicle are merged. If `models/yolo11n-seg.pt` is missing, the first run downloads it (needs internet).
2. Each crop goes through a **brand ensemble** of two models, **25 brands**: a frozen **DINOv2 ViT-S/14** with a trained linear head (`models/car_brand_classifier_candidate_v5.pth`) and **MobileNetV3-Small** fine-tuned on our brand folders (`models/car_brand_classifier_candidate_v6.pth`). Their brand probabilities are averaged 60% DINOv2 / 40% MobileNet.
3. Colour uses **HSV rules** on the car's own pixels (its segmentation outline, door/bonnet band). No colour model has been trained yet: `data/color_dataset` is empty. The code would use a trained colour model only if `models/car_color_classifier.pth` existed and was at least 0.70 confident.
4. Small or low-confidence boxes are dropped. Every prediction prints its brand, colour and detection confidence.

## Live test

No internet. One photo, under two minutes.

```bat
cd "Car Brand Classifier"
venv\Scripts\activate
python infer.py path\to\photo.jpg
```

The default brand model is the DINOv2 + MobileNet ensemble (`candidate_v5` + `candidate_v6`, 25 brands), set by `BRAND_MODEL_PATH` and `BRAND_ENSEMBLE_WEIGHTS` in `src/config.py`. Set `BRAND_MODEL_PATH` to a single checkpoint to run that model alone.

Example output (`data\tashkent_test\IMG_20261023_164620.jpg`):

```text
Car 1: Chevrolet (71%) — grey (68%)   [detected 92%]
Car 2: Lexus (21%) — white (79%)   [detected 91%]
Car 3: Hyundai (35%) — black (72%)   [detected 90%]
Car 4: Chevrolet (18%) — white (74%)   [detected 76%]
```

Brand % is the model's softmax probability. Colour % says how clearly the HSV rule decided (50% = right on a threshold, capped at 95%; not a calibrated probability).

To select a specific brand checkpoint, or several to run as an equal-weight ensemble:

```bat
python infer.py path\to\photo.jpg --brand-model models\car_brand_classifier_candidate_v6.pth
python infer.py path\to\photo.jpg --brand-model models\car_brand_classifier_candidate_v5.pth models\car_brand_classifier_candidate_v6.pth
```

For a single image that is already cropped to one car, skip YOLO explicitly:

```bat
python infer.py data\brand_dataset\BYD\image.png --crop
```

Normal street photos should not use `--crop`; YOLO is required to find and crop each vehicle.

You can also pass a photo's filename by itself if it is stored somewhere under
`data\`; if that filename occurs more than once, pass the full path instead.

### Phone camera → laptop (recommended on test day)

`serve.py` serves a small web page **from the laptop** on the local network. The phone only takes the photo and shows the answer; YOLO, the brand model and the colour rules all run on the laptop. No internet and no cloud service is used.

1. Turn on the **phone's hotspot** and connect the laptop to it (no mobile data needed; the classroom Wi-Fi may block phone↔laptop traffic).
2. On the laptop:

   ```bat
   venv\Scripts\activate
   python serve.py
   ```

   Wait for `Open one of these on the phone`, e.g. `http://192.168.43.10:8000`.
3. Open that address in the phone's browser and tap **Take photo**. The phone's own camera opens; take the photo of the screen.
4. The answer appears on the phone and in the laptop terminal within a few seconds, e.g. `Chevrolet white; Kia grey`, with each car's confidence and the photo with boxes. Write the answer line on the sheet.

Every photo and its annotated result are saved in `outputs/live/`.

**First run on a laptop:** Windows asks whether to allow Python on networks. Tick **Private** *and* **Public** (a phone hotspot is often treated as a public network), then **Allow**. If the phone still cannot open the page: Windows Security → Firewall & network protection → Allow an app through firewall → enable `python.exe` for both, or set the hotspot connection to *Private* in Settings → Network. Rehearse this once before test day.

Backup if the hotspot fails: send the photo to yourself in Telegram, save it on the laptop, and run `python infer.py path\to\photo.jpg`.

Phone as webcam (SPACE classifies the frame, Q quits):

```bat
python infer.py --camera 0
```

```text
BYD white; Chevrolet grey
```

`BRAND_UNKNOWN_THRESHOLD` is 0.00, so the model always names its best brand, even at low confidence. Raise it in `src/config.py` to print `unknown` below that confidence.

## Clean laptop setup

Python 3.10–3.12 is safest. This repo has also been used with a local 3.14 CPU Torch install.

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python infer.py data\tashkent_test\IMG_20261023_164608.jpg
```

`models\yolo11n-seg.pt` must be present **before** test day because the classroom has no network (the first detector run downloads it into `models\` automatically). 

Weights:

- `models/car_brand_classifier_candidate_v5.pth` — **brand model, part 1**: DINOv2 ViT-S/14 + linear head (25 brands), trained on `data/split_dataset_candidate_v6` by `training/train_brand_dinov2.py`
- `models/car_brand_classifier_candidate_v6.pth` — **brand model, part 2**: MobileNetV3-Small (25 brands), trained on `data/split_dataset_candidate_v6`
- `models/car_brand_classifier_candidate_v3.pth`, `models/car_brand_classifier_candidate_v2.pth` — previous ensemble (24 brands, trained on `data/split_dataset_candidate_v2`), kept for comparison
- `models/dinov2_vits14_224.pth` — pretrained DINOv2 weights, needed only to train a new DINOv2 head
- `models/car_brand_classifier_candidate_v1.pth` — previous MobileNet model (22 brands), kept for comparison
- `models/yolo11n-seg.pt` — detector with car outlines, required on test day
- `models/color_names.json` — the eight colour labels
- `models/car_color_classifier.pth` — not trained yet; HSV is used without it

Earlier experiment checkpoints (the old v1–v5, `fixed`, `v4_reduced`, the YOLOv8n detector) were removed or replaced; tracked ones remain in git history. The numbering restarted at `candidate_v1` after the dataset was re-cleaned.

## Data

```text
data/brand_dataset/<BrandName>/*.jpg
data/color_dataset/white|black|grey|red|blue|green|brown|yellow/*.jpg
```

Brand folder names must match the badge in English: `Chevrolet`, `Daewoo`, `BYD`, `Jetour`, `Mercedes-Benz`, `Li Auto`, …

Removed brands: Audi, Ford, Jeep, Mitsubishi, Tesla, Toyota and Volkswagen are listed in `EXCLUDED_BRANDS` (`src/config.py`). Their image folders were deleted, `split_dataset.py` skips them, and the active model cannot predict them.

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

Then clean, split and train:

```bat
python training\clean_brand_dataset.py
python training\split_dataset.py
python training\train_brand_classifier.py
python training\train_color_classifier.py
python training\evaluate.py
```

`clean_brand_dataset.py` runs the detector over `data/brand_dataset/` and writes the largest vehicle box of each image to `data/brand_dataset_clean/`, so training sees the same kind of crop as inference. Images where no vehicle covers at least 8% of the frame (and 64 px per side) (interiors, wheels, light close-ups) and byte-identical duplicates are left out; rejected images are copied to `data/brand_dataset_review/rejected_by_yolo/` for a manual check, and `clean_report.csv` lists every decision. The original folders are never changed.

`split_dataset.py` reads the clean folder when it exists and splits by group: all listing photos of the same model-year car, and all crops from the same street photo, land on the same side, so near-duplicates cannot inflate validation accuracy.

`train_brand_classifier.py` pads each crop to a square instead of stretching it, uses AdamW with warmup and cosine decay, label smoothing 0.1, oversampling of small brands capped at 3x, and keeps the checkpoint with the best *balanced* (per-brand average) validation accuracy. `--arch mobilenet_v3_large` or `--arch efficientnet_b0` and `--image-size` select a bigger backbone or input. Each checkpoint stores its architecture, image size and preprocessing, so older stretched-input checkpoints still load correctly. Inference and `evaluate.py` average predictions over the crop and its mirror image.

`split_dataset.py` skips a brand with fewer than 8 images or listed in `EXCLUDED_BRANDS`, and writes the split configured as `SPLIT_BRAND_DIR` (currently `data/split_dataset_candidate_v6/`); it never deletes an existing split. `train_brand_classifier.py` writes the next model to `BRAND_CANDIDATE_MODEL_PATH`, keeping the active model unchanged. `python training\evaluate.py` evaluates the active model on that split and prints validation accuracy, per-brand accuracy, a confusion matrix (also saved under `outputs/`), and a dry run on `data/tashkent_test`.

`training/remove_brands.py` copies a trained checkpoint and removes the output rows of the given brands, so the copy can only choose among the remaining ones. It does not retrain.

### Brand results (active: DINOv2 + MobileNet ensemble, 25 brands)

Retrained on 16 Nov after adding Daewoo, Jetour, Lada, Leapmotor, Changan and Chery photos and re-cleaning the dataset. `candidate_v5` (frozen DINOv2 ViT-S/14 + linear head) and `candidate_v6` (MobileNetV3-Small, 40 epochs) were both trained on `data/split_dataset_candidate_v6` (4,335 train / 1,141 validation, 25 brands including Lada).

| Model | Accuracy | Balanced accuracy |
|---|---|---|
| candidate_v5 (DINOv2) alone | 47.0% | 59.2% |
| candidate_v6 (MobileNet) alone | 53.5% | 57.4% |
| ensemble, equal weights | 56.7% (647/1141) | — |

Comparison with the previous ensemble (`candidate_v3` + `candidate_v2`), both at 60% DINOv2 / 40% MobileNet, on the 400 validation images that were not in the previous training split (741 of the 1,141 were, so the full split would flatter the old models):

| Ensemble | Accuracy | Balanced accuracy |
|---|---|---|
| **candidate_v5 + candidate_v6 (active)** | **209/400 = 52.2%** | **70.9%** |
| candidate_v3 + candidate_v2 (previous) | 190/400 = 47.5% | 54.7% |

Gains on those images: BMW 43→49 of 60, Hyundai 5→9 of 11, Lexus 5→9 of 9, Kia 4→7 of 11, Lada 0→2 of 2 (new class). Changan got worse (28→21 of 101) and stays the weakest brand together with Chery (15 of 55): the two are mostly confused with each other, BYD, Geely and GAC Aion. Per-brand accuracy on the full validation split: Daewoo 100% (9 images), Li Auto 100% (6), Hongqi 96%, Deepal 91%, Jetour 89% (9), BMW 80%, Lexus 79%, Jaguar 76%, Zeekr 75%, Haval 68%, Nissan 67%, Chevrolet 65%, Mercedes-Benz 64%, Lada 60% (5), Leapmotor 59%, JAC 56%, Kia 55%, Hyundai 55%, Geely 54%, Honda 52%, BYD 49%, GAC Aion 41%, Dongfeng 31%, Chery 26%, Changan 25%. Reproduce with `python training\evaluate.py`.

### Earlier brand results (candidate_v3 + candidate_v2, 24 brands)

`candidate_v2` (MobileNetV3-Small, 40 epochs) and `candidate_v3` (frozen DINOv2 ViT-S/14 + linear head) were both trained on `data/split_dataset_candidate_v2` (3,937 train / 1,057 validation). DINOv2 is not fine-tuned: `train_brand_dinov2.py` extracts its features once (clean crop, mirror, one augmented copy; cached in `outputs/`) and trains only the head, because fine-tuning a ViT on a laptop CPU takes hours. The two models make different mistakes, so averaging their probabilities beats both:

| Model | Val acc | Val balanced | 549 images no model trained on: acc | balanced |
|---|---|---|---|---|
| candidate_v2 (MobileNet) alone | 51.5% | 49.0% | 51.0% | 48.6% |
| candidate_v3 (DINOv2) alone | 46.5% | 53.5% | 42.4% | 53.4% |
| ensemble, 40% DINOv2 / 60% MobileNet | 56.1% | 57.1% | 54.8% | 55.3% |
| ensemble, 60% DINOv2 / 40% MobileNet | 55.0% | 57.7% | 51.9% | 57.5% |

Every DINOv2 weight from 0.3 to 0.6 beat both single models on this split. 0.4 scores best on listing photos, but DINOv2 holds up better on phone photos of a screen (the live test): on 7 hand-checked live cars (16 Nov) DINOv2 alone was right on a Leapmotor, a Jetour X70 and a Chery sedan where MobileNet was not, and 0.6 gets 2/7 against 0/7 at 0.4. So 0.6 was chosen, and is kept for the retrained models. Biggest per-brand gains of the ensemble over v2 alone (at 0.5): Lexus 67→80%, Nissan 41→52%, Hyundai 49→66%, BMW 60→68%, Chevrolet 59→67%. Still weak: Changan, Chery, Honda. Reproduce with `python training\evaluate.py`.

**Known data gap:** the Chery training images have **no Tiggo** (they are Fulwin T9, Unbounded Pro, eQ7 and old CompCars photos), but Tiggo 7 Pro / 8 Pro and Arrizo are the Chery cars on Uzbek streets; all four Tiggo/Arrizo cars in the 16 Nov live photos were wrong at every ensemble weight. Changan images are mostly Qiyuan models. Jetour (13 images) and Leapmotor (29) are too small. Add photos of those models to `data/brand_dataset/<Brand>/`, then clean, split and retrain.

### Earlier brand results (candidate_v1, 22 brands)

Dataset: every image re-cleaned with YOLO (3,513 crops kept, 97 exact duplicates and images with no vehicle dropped, misfiled images moved to the right brand). Grouped split: 2,792 train / 717 validation. Brands with fewer than 8 images (Deepal, Lada) are skipped.

| Model | Validation images | Correct | Accuracy |
|---|---|---|---|
| **candidate_v1 (active)** | full v1 validation split | 424/717 | **59.1%** |
| **candidate_v1 (active)** | 173 images neither model trained on | 121/173 | **69.9%** |
| previous model (v4_reduced) | same 173 images | 86/173 | 49.7% |

The previous model was trained on an older split, so most of the new validation images were in its training data; only the 173 images unseen by both models give a fair comparison. On those, Chevrolet improved from 32% to 73% after adding Uzbek-market models (Onix, Cobalt, Gentra, Nexia, Spark, Tracker, Damas), and Dongfeng (new, 61 images) reached 67%.

Per-brand accuracy on the full validation split: Hongqi 100%, GAC Aion 92%, Li Auto 83%, Zeekr 75%, Lexus 73%, BMW 72%, Haval 71%, Mercedes-Benz 70%, Dongfeng 67%, Chevrolet 60%, Jaguar 59%, Leapmotor 57%, Honda 53%, Kia 53%, Chery 50%, JAC 50%, Nissan 46%, BYD 45%, Hyundai 45%, Geely 39%, Changan 25%, Jetour 0% (3 validation images, 13 training). Small brands have only 3–20 validation images, so their percentages are rough. The confusion matrix is written to `outputs/brand_confusion_matrix_car_brand_classifier_candidate_v1.csv` by `python training\evaluate.py`. The validation images are mostly listing and dealer photos, not Tashkent street photos, and `data/tashkent_test` still has no ground-truth labels, so live-test accuracy is not yet measured.

The training and inference brand pipelines share their resize, RGB tensor conversion, and ImageNet normalization. Class order is stored in each checkpoint and loaded from that checkpoint. Validation images are scored by normalized class name, not by assuming the folder indices match the model's output indices.

The street-photo dry run is not an accuracy measurement: `data/tashkent_test` currently has no per-car ground-truth labels. For domain-matched training, run `crop_street_photos.py`, manually identify every crop, then put labelled crops into `data/brand_dataset/<Brand>/`. Never train on unlabelled crops. Report the source of every imported dataset. The brand unknown threshold is 0.00, so a brand is always guessed. Detector confidence remains at 0.35 because the test set has no box annotations to verify whether lowering it adds true cars or false/duplicate detections.

## Colour labels and HSV method

Keep the eight exam labels in `src/config.py`: white, black, grey, red, blue, green, brown, yellow. No colour model has been trained (`data/color_dataset` is empty), so every colour currently comes from the HSV rules in `src/color_model.py`.

Pixels used: the car's YOLO segmentation outline (slightly eroded so road and neighbouring cars do not leak in), restricted to the door/bonnet band (30–70% of crop height, 15–85% of width). Falls back to the band alone if the outline is missing or too small.

Rules, tuned on Tashkent street photos where the phone camera adds a bluish cast (grey, white and black paint, and sky reflections, show up as hue ~104 with saturation 30–60):

1. **Red / green / yellow / blue** win when well-lit, saturated pixels cover at least 15% of the paint. Blue needs saturation ≥ 100 because the cast is blue; red and green need ≥ 60; yellow needs ≥ 140 so beige is not called yellow. Blue does not count on a dark body (darkest quarter of the paint pixels below brightness 80) unless the blue is vivid (saturation ≥ 185): that is black paint under the screen's blue cast. Red does not count when the red pixels are very bright (≥ 180): lit tail lights on a rear view.
2. **Brown** (also beige/gold): warm hue with saturation ≥ 30 covering at least 25%, and at least 15 points more than the photo's background (outside the car boxes). If the road and buildings are just as warm, the tint comes from the light (golden hour, warm camera setting), not the paint.
3. Otherwise **black / grey / white** by brightness: median below 65 is black. Glossy black photographed off a screen is covered in bright reflections (trees, sky, watermarks) that lift the median up to ~125, but it keeps a dark core, so darkest quarter ≤ 56 with median ≤ 125 is black too. A dark (median < 110), strongly saturated (≥ 75) body such as greenish black is black. Median 175 and above is white; white paint in shade (150–175) is told apart from silver by its lower saturation.

### Colour accuracy on live-test photos

`outputs/live/` holds every photo taken through `serve.py`, i.e. phone photos of a screen, like the live test. `python training\make_colour_labelling_sheets.py` crops every car into numbered sheets plus `outputs/colour_labelling/cars.csv`; fill its `true_colour` column, then `python training\evaluate_colour.py` re-scores the current rules on exactly those pixels and lists every change and miss.

106 cars were labelled on 16 Nov (48 white, 27 black, 17 grey, 9 blue, 3 brown, 1 red, 1 yellow). The rules above went from **76/106 (72%) to 90/106 (85%)**; black from 7/27 to 22/27, white 46/48, grey 10/17, blue 8/9. The new cut-offs were set on these same cars (about 37 distinct cars photographed repeatedly), so expect somewhat lower on new photos. A learned colour classifier on DINOv2 features was tried and scored only ~45% in grouped cross-validation: too few distinct cars.

Remaining misses: black cars washed out by the screen with no dark pixels left (look grey), the rear of a black car with tail lights (grey), grey vs white in the overlapping 160–205 brightness range, silver with a strong blue cast (blue), pale blue (white), beige with a pink tint (red). `outputs/` is not in git: back up `outputs/colour_labelling/` to keep the labels.

## Design choices you must be able to explain

- **Why YOLO + classifier, not one whole-image network?** Test photos have one or many cars. Extra leftover predictions cost 1 point, so we filter by box size and detector confidence instead of guessing a car in every frame.
- **Why MobileNetV3-Small?** Small enough for a CPU laptop, ImageNet pretrained, and already our brand backbone. Features are frozen for a few epochs, then unfrozen at a lower learning rate.
- **Why no `unknown`?** A wrong brand and `unknown` both score 0 for brand, and colour still scores either way, so always guessing can only gain points. The threshold can be raised in `src/config.py` if a confident wrong badge is penalised.
- **Why HSV colour until crops are labelled?** The brief allows a non-learned colour method if we describe it. HSV on the **crop** is much less noisy than HSV on the whole street. Replace it by training `train_color_classifier.py` once `data/color_dataset` is filled.

## Repository layout

```text
infer.py                     live-test entry point (photo file or webcam)
serve.py                     phone-camera page served from the laptop (local network only)
src/config.py                paths and thresholds
src/detect.py                YOLO11n-seg car finder and outlines (YOLO11n / YOLOv8n fallback)
src/brand_model.py           MobileNetV3-Small brand head
src/color_model.py           colour classifier or HSV
src/pipeline.py              ties the three stages together
training/clean_brand_dataset.py
training/split_dataset.py
training/train_brand_classifier.py
training/train_color_classifier.py
training/evaluate.py
training/remove_brands.py
training/crop_street_photos.py
models/                      trained weights
data/brand_dataset/          labelled brand crops
```

Stanford Cars download (optional supplement): `python src/prepare_dataset.py`
