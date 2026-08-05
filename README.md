# Beyond Self-Report: Physiologically-Grounded Employee Wellness Monitoring

Thermal-only employee engagement monitoring from a **single thermal image** — expression
features (CNN) fused with physiological personality-proxy signals (ROI temperature
analysis) — avoiding self-report questionnaires and visible-spectrum surveillance.

## Core design decision

One thermal image per employee per session — no sequence, no multiple shots, no waiting.
The system is designed for seamless integration with existing single-frame biometric
attendance infrastructure (badge-in style capture), not a continuous monitoring feed.

## Architecture

Two stages: (1) pretrain a thermal-pattern feature extractor on a large, cleanly-labeled
proxy task, then (2) freeze it and train a small classifier head on top for the actual
(small, noisily-labeled) engagement task.

**Input**
FLIR E8 thermal capture (USB/WiFi transfer) → `flirimageextractor` extracts the raw
per-pixel temperature array (°C) → normalized to 0-255 grayscale → resized to 48×48 for
the CNN branch. Landmark detection runs on the normalized grayscale image **before** the
48×48 resize — 48×48 is too small for reliable landmark detection.

**Stage 1 — Backbone pretraining on room-temperature condition** (`scripts/train_room_temp_backbone.py`)
Charlotte-ThermalFace was captured at 4 room-temperature conditions, encoded directly in
each filename (`src/data/thermal_dataset.py::parse_room_condition` — the digit right
after the subject number; verified across all 10 subjects, watch for the varying
subject-digit width: S1-S9 are 1 digit, S10 is 2). This gives a large (~10.1k images),
cleanly-labeled task — no face detection needed, the label comes from the filename, not
image content — used to pretrain a ResNet18 (ImageNet-pretrained, first conv layer
averaged down to 1-channel input) via a `FC(512→4)` head. Trained 50 epochs, cosine
schedule with 5-epoch warmup, `WeightedRandomSampler` for the (near-balanced) 4 classes.
**Result: 93.3% test accuracy**, 92-94% per class. Backbone saved without its head to
`checkpoints/room_temp_backbone/backbone_only.pt` — this becomes the frozen feature
extractor for Stage 2. (`src/models/thermal_cnn.py`'s from-scratch CNN was the original
Stage-1 approach before this pivot; it's superseded but left in place, still tested.)

**Stage 2 — Engagement classifier on the frozen backbone** (`train.py`, `src/models/fusion_model.py`)
- **Expression path**: frozen ResNet18 backbone (`load_frozen_backbone`, `requires_grad=False`
  on all params, pinned in `eval()` mode even during training — otherwise BatchNorm would
  keep updating its running stats from new data despite being "frozen") → 512-dim pooled features.
- **Physiological path** (`src/roi/`): MediaPipe/Haar/LBP cascade landmark detection → ROI
  temperature extraction (Nose Tip: landmark 4; Forehead: 10,338,297,332,284; Periorbital:
  33,133,362,263; Upper Lip: 13) → `raw_temperature_vector()`: 5-dim
  `[nose_temp, forehead_temp, periorbital_temp, upper_lip_temp, differential_index]`.
  This is raw, un-thresholded temperatures — **not** the binary N/C proxy (see below).
- **Fusion**: `concat(512 + 5) = 517` → `FC(517→128)` → ReLU → Dropout(0.3) → `FC(128→64)`
  → ReLU → Dropout(0.3) → `FC(64→3)` → `P(Disengaged), P(Burned Out), P(Engaged)`. Returns
  raw logits from `forward()`; `predict_proba()` applies softmax separately at inference
  (avoids double-softmax with `CrossEntropyLoss`).
- Only the classifier head trains (100 epochs max, cosine+warmup, early stopping patience
  15); the backbone never sees a gradient.

**Why raw temperatures, not the binary proxy**: an earlier version concatenated the
2-dim binary `[N_proxy, C_proxy]` into the classifier. But `synthesize_engagement_label()`
derives the training label from that exact same proxy — so the classifier hit ~100%
accuracy by trivially inverting its own label-generation rule (confirmed: val_acc=1.000 by
epoch 4) instead of learning anything from the image. Raw temperatures are less directly
leaky — they're a superset of information the proxy is thresholded *from*, not the literal
label-generating value — but are still correlated with the label, since the label is
itself a threshold function of these same quantities. Worth scrutinizing results with this
in mind rather than treating it as leakage-free. The binary proxy is still computed and
used for label synthesis and the dashboard's personality-aware explanation text — just
never fed to the classifier.

**Engagement labels** (`src/roi/labeling.py::synthesize_engagement_label`) — 3 classes,
grounded in Barrick & Mount (1991) treating Conscientiousness as the primary split:
- `0 = Disengaged` (C=0, any N) — low sustained-attention marker dominates
- `1 = Burned Out` (N=1, C=1) — high stress but still pushing through; distinct,
  high-retention-risk state
- `2 = Engaged` (N=0, C=1) — calm and focused, the target state

**Output**: engagement class, wellness score (0-1, `P(Disengaged)`), N/C proxy values,
personality-aware natural-language explanation, on-device Streamlit dashboard.

## Project layout

```
data/raw/Charlotte-ThermalFace/  extracted dataset: S1-S10 subject folders, N<id>.jpg + R<id>.tiff pairs
data/labels/                     synthesized labels + raw ROI temps cache, population_baseline.json
src/data/                        preprocessing.py (calibration/normalize/resize/augmentation transforms),
                                  thermal_dataset.py (ThermalDataset for engagement, RoomTempDataset for backbone pretraining)
src/roi/                         extraction.py (landmarks+ROI temps), differential_index.py,
                                  labeling.py (thresholds, binary proxy, raw_temperature_vector, label synthesis, baseline)
src/models/                      thermal_cnn.py (superseded from-scratch encoder), resnet_backbone.py (ResNet18 1-channel + frozen loader),
                                  fusion_model.py (classifier head + end-to-end wrapper)
src/training/scheduling.py       shared cosine+warmup LR schedule builder
dashboard/app.py                 Streamlit image-upload dashboard
configs/                         cnn_config.yaml (training hyperparams), fusion_config.yaml (model dims, backbone checkpoint path), roi_thresholds.yaml
scripts/compute_population_baseline.py   builds data/labels/population_baseline.json
scripts/precompute_labels_cache.py       builds data/labels/engagement_labels.json (labels + raw roi_temps, ~15min full pass)
scripts/train_room_temp_backbone.py      Stage 1: pretrain the frozen backbone
train.py                         Stage 2: train the engagement classifier head
inference.py                     single FLIR E8 image -> classification + explanation
checkpoints/, logs/               (gitignored)
docs/paper/                       paper draft + references.bib
tests/                            model shape tests
```

## Hardware / target camera

- Training: NVIDIA GTX 1650 class GPU (4GB VRAM) — sufficient since only the small
  classifier head trains; the frozen ResNet18 backbone only ever does a forward pass.
- Deployment: FLIR E8, single radiometric JPEG per capture, read via `flirimageextractor`
  (also requires [ExifTool](https://exiftool.org/) installed system-wide).

## Dataset

[Charlotte-ThermalFace](https://github.com/TeCSAR-UNCC) — 10,376 individual thermal images
(no sequences), 10 subjects (S1-S10), each a paired `N<id>.jpg` (8-bit preview) +
`R<id>.tiff` (16-bit raw, `°C = raw/100 - 273.15`). No accompanying annotation files ship
with the archive — engagement labels are synthesized via the ROI threshold pipeline; room-
temperature condition labels are decoded directly from filenames (see Stage 1 above).
Split 70% train / 15% val / 15% test.

## Ethics

Employee data stays on-device; the employer only ever sees anonymized aggregate team
wellness scores. No visible-spectrum imagery is captured. Designed for compliance with
India's DPDP Act 2023.

## Status

Stage 1 (backbone pretraining) complete: 93.3% test accuracy on room-temperature-condition
classification. Stage 2 (engagement classifier on the frozen backbone) is wired up; see
git history / conversation log for the latest per-class metrics and confusion matrix.
