# Beyond Self-Report: Physiologically-Grounded Employee Wellness Monitoring

Thermal-only employee wellness monitoring from a **single thermal image** — continuous
Stress Index, Cognitive Load Index, and Wellness Score derived from ambient-normalized ROI
temperature analysis — avoiding self-report questionnaires and visible-spectrum
surveillance.

**v2.0 reframe**: the original design produced a 3-class engagement label (Disengaged /
Burned Out / Engaged) from a fused CNN+ROI classifier. That approach was abandoned in
favor of continuous 0-100 physiological scores with no classification labels — a cleaner
fit for a wellness signal that's genuinely a spectrum, not a category, and one that doesn't
require a labeled training set for the target task itself. See "Status" below for what of
the original Stage 1/Stage 2 pipeline is still live.

## Core design decision

One thermal image per employee per session — no sequence, no multiple shots, no waiting.
The system is designed for seamless integration with existing single-frame biometric
attendance infrastructure (badge-in style capture), not a continuous monitoring feed.

## Architecture (current — v2.0)

**Input**
FLIR E8 thermal capture (USB/WiFi transfer, radiometric JPEG) → `flirimageextractor`
extracts the raw per-pixel temperature array (°C) → normalized to 0-255 grayscale for
landmark detection. A raw 16-bit Charlotte-ThermalFace-style TIFF is also accepted, for
testing without a physical camera.

**Face + ROI detection** (`src/roi/extraction.py`) — a 3-stage cascade (MediaPipe →
Haar → LBP) locates the face and four ROI landmark groups: Nose Tip, Forehead,
Periorbital, Upper Lip.

**Ambient normalization** — rather than comparing against a fixed population baseline,
each ROI temperature is expressed as a delta above *this image's own* background/ambient
temperature (`compute_ambient_temperature`: mean of the padded-bbox-masked non-face
pixels). This was added after validating the pipeline on a real FLIR E8 capture taken in
an air-conditioned room — population-baseline deltas conflated ambient drift with genuine
physiological signal.

**Scoring** (`src/scoring/`) — four independent, deterministic 0-100 scores, each a linear
rescale of a raw physiological quantity against P5/P95 population bounds
(`configs/normalization_bounds.yaml`, fit on the ambient-normalized Charlotte-ThermalFace
population, 6,640 successfully-detected images):

- **Stress Index** (`stress_index.py`) — `raw = abs(differential)`, where
  `differential = nose_delta - periorbital_delta`. Originally blended in `nose_delta`
  directly (`-nose_delta + abs(differential)) / 2`, per Fernández et al. (2024)'s nose-tip
  vasoconstriction claim. **Empirically invalidated and dropped**: on a 16-image
  self-collected FLIR E8 validation set with three self-labeled conditions
  (neutral/engaged/stressed, `scripts/validate_collected_faces.py`), `nose_delta` ranked
  the conditions backwards (the coldest average nose was the *neutral* condition), while
  `abs(differential)` alone cleanly separated `stressed > neutral > engaged` matching the
  labels. The current formula is grounded solely in Gioia et al. (2023, Sensors) —
  differential index as the primary autonomic nervous system marker.
- **Cognitive Load Index** (`cognitive_load_index.py`) — `raw = forehead_delta` directly.
  Forehead thermal elevation under cognitive load, per Frontiers in Psychiatry (2025).
- **Wellness Score** (`wellness_score.py`) — `100 - ((stress_index + (100 -
  cognitive_load_index)) / 2)`. High stress + low cognitive load → 0; low stress + high
  cognitive load → 100; high stress + high cognitive load → 50 ("burned out risk" —
  pushing through despite strain).
- **Measurement Confidence** (`confidence_score.py`) — cosine similarity between the
  current capture's frozen-ResNet18 feature vector and a precomputed population-mean
  feature vector (`data/labels/population_mean_features.npy`, gitignored, rebuilt via
  `scripts/compute_population_mean_features.py`). Not a confidence in the score *values*
  (those are deterministic formulas) — a face/image-quality sanity check.

**Personal-baseline calibration mode** (`scripts/compute_personal_baseline.py`,
`inference.py::run_inference(..., personal_baseline_path=...)`) — population P5/P95 bounds
saturate on real captures of a single subject (most `stress_index` values pinned at
90-100 regardless of condition, in the same 16-image validation run). Personal-baseline
mode re-centers each capture's ambient-normalized deltas against that subject's own
resting-state mean (5+ neutral captures) before scoring. Stress uses the signed quantity
`-relative_differential` (not `abs()` — the relative differential crosses zero in both
directions across real conditions, so taking its magnitude loses the sign that
distinguishes "more relaxed than usual" from "more stressed than usual"). On the same
16-image set, personal-baseline mode gives the cleanest separation of the three conditions
of any mode tried (`stressed`: 89.3 vs. `neutral`: 74.1 vs. `engaged`: 15.6, mean
`stress_index`).

**Backbone** (`src/models/resnet_backbone.py`) — a ResNet18 (ImageNet-pretrained, first
conv layer averaged down to 1-channel input), pretrained on Charlotte-ThermalFace's
room-temperature-condition task (see "Legacy Stage 1/2 pipeline" below). In the v2.0
pipeline it's used frozen, purely as a feature extractor for `measurement_confidence` —
it no longer feeds a classifier head.

**Output**: `stress_index`, `cognitive_load_index`, `wellness_score`,
`measurement_confidence` (all 0-100), raw + ambient-normalized ROI temperatures,
categorical interpretation (`interpretation.stress_level` / `cognitive_state` /
`wellness_flag` / a rule-based `recommendation` string), research grounding citations,
detector used. See `inference.py::run_inference` for the full JSON schema.

## Dashboards

- **`dashboard/index.html`** — static, single-page, no backend. Pure HTML/CSS/JS, Chart.js
  gauges (via CDN), dark glassmorphism design. Shows demo values from a real FLIR3643.jpg
  capture on load; file upload gives a client-side image preview only (no re-scoring,
  since there's no backend). Open directly via `file://` — no server needed.
- **`dashboard/app.py`** — Streamlit dashboard with an actual upload → `run_inference()` →
  live-scored round trip (gauges, ROI tables, confidence bar, interpretation, research
  grounding). Run with `streamlit run dashboard/app.py --server.fileWatcherType none` (the
  file-watcher flag avoids a known Streamlit+PyTorch hang inspecting
  `torch.classes.__path__`).

## Project layout

```
data/raw/Charlotte-ThermalFace/  extracted dataset: S1-S10 subject folders, N<id>.jpg + R<id>.tiff pairs
data/raw/FLIR_E8_collected/      real FLIR E8 captures (gitignored) — FLIR3643.jpg demo capture,
                                  neutral/engaged/stressed/ subfolders (self-collected validation set)
data/labels/                     synthesized labels + raw ROI temps cache, population_baseline.json,
                                  population_mean_features.npy (gitignored), personal_baseline_*.json (gitignored)
src/data/                        preprocessing.py (calibration/normalize/resize/augmentation transforms),
                                  thermal_dataset.py (ThermalDataset, RoomTempDataset for backbone pretraining)
src/roi/                         extraction.py (landmarks + ROI temps + ambient_temperature),
                                  labeling.py (ambient_normalized_roi_temps + legacy proxy/threshold code)
src/scoring/                     stress_index.py, cognitive_load_index.py, wellness_score.py, confidence_score.py
src/models/                      resnet_backbone.py (ResNet18 1-channel + frozen loader);
                                  thermal_cnn.py, fusion_model.py — legacy, superseded (see Status)
src/training/scheduling.py       shared cosine+warmup LR schedule builder
dashboard/index.html             static premium dashboard (no backend)
dashboard/app.py                 Streamlit dashboard (live scoring)
configs/normalization_bounds.yaml   P5/P95 bounds for stress_index & cognitive_load_index (current pipeline)
configs/ambient_thresholds.yaml, roi_thresholds.yaml, fusion_config.yaml, cnn_config.yaml
                                  legacy — from the pre-reframe classification approach, kept for reference
scripts/compute_normalization_bounds.py    refits configs/normalization_bounds.yaml from the Charlotte population
scripts/compute_population_mean_features.py  builds data/labels/population_mean_features.npy
scripts/compute_personal_baseline.py       builds a data/labels/personal_baseline_*.json from resting captures
scripts/validate_collected_faces.py        batch-scores a labeled real-FLIR-E8 folder set, absolute or personal-baseline mode
scripts/precompute_labels_cache.py         builds data/labels/engagement_labels.json (labels + raw roi_temps + normalized_roi)
scripts/train_room_temp_backbone.py        Stage 1: pretrain the frozen backbone
train.py                         legacy Stage 2 (engagement classifier head) — superseded, see Status
inference.py                     single FLIR E8/TIFF image -> full scoring JSON (see Architecture above)
checkpoints/, logs/               (gitignored)
docs/paper/                       paper draft + references.bib
tests/                            model shape tests (legacy fusion model — see Status)
```

## Hardware / target camera

- Training: NVIDIA GTX 1650 class GPU (4GB VRAM) — sufficient since only the small
  classifier head trains; the frozen ResNet18 backbone only ever does a forward pass.
- Deployment: FLIR E8, single radiometric JPEG per capture, read via `flirimageextractor`
  (bundles its own ExifTool via a `dji_executables` transitive dependency — no separate
  system-wide ExifTool install needed).

## Dataset

[Charlotte-ThermalFace](https://github.com/TeCSAR-UNCC) — 10,376 individual thermal images
(no sequences), 10 subjects (S1-S10), each a paired `N<id>.jpg` (8-bit preview) +
`R<id>.tiff` (16-bit raw, `°C = raw/100 - 273.15`). No accompanying annotation files ship
with the archive — room-temperature condition labels are decoded directly from filenames
(the digit right after the subject number in `R<id>.tiff`; see `parse_room_condition`).
Used both for Stage 1 backbone pretraining and as the population source for
`configs/normalization_bounds.yaml`'s P5/P95 bounds (6,640/10,376 images with a
successfully-detected face). Split 70% train / 15% val / 15% test for Stage 1.

A separate, small (16-image) real-FLIR-E8 self-collected set with three self-labeled
conditions (neutral/engaged/stressed) was used to validate the scoring formulas against
real captures — see the Stress Index section above and `scripts/validate_collected_faces.py`.
It's gitignored (real personal thermal images), not redistributed with the repo.

## Ethics

Employee data stays on-device; the employer only ever sees anonymized aggregate team
wellness scores. No visible-spectrum imagery is captured. Designed for compliance with
India's DPDP Act 2023.

## Status

**Current (v2.0) pipeline — live and validated**: ambient-normalized ROI extraction,
all four scoring modules, personal-baseline calibration mode, both dashboards. Stress Index
formula (differential-only) and the population normalization bounds were validated/refit
against a real 16-image self-collected FLIR E8 set — see the Architecture section above.

**Legacy Stage 1/2 pipeline** — Stage 1 (backbone pretraining on the room-temperature-
condition task) is still the source of the frozen ResNet18 used for `measurement_confidence`:
**93.3% test accuracy**, 92-94% per class (`checkpoints/room_temp_backbone/backbone_only.pt`).
Stage 2 (the 3-class engagement classifier — `src/models/fusion_model.py`, `train.py`,
`tests/test_models.py`) is superseded by the v2.0 continuous-score reframe and no longer
part of the live pipeline; the code is still present but not maintained against the
current formulas.
