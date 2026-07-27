# Beyond Self-Report: Physiologically-Grounded Employee Wellness Monitoring

Thermal-only employee engagement monitoring from a **single thermal image** — expression
features (CNN) fused with physiological personality-proxy signals (ROI temperature
analysis) — avoiding self-report questionnaires and visible-spectrum surveillance.

## Core design decision

One thermal image per employee per session — no sequence, no multiple shots, no waiting.
The system is designed for seamless integration with existing single-frame biometric
attendance infrastructure (badge-in style capture), not a continuous monitoring feed.

## Architecture

A single processed thermal image feeds two parallel pipelines, fused into a 3-way
engagement classifier.

**Input**
FLIR E8 thermal capture (USB/WiFi transfer) → `flirimageextractor` extracts the raw
per-pixel temperature array (°C) → normalized to 0-255 grayscale → resized to 48×48 for
the CNN branch. Landmark detection (Pipeline 2) runs on the normalized grayscale image
**before** the 48×48 resize — 48×48 is too small for reliable MediaPipe landmark detection.

**Pipeline 1 — Expression Analysis** (`src/models/thermal_cnn.py`)
Single 48×48 thermal image → Thermal CNN encoder:
- Block 1: Conv(1→32)×2 + BN + ReLU → MaxPool → Dropout(0.25) → (32, 24, 24)
- Block 2: Conv(32→64)×2 + BN + ReLU → MaxPool → Dropout(0.25) → (64, 12, 12)
- Block 3: Conv(64→128) + BN + ReLU → MaxPool → Dropout(0.25) → (128, 6, 6)
- Flatten (4608) → FC(4608→256) → ReLU → Dropout(0.5)
→ 256-dim expression feature vector. No temporal module — BiLSTM+Attention has been
removed entirely; this path is per-image only.

**Pipeline 2 — Physiological Analysis** (`src/roi/`)
MediaPipe FaceMesh landmark detection → ROI temperature extraction:
- Nose Tip → landmark 4
- Forehead → landmarks 10, 338, 297, 332, 284
- Periorbital → landmarks 33, 133, 362, 263
- Upper Lip → landmark 13

Differential Index = `mean(nose_tip_temp) - mean(periorbital_temp)`. Threshold-based
proxy labeling against a **fixed population baseline** (not a per-subject resting-window
calibration — there's no time for that in a single-shot attendance capture):
- Nose tip drop > 0.5°C from baseline, or differential index < -0.5°C → High N proxy
- Nose tip stable within ±0.2°C of baseline → Low N proxy
- Forehead elevation > 0.3°C from baseline → High C proxy
→ 2-dim personality proxy vector `[N_proxy, C_proxy]`.

**Fusion** (`src/models/fusion_model.py`)
`concat(256 + 2) = 258` → FC(258→128) → ReLU → Dropout(0.3) → FC(128→64) → ReLU →
Dropout(0.3) → FC(64→3) → `P(Disengaged), P(Neutral), P(Engaged)`. The model returns raw
logits from `forward()` (for `CrossEntropyLoss` during training) and applies softmax only
in `predict_proba()` at inference — this avoids a double-softmax bug.

**Output**: engagement class, wellness score (0-1, `P(Disengaged)`), N/C proxy values,
personality-aware natural-language explanation, on-device Streamlit dashboard.

Because the sequence/BiLSTM path is gone and the CNN input shrank to 48×48, the whole
model is small enough to **train end-to-end in a single stage** on a 4GB-VRAM GPU — the
earlier "train paths separately to manage VRAM" workaround is no longer needed.

## Open item — needs your input

`src/roi/labeling.py::synthesize_engagement_label` currently maps `[N_proxy, C_proxy]` to
one of the 3 engagement classes with a placeholder rule (high N + low C → Disengaged, low
N + high C → Engaged, else Neutral) purely so the pipeline runs end-to-end. This rule has
**not** been derived from or validated against the cited literature and directly defines
the training targets for the whole classifier — it needs your review before results are
treated as anything more than a working scaffold.

## Project layout

```
data/raw/Charlotte-ThermalFace/  extracted dataset: S1-S10 subject folders, N<id>.jpg + R<id>.tiff pairs
data/labels/                     synthesized labels, population_baseline.json
src/data/                        preprocessing.py (calibration/normalize/resize), thermal_dataset.py (single-image samples)
src/roi/                         extraction.py (landmarks+ROI temps), differential_index.py, labeling.py (thresholds, proxy, baseline)
src/models/                      thermal_cnn.py (encoder), fusion_model.py (fusion classifier + end-to-end wrapper)
dashboard/app.py                 Streamlit image-upload dashboard
configs/                         cnn_config.yaml, fusion_config.yaml, roi_thresholds.yaml
scripts/compute_population_baseline.py   run once before training — builds data/labels/population_baseline.json
train.py                         end-to-end training entrypoint
inference.py                     single FLIR E8 image -> classification + explanation
checkpoints/, logs/               (gitignored)
docs/paper/                       paper draft + references.bib
tests/                            model shape tests
```

## Hardware / target camera

- Training: NVIDIA GTX 1650, 4GB VRAM — sufficient for joint end-to-end training post-BiLSTM-removal.
- Deployment: FLIR E8, single radiometric JPEG per capture, read via `flirimageextractor`
  (also requires [ExifTool](https://exiftool.org/) installed system-wide).

## Dataset

[Charlotte-ThermalFace](https://github.com/TeCSAR-UNCC) — 10,376 individual thermal images
(no sequences), 10 subjects (S1-S10), each a paired `N<id>.jpg` (8-bit preview, used for
sizing/inspection) + `R<id>.tiff` (16-bit raw, `°C = raw/100 - 273.15`). Split 70% train /
15% val / 15% test. Labels synthesized via the ROI threshold pipeline — see the open item
above.

## Ethics

Employee data stays on-device; the employer only ever sees anonymized aggregate team
wellness scores. No visible-spectrum imagery is captured. Designed for compliance with
India's DPDP Act 2023.

## Status

Architecture scaffolding for the single-image (no-BiLSTM) design is in place and
shape-tested (`pytest tests/`). Not yet run: `scripts/compute_population_baseline.py`
against the full dataset, and a real training pass via `train.py`.
