# Beyond Self-Report: Physiologically-Grounded Employee Wellness Monitoring

Thermal-only employee engagement monitoring using facial expression dynamics (CNN + BiLSTM-Attention)
fused with physiological personality-proxy signals (ROI temperature analysis), avoiding self-report
questionnaires and visible-spectrum surveillance.

## Architecture

Two parallel paths from a single thermal camera feed, fused into a 3-way engagement classifier.

**Path 1 — Expression Analysis** (`src/models/thermal_cnn.py`, `src/models/bilstm_attention.py`)
Thermal CNN (pretrained on Charlotte-ThermalFace) → 256-dim spatial features/frame → BiLSTM + Attention
over a 10-frame sequence → 256-dim temporal context vector.

**Path 2 — Physiological Analysis** (`src/roi/`)
72-landmark detection (MediaPipe) → ROI temperature extraction (Nose Tip, Forehead, Periorbital, Upper Lip)
→ Differential Index (Nose Tip − Periorbital) → threshold-based N/C personality-proxy labeling
→ 2-dim proxy vector `[N_proxy, C_proxy]`.

**Fusion** (`src/models/fusion.py`)
`concat(256 + 2) = 258` → FC(258→128) → ReLU → Dropout(0.3) → FC(128→64) → ReLU → Dropout(0.3) →
FC(64→3) → Softmax → `P(Disengaged), P(Neutral), P(Engaged)`.

**Output**: engagement class, wellness score (0-1, derived from `P(Disengaged)`), personality-aware
natural-language explanation, on-device Streamlit dashboard.

## Project layout

```
data/                   Charlotte-ThermalFace raw/processed data, landmarks, ROI temps, synthesized labels
src/data/               loading, normalization, PyTorch Dataset/DataLoader for 16-bit thermal frames
src/roi/                landmark-driven ROI extraction, differential index, threshold labeling
src/models/             thermal CNN, BiLSTM+attention, fusion/classifier head, end-to-end wrapper
src/training/           per-path training scripts (trained separately for 4GB VRAM budget)
src/inference/          single-sample / live inference pipeline
src/utils/              config loading, visualization helpers
dashboard/              Streamlit app (on-device, no raw data leaves the machine)
configs/                YAML hyperparameter/config files per training stage
checkpoints/            saved model weights (gitignored)
logs/                   training logs / tensorboard (gitignored)
notebooks/              EDA, ROI threshold validation
docs/paper/             research paper draft, references, figures
scripts/                dataset download, pipeline runners
tests/                  unit tests for ROI extraction and model shapes
```

## Hardware constraints

NVIDIA GTX 1650 (4GB VRAM) — Path 1 (CNN+BiLSTM) and the fusion head are trained as separate stages
so peak VRAM stays within budget; Path 2 is largely rule-based (no training required beyond threshold
calibration).

## Dataset

[Charlotte-ThermalFace](https://github.com/TeCSAR-UNCC) — 10,000+ raw 16-bit thermal facial images with
per-pixel temperature values and 72 landmarks. Engagement/personality-proxy labels are synthesized from
ROI temperature thresholds grounded in the literature (see `docs/paper/references.bib`).

## Ethics

Employee data stays on-device; the employer only ever sees anonymized aggregate team wellness scores.
No visible-spectrum imagery is captured. Designed for compliance with India's DPDP Act 2023.

## Status

Stage 0 — project scaffolding. See `docs/paper/` for the running research write-up as later stages land.
