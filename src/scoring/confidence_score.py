"""Measurement Confidence (0-100) — how closely the current capture's ResNet18 features
resemble the Charlotte-ThermalFace population's typical thermal face pattern.

Not a confidence in the stress/cognitive-load *values* themselves (those are deterministic
formulas, not model predictions) — it's a face/image-quality sanity check: a low score
means this capture's thermal pattern looks atypical (poor angle, partial occlusion, sensor
noise, non-face content), so the ROI-based readings for it are less trustworthy.

cosine_similarity ranges [-1, 1]; in practice, for real face crops against a population
mean of real face crops, it's almost always well above 0 (all 512-dim ResNet18 features
here come from thermal face images, which share broad structure) — so this rescales
[0, 1] -> [0, 100] rather than [-1, 1] -> [0, 100], or genuinely dissimilar images would
never register below ~50.
"""
from pathlib import Path
from typing import Optional

import numpy as np

MEAN_FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "labels" / "population_mean_features.npy"


def load_population_mean_features(path: Path = MEAN_FEATURES_PATH) -> np.ndarray:
    return np.load(path)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_confidence_score(current_features: np.ndarray, population_mean_features: Optional[np.ndarray] = None) -> float:
    """0-100 measurement confidence. `current_features`: (512,) ResNet18 backbone output
    for the current capture (same frozen backbone used everywhere else in the pipeline).
    """
    if population_mean_features is None:
        population_mean_features = load_population_mean_features()
    sim = cosine_similarity(current_features, population_mean_features)  # ~[0, 1] in practice, see module docstring
    return max(0.0, min(100.0, sim * 100.0))
