"""Stress Index (0-100) from ambient-normalized ROI deltas.

Grounded in Fernandez et al. (2024, Anxiety Stress & Coping) and Gioia et al. (2023,
Sensors): nose-tip cooling (peripheral vasoconstriction under autonomic stress) and a
larger nose-vs-periorbital differential both track stress response. Ambient-normalized
(delta above this image's own background temperature) rather than population-baseline
comparison — see src/roi/extraction.py::compute_ambient_temperature for why.
"""
from pathlib import Path
from typing import Optional, Tuple

import yaml

BOUNDS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "normalization_bounds.yaml"


def compute_stress_raw(nose_delta: float, differential: float) -> float:
    """stress_raw = (-nose_delta + abs(differential)) / 2.

    Lower nose_delta (colder nose relative to ambient) -> higher stress.
    Larger |differential| (bigger nose-vs-periorbital gap, either direction) -> higher stress.
    """
    return (-nose_delta + abs(differential)) / 2


def load_stress_bounds(config_path: Path = BOUNDS_CONFIG_PATH) -> Tuple[float, float]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    bounds = config["stress_index"]
    return bounds["p5"], bounds["p95"]


def compute_stress_index(
    nose_delta: float,
    differential: float,
    p5: Optional[float] = None,
    p95: Optional[float] = None,
) -> float:
    """0-100 Stress Index. Population P5 maps to 0, P95 maps to 100 (from
    scripts/compute_normalization_bounds.py), clipped to [0, 100] outside that range —
    real captures can fall outside the P5-P95 band the population bounds were derived from.
    """
    if p5 is None or p95 is None:
        p5, p95 = load_stress_bounds()
    raw = compute_stress_raw(nose_delta, differential)
    scaled = 100 * (raw - p5) / (p95 - p5)
    return max(0.0, min(100.0, scaled))
