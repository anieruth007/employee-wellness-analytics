"""Cognitive Load Index (0-100) from ambient-normalized forehead temperature.

Forehead thermal elevation under cognitive load — Frontiers in Psychiatry (2025).
Ambient-normalized (delta above this image's own background temperature) rather than
population-baseline comparison — see src/roi/extraction.py::compute_ambient_temperature.
"""
from pathlib import Path
from typing import Optional, Tuple

import yaml

BOUNDS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "normalization_bounds.yaml"


def compute_cognitive_raw(forehead_delta: float) -> float:
    """cognitive_raw = forehead_delta. Higher forehead_delta (warmer forehead relative to
    ambient) -> higher cognitive load. No transform beyond the ambient normalization
    itself — the delta IS the raw signal here.
    """
    return forehead_delta


def load_cognitive_bounds(config_path: Path = BOUNDS_CONFIG_PATH) -> Tuple[float, float]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    bounds = config["cognitive_load_index"]
    return bounds["p5"], bounds["p95"]


def compute_cognitive_load_index(
    forehead_delta: float,
    p5: Optional[float] = None,
    p95: Optional[float] = None,
) -> float:
    """0-100 Cognitive Load Index. Population P5 maps to 0, P95 maps to 100 (from
    scripts/compute_normalization_bounds.py), clipped to [0, 100] outside that range.
    """
    if p5 is None or p95 is None:
        p5, p95 = load_cognitive_bounds()
    raw = compute_cognitive_raw(forehead_delta)
    scaled = 100 * (raw - p5) / (p95 - p5)
    return max(0.0, min(100.0, scaled))
