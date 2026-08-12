"""Threshold-based personality-proxy labeling (Pipeline 2 output stage).

Baseline is a fixed POPULATION statistic (mean ROI temps under neutral-labeled training
frames), not a per-subject resting-window calibration — the system captures a single
thermal image per employee per session, so there's no live baseline-collection step.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import yaml

from .differential_index import differential_index

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "roi_thresholds.yaml"
DEFAULT_BASELINE_PATH = Path(__file__).resolve().parents[2] / "data" / "labels" / "population_baseline.json"


@dataclass
class ProxyThresholds:
    nose_tip_drop_c: float = 0.5
    nose_tip_stable_band_c: float = 0.2
    forehead_elevation_c: float = 0.3
    differential_index_c: float = -0.5

    @classmethod
    def from_config(cls, config_path: Path = CONFIG_PATH) -> "ProxyThresholds":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(**config["thresholds"])


def compute_personality_proxy(roi_temps: dict, baseline: dict, thresholds: ProxyThresholds) -> tuple:
    """Returns (N_proxy, C_proxy) as 0/1 ints per the research-backed threshold rules:
      - nose tip drop > 0.5C from baseline, OR differential index < -0.5C -> High N (1)
      - forehead elevation > 0.3C from baseline -> High C (1)
    """
    nose_drop = baseline["nose_tip"] - roi_temps["nose_tip"]
    diff_idx = differential_index(roi_temps)
    n_proxy = 1 if (nose_drop > thresholds.nose_tip_drop_c or diff_idx < thresholds.differential_index_c) else 0

    forehead_rise = roi_temps["forehead"] - baseline["forehead"]
    c_proxy = 1 if forehead_rise > thresholds.forehead_elevation_c else 0

    return n_proxy, c_proxy


def ambient_normalized_roi_temps(roi_temps: dict, ambient_temp: float) -> dict:
    """Each ROI temperature expressed as a delta above the image's own ambient
    (background) reference — see src/roi/extraction.py::compute_ambient_temperature.
    """
    return {roi: temp - ambient_temp for roi, temp in roi_temps.items()}


AMBIENT_THRESHOLDS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "ambient_thresholds.yaml"


@dataclass
class AmbientProxyThresholds:
    """Direct cutoff values on ambient-normalized deltas — NOT a magnitude-to-negate like
    ProxyThresholds. ProxyThresholds' fields (e.g. nose_tip_drop_c=0.5) were calibrated for
    skin-vs-population-average-skin comparisons (both ~33-34C, so naturally small, ~0-3C
    deltas). Ambient-normalized deltas (skin-vs-room-background) are structurally larger
    positive numbers (~0-15C, since skin is always warmer than the room) — reusing the old
    "0.5C magnitude below baseline" convention against this differently-scaled quantity
    makes the threshold condition nearly always-false or always-true (see the smoke-test
    finding that motivated this class). These fields are the actual cutoff value the
    normalized delta is compared against, empirically derived from the Charlotte-ThermalFace
    population's own ambient-normalized distribution — see scripts/derive_ambient_thresholds.py.
    """

    nose_delta_cutoff_c: float
    forehead_delta_cutoff_c: float
    differential_cutoff_c: float

    @classmethod
    def from_config(cls, config_path: Path = AMBIENT_THRESHOLDS_CONFIG_PATH) -> "AmbientProxyThresholds":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(**config["thresholds"])


def compute_personality_proxy_ambient(roi_temps: dict, ambient_temp: float, thresholds: AmbientProxyThresholds) -> tuple:
    """Ambient-normalized variant of compute_personality_proxy — compares ROI temperatures
    against this image's own background temperature instead of the fixed population
    baseline, to avoid confusing "cold capture room" with "physiological stress response"
    (both produce a cold-nose/warm-forehead pattern relative to an absolute population
    reference; normalizing against in-image ambient is meant to separate them).

    NOTE: differential_index(roi_temps) = nose_tip - periorbital is unaffected by ambient
    normalization — ambient cancels algebraically: (nose-ambient) - (periorbital-ambient)
    = nose - periorbital, identical to the unnormalized value. Flagged here since it's a
    real, non-obvious consequence of the normalization, not an oversight.
    """
    normalized = ambient_normalized_roi_temps(roi_temps, ambient_temp)
    nose_delta = normalized["nose_tip"]
    forehead_delta = normalized["forehead"]
    diff_idx = differential_index(roi_temps)  # ambient-invariant, see docstring above

    n_proxy = 1 if (nose_delta < thresholds.nose_delta_cutoff_c or diff_idx < thresholds.differential_cutoff_c) else 0
    c_proxy = 1 if forehead_delta > thresholds.forehead_delta_cutoff_c else 0
    return n_proxy, c_proxy


def proxy_vector_ambient(roi_temps: dict, ambient_temp: float, thresholds: AmbientProxyThresholds) -> List[float]:
    """[N_proxy, C_proxy] using ambient-normalized thresholding (see compute_personality_proxy_ambient)."""
    n_proxy, c_proxy = compute_personality_proxy_ambient(roi_temps, ambient_temp, thresholds)
    return [float(n_proxy), float(c_proxy)]


def ambient_normalized_temperature_vector(roi_temps: dict, ambient_temp: float) -> List[float]:
    """5-dim [nose_delta, forehead_delta, periorbital_delta, upper_lip_delta, differential]
    — each ROI temperature expressed relative to this image's own ambient (background)
    reference, replacing raw_temperature_vector as the fusion classifier's ROI-derived
    input. `differential = nose_delta - periorbital_delta`, algebraically identical to the
    unnormalized differential_index(roi_temps) since ambient cancels — computed via the
    normalized values here anyway, per spec, for directness/symmetry with the other deltas.
    """
    normalized = ambient_normalized_roi_temps(roi_temps, ambient_temp)
    return [
        normalized["nose_tip"],
        normalized["forehead"],
        normalized["periorbital"],
        normalized["upper_lip"],
        normalized["nose_tip"] - normalized["periorbital"],
    ]


def proxy_vector(roi_temps: dict, baseline: dict, thresholds: ProxyThresholds) -> List[float]:
    """[N_proxy, C_proxy] as floats. Used ONLY for label synthesis (synthesize_engagement_label)
    and the dashboard's personality-aware explanation text — NOT fed to the classifier
    (see raw_temperature_vector, and fusion_model.py's module docstring for why: this
    binary proxy directly determines the training label, so it caused ~100% accuracy via
    target leakage when it was part of the classifier's input).
    """
    n_proxy, c_proxy = compute_personality_proxy(roi_temps, baseline, thresholds)
    return [float(n_proxy), float(c_proxy)]


def raw_temperature_vector(roi_temps: dict) -> List[float]:
    """5-dim [nose_temp, forehead_temp, periorbital_temp, upper_lip_temp,
    differential_index] — raw, un-thresholded ROI temperatures, fed to the fusion
    classifier as its ROI-derived input (replacing the binary N/C proxy).

    Less directly leaky than proxy_vector's binary output — these are a superset of
    information the proxy is thresholded FROM, not the literal label-generating value —
    but still correlated with the label, since the label is itself a threshold function of
    these same quantities. Worth scrutinizing results with this in mind rather than
    treating it as leakage-free.
    """
    return [
        roi_temps["nose_tip"],
        roi_temps["forehead"],
        roi_temps["periorbital"],
        roi_temps["upper_lip"],
        differential_index(roi_temps),
    ]


CLASS_NAMES = ["Disengaged", "Burned Out", "Engaged"]


def synthesize_engagement_label(n_proxy: float, c_proxy: float) -> int:
    """Maps personality-proxy signals to a 3-way engagement label for supervised training,
    since Charlotte-ThermalFace has no ground-truth engagement labels. Grounded in
    Barrick & Mount (1991): Conscientiousness is the stronger, more direct predictor of
    engagement/performance than Neuroticism, so C is the primary split and N only
    distinguishes within the "high C" branch.

      0 = Disengaged (C=0, any N): low sustained-attention marker dominates regardless of
          stress state — (N=1,C=0) and (N=0,C=0) are merged because both present as
          disengaged; the defining signal is the absence of Conscientiousness, not N.
      1 = Burned Out (N=1, C=1): high stress marker BUT still high sustained-attention —
          pushing through despite strain. Kept as its own class (not merged into
          Disengaged or Engaged) since it's a physiologically and practically distinct,
          high-retention-risk state despite outwardly "engaged" behavior.
      2 = Engaged (N=0, C=1): calm and focused — the intended target state.

    Second iteration of this rule (project-owner signed off, not itself derived from the
    cited literature). The prior 4-class version (Disengaged / Checked Out / Burned Out /
    Engaged) split C=0 into two classes by N; collapsing them back here is a deliberate
    redesign choice, not a reversion of the earlier fix that removed the old *3-class*
    "Neutral" catch-all (which conflated Checked-Out and Burned-Out under one label) —
    this new Disengaged class conflates Checked-Out and (low-C) Disengaged instead, which
    is a different, intentional merge.
    """
    if c_proxy == 0:
        return 0  # Disengaged (low C, regardless of N)
    if n_proxy == 1 and c_proxy == 1:
        return 1  # Burned Out
    return 2  # Engaged (n_proxy == 0 and c_proxy == 1)


def compute_population_baseline(roi_temp_samples: List[dict]) -> dict:
    """Average ROI temperatures across a population sample, e.g. all training-split frames."""
    keys = roi_temp_samples[0].keys()
    return {k: float(np.mean([s[k] for s in roi_temp_samples])) for k in keys}


def save_baseline(baseline: dict, path: Path = DEFAULT_BASELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)


def load_baseline(path: Path = DEFAULT_BASELINE_PATH) -> dict:
    with open(path) as f:
        return json.load(f)
