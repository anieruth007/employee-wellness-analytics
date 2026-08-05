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
