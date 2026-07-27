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
    """[N_proxy, C_proxy] as floats, ready to feed into the fusion layer."""
    n_proxy, c_proxy = compute_personality_proxy(roi_temps, baseline, thresholds)
    return [float(n_proxy), float(c_proxy)]


def synthesize_engagement_label(n_proxy: float, c_proxy: float) -> int:
    """Maps personality-proxy signals to a 3-way engagement label for supervised training
    (0=Disengaged, 1=Neutral, 2=Engaged), since Charlotte-ThermalFace has no ground-truth
    engagement labels.

    UNVALIDATED PLACEHOLDER: this specific N/C -> engagement mapping rule has not been
    derived from or checked against the cited literature (Barrick & Mount 1991 establishes
    N/C as predictors, not this exact rule). Treat trained results as provisional until this
    rule is reviewed — it directly defines the training targets for the whole classifier.
    """
    if n_proxy == 1 and c_proxy == 0:
        return 0  # high stress marker + low sustained-attention marker
    if n_proxy == 0 and c_proxy == 1:
        return 2  # calm + high conscientiousness marker
    return 1  # mixed/ambiguous signal


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
