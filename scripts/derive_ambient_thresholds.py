"""Derive empirical ambient-normalization thresholds from the Charlotte-ThermalFace
population itself, instead of reusing the old population-baseline thresholds (which don't
work on the ambient-normalized scale — see the smoke-test finding that motivated this).

For every successfully-detected image, reads the already-cached `normalized_roi` (5-dim:
[nose_delta, forehead_delta, periorbital_delta, upper_lip_delta, differential]) from
data/labels/engagement_labels.json, reports distribution stats per delta, derives
mean±1std cutoffs, and reports the resulting class distribution those cutoffs would
produce — WITHOUT re-running detection (reuses the cache scripts/precompute_labels_cache.py
already built).

Usage:
    python scripts/derive_ambient_thresholds.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.roi.labeling import AMBIENT_THRESHOLDS_CONFIG_PATH, CLASS_NAMES, AmbientProxyThresholds, synthesize_engagement_label

CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "labels" / "engagement_labels.json"
DELTA_NAMES = ["nose_delta", "forehead_delta", "periorbital_delta", "upper_lip_delta", "differential"]


def main():
    with open(CACHE_PATH) as f:
        records = json.load(f)

    detected = [r for r in records if r.get("normalized_roi") is not None]
    print(f"Successfully-detected images with normalized_roi: {len(detected)}/{len(records)}")

    deltas = {name: np.array([r["normalized_roi"][i] for r in detected]) for i, name in enumerate(DELTA_NAMES)}

    print("\n=== Distribution statistics per delta (deg C) ===")
    stats = {}
    for name, values in deltas.items():
        s = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        stats[name] = s
        print(f"  {name:20s} mean={s['mean']:7.3f}  std={s['std']:6.3f}  "
              f"p25={s['p25']:7.3f}  p75={s['p75']:7.3f}  min={s['min']:7.3f}  max={s['max']:7.3f}")

    nose_cutoff = stats["nose_delta"]["mean"] - stats["nose_delta"]["std"]
    forehead_cutoff = stats["forehead_delta"]["mean"] + stats["forehead_delta"]["std"]
    diff_cutoff = stats["differential"]["mean"] - stats["differential"]["std"]

    print("\n=== Derived thresholds (mean +/- 1 std) ===")
    print(f"  nose_delta_cutoff_c       = mean({stats['nose_delta']['mean']:.3f}) - std({stats['nose_delta']['std']:.3f}) = {nose_cutoff:.3f}")
    print(f"  forehead_delta_cutoff_c   = mean({stats['forehead_delta']['mean']:.3f}) + std({stats['forehead_delta']['std']:.3f}) = {forehead_cutoff:.3f}")
    print(f"  differential_cutoff_c     = mean({stats['differential']['mean']:.3f}) - std({stats['differential']['std']:.3f}) = {diff_cutoff:.3f}")

    thresholds = AmbientProxyThresholds(
        nose_delta_cutoff_c=nose_cutoff,
        forehead_delta_cutoff_c=forehead_cutoff,
        differential_cutoff_c=diff_cutoff,
    )

    label_counts = {name: 0 for name in CLASS_NAMES}
    for r in detected:
        nose_delta, forehead_delta, _, _, differential = r["normalized_roi"]
        n_proxy = 1 if (nose_delta < thresholds.nose_delta_cutoff_c or differential < thresholds.differential_cutoff_c) else 0
        c_proxy = 1 if forehead_delta > thresholds.forehead_delta_cutoff_c else 0
        label = synthesize_engagement_label(n_proxy, c_proxy)
        label_counts[CLASS_NAMES[label]] += 1

    total = len(detected)
    print(f"\n=== Resulting class distribution with derived thresholds ({total} images) ===")
    for name in CLASS_NAMES:
        count = label_counts[name]
        print(f"  {name}: {count} ({count / total:.1%})")

    max_share = max(label_counts.values()) / total
    print(f"\nLargest single class share: {max_share:.1%}", "-- ", end="")
    if max_share > 0.90:
        print("DEGENERATE: one class dominates, thresholds likely still not usable as-is.")
    else:
        print("looks reasonable (no single class >90%).")

    AMBIENT_THRESHOLDS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AMBIENT_THRESHOLDS_CONFIG_PATH, "w") as f:
        yaml.safe_dump(
            {
                "thresholds": {
                    "nose_delta_cutoff_c": nose_cutoff,
                    "forehead_delta_cutoff_c": forehead_cutoff,
                    "differential_cutoff_c": diff_cutoff,
                },
                "derivation": "mean-1std (nose, differential), mean+1std (forehead), from Charlotte-ThermalFace's own ambient-normalized distribution — see scripts/derive_ambient_thresholds.py",
                "n_images": total,
            },
            f,
        )
    print(f"\nSaved derived thresholds to {AMBIENT_THRESHOLDS_CONFIG_PATH}")


if __name__ == "__main__":
    main()
