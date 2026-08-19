"""Step 1 of the physiological-monitoring reframe: compute P5/P95 normalization bounds
for the Stress Index and Cognitive Load Index, from the Charlotte-ThermalFace population's
own ambient-normalized ROI distribution (already cached — no re-detection needed).

  stress_raw     = abs(differential)
  cognitive_raw  = forehead_delta

stress_raw was originally (-nose_delta + abs(differential)) / 2 — dropped nose_delta after
validating on a 16-image self-collected FLIR E8 set (scripts/validate_collected_faces.py):
abs(differential) alone cleanly ranked stressed > neutral > engaged matching self-report
labels, while nose_delta ranked the conditions backwards. See src/scoring/stress_index.py.

Saves bounds to configs/normalization_bounds.yaml, consumed by
src/scoring/stress_index.py and src/scoring/cognitive_load_index.py.

Usage:
    python scripts/compute_normalization_bounds.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "labels" / "engagement_labels.json"
BOUNDS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "normalization_bounds.yaml"


def main():
    with open(CACHE_PATH) as f:
        records = json.load(f)

    detected = [r for r in records if r.get("normalized_roi") is not None]
    print(f"Successfully-detected images with normalized_roi: {len(detected)}/{len(records)}")

    # normalized_roi = [nose_delta, forehead_delta, periorbital_delta, upper_lip_delta, differential]
    forehead_delta = np.array([r["normalized_roi"][1] for r in detected])
    differential = np.array([r["normalized_roi"][4] for r in detected])

    stress_raw = np.abs(differential)
    cognitive_raw = forehead_delta  # as specified, no transform

    stress_p5, stress_p95 = float(np.percentile(stress_raw, 5)), float(np.percentile(stress_raw, 95))
    cognitive_p5, cognitive_p95 = float(np.percentile(cognitive_raw, 5)), float(np.percentile(cognitive_raw, 95))

    print("\n=== stress_raw population distribution ===")
    print(f"  mean={stress_raw.mean():.3f}  std={stress_raw.std():.3f}  "
          f"P5={stress_p5:.3f}  P95={stress_p95:.3f}  min={stress_raw.min():.3f}  max={stress_raw.max():.3f}")

    print("\n=== cognitive_raw (forehead_delta) population distribution ===")
    print(f"  mean={cognitive_raw.mean():.3f}  std={cognitive_raw.std():.3f}  "
          f"P5={cognitive_p5:.3f}  P95={cognitive_p95:.3f}  min={cognitive_raw.min():.3f}  max={cognitive_raw.max():.3f}")

    bounds = {
        "stress_index": {"p5": stress_p5, "p95": stress_p95},
        "cognitive_load_index": {"p5": cognitive_p5, "p95": cognitive_p95},
        "derivation": (
            "P5/P95 of stress_raw=abs(differential) and cognitive_raw=forehead_delta, "
            "computed over Charlotte-ThermalFace's ambient-normalized population — see "
            "scripts/compute_normalization_bounds.py"
        ),
        "n_images": len(detected),
    }
    BOUNDS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BOUNDS_CONFIG_PATH, "w") as f:
        yaml.safe_dump(bounds, f)
    print(f"\nSaved normalization bounds to {BOUNDS_CONFIG_PATH}")


if __name__ == "__main__":
    main()
