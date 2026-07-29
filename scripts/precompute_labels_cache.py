"""Precompute per-image ROI proxy vectors + engagement labels for the full dataset and
cache them to data/labels/engagement_labels.json, in the same sample order
src/data/thermal_dataset.py's _index_samples() produces.

Why this exists: WeightedRandomSampler needs to know every training-sample's label
upfront, and re-running the 3-step detection cascade on every __getitem__ call (every
epoch) would make training impractically slow — the full-dataset cascade pass takes
close to 15 minutes on its own. This script pays that cost once; ThermalDataset then
just loads the cached result.

Must be run after scripts/compute_population_baseline.py (needs
data/labels/population_baseline.json to already exist).

Usage:
    python scripts/precompute_labels_cache.py
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import normalize_to_grayscale, raw_to_celsius
from src.roi.extraction import CascadeLandmarkPipeline, extract_roi_temperatures
from src.roi.labeling import (
    DEFAULT_BASELINE_PATH,
    ProxyThresholds,
    load_baseline,
    proxy_vector,
    synthesize_engagement_label,
)

CLASS_NAMES = ["Disengaged", "Neutral", "Engaged"]
DETECTOR_NAMES = ["mediapipe", "haar_default", "lbp", "none"]
DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "labels" / "engagement_labels.json"


def _index_samples(root: Path):
    """Must match src/data/thermal_dataset.py's _index_samples() exactly — cache entries
    are positional, aligned to this same ordering.
    """
    samples = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(subject_dir.iterdir()):
            if f.name.startswith("R") and f.suffix == ".tiff":
                samples.append(f)
    return samples


def main(raw_dir: str, cache_path: Path) -> None:
    root = Path(raw_dir)
    samples = _index_samples(root)
    if not samples:
        raise RuntimeError(f"No R*.tiff files found under {root}")

    baseline = load_baseline(DEFAULT_BASELINE_PATH)
    thresholds = ProxyThresholds.from_config()
    pipeline = CascadeLandmarkPipeline()

    records = []
    label_counts = Counter()
    detector_counts = Counter()
    for i, path in enumerate(samples):
        raw = np.array(Image.open(path))
        temp_c = raw_to_celsius(raw)
        gray = normalize_to_grayscale(temp_c)

        result = pipeline.detect(gray)
        detector_counts[result["detector_used"]] += 1
        if result["success"]:
            roi_temps = extract_roi_temperatures(temp_c, result["landmarks"])
            proxy = proxy_vector(roi_temps, baseline, thresholds)
        else:
            # No face detected by any cascade stage: fall back to a neutral proxy,
            # matching ThermalDataset's prior in-line fallback behavior.
            proxy = [0.0, 0.0]

        label = synthesize_engagement_label(proxy[0], proxy[1])
        label_counts[CLASS_NAMES[label]] += 1
        records.append(
            {
                "path": str(path.relative_to(root)),
                "proxy": proxy,
                "label": label,
                "detector_used": result["detector_used"],
            }
        )

        if (i + 1) % 500 == 0:
            print(f"  cached {i + 1}/{len(samples)}")

    pipeline.close()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(records, f)

    total = len(samples)
    print(f"\nCached {len(records)} labels to {cache_path}")
    print("\nDetector breakdown:")
    for name in DETECTOR_NAMES:
        count = detector_counts.get(name, 0)
        print(f"  {name}: {count} ({count / total:.1%})")
    print("\nLabel distribution (undetected frames fall back to Neutral proxy [0,0]):")
    for name in CLASS_NAMES:
        count = label_counts.get(name, 0)
        print(f"  {name}: {count} ({count / total:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/Charlotte-ThermalFace")
    parser.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH))
    args = parser.parse_args()
    main(args.raw_dir, Path(args.cache_path))
