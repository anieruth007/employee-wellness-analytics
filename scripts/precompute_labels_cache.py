"""Precompute per-image ambient-normalized ROI vectors + engagement labels for the full
dataset and cache them to data/labels/engagement_labels.json, in the same sample order
src/data/thermal_dataset.py's _index_samples() produces.

Why this exists: WeightedRandomSampler needs to know every training-sample's label
upfront, and re-running the 3-step detection cascade on every __getitem__ call (every
epoch) would make training impractically slow — the full-dataset cascade pass takes
close to 15 minutes on its own. This script pays that cost once; ThermalDataset then
just loads the cached result.

Labels are now synthesized via ambient-normalized thresholding (compare ROI temps against
this image's own background/non-face pixels, not a fixed population baseline) — see
src/roi/labeling.py::compute_personality_proxy_ambient. The old population-baseline-based
proxy/label is also computed and cached (as *_baseline fields) purely for before/after
comparison; nothing downstream reads them.

Must be run after scripts/compute_population_baseline.py (needs
data/labels/population_baseline.json to already exist — used only for the comparison
fields now, not for the actual labels).

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
from src.roi.extraction import CascadeLandmarkPipeline, compute_ambient_temperature, extract_roi_temperatures
from src.roi.labeling import (
    CLASS_NAMES,
    AmbientProxyThresholds,
    DEFAULT_BASELINE_PATH,
    ProxyThresholds,
    ambient_normalized_temperature_vector,
    load_baseline,
    proxy_vector,
    proxy_vector_ambient,
    synthesize_engagement_label,
)

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
    label_counts_baseline = Counter()
    detector_counts = Counter()
    for i, path in enumerate(samples):
        raw = np.array(Image.open(path))
        temp_c = raw_to_celsius(raw)
        gray = normalize_to_grayscale(temp_c)

        result = pipeline.detect(gray)
        detector_counts[result["detector_used"]] += 1
        if result["success"]:
            roi_temps = extract_roi_temperatures(temp_c, result["landmarks"])
            ambient_temp = compute_ambient_temperature(temp_c, result["bbox"])
            normalized_roi = ambient_normalized_temperature_vector(roi_temps, ambient_temp)
            proxy = proxy_vector_ambient(roi_temps, ambient_temp, thresholds)
            proxy_baseline = proxy_vector(roi_temps, baseline, thresholds)  # old method, comparison only
        else:
            # No face detected by any cascade stage: no real roi_temps exist. Fall back to
            # a neutral proxy for the (unused, ThermalDataset excludes these) label field.
            roi_temps = None
            ambient_temp = None
            normalized_roi = None
            proxy = [0.0, 0.0]
            proxy_baseline = [0.0, 0.0]

        label = synthesize_engagement_label(proxy[0], proxy[1])
        label_baseline = synthesize_engagement_label(proxy_baseline[0], proxy_baseline[1])
        label_counts[CLASS_NAMES[label]] += 1
        label_counts_baseline[CLASS_NAMES[label_baseline]] += 1
        records.append(
            {
                "path": str(path.relative_to(root)),
                "roi_temps": roi_temps,
                "ambient_temp": ambient_temp,
                "normalized_roi": normalized_roi,
                "proxy": proxy,
                "label": label,
                "proxy_baseline": proxy_baseline,
                "label_baseline": label_baseline,
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

    print(f"\nLabel distribution — AMBIENT-NORMALIZED (new, actually used):")
    for name in CLASS_NAMES:
        count = label_counts.get(name, 0)
        print(f"  {name}: {count} ({count / total:.1%})")

    print(f"\nLabel distribution — POPULATION BASELINE (old, comparison only):")
    for name in CLASS_NAMES:
        count = label_counts_baseline.get(name, 0)
        print(f"  {name}: {count} ({count / total:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/Charlotte-ThermalFace")
    parser.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH))
    args = parser.parse_args()
    main(args.raw_dir, Path(args.cache_path))
