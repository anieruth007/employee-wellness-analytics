"""Compute the fixed population-baseline ROI temperatures used by src/roi/labeling.py's
threshold rules (mean nose_tip/forehead/periorbital/upper_lip temps across a sample of
thermal frames). Must be run once before ThermalDataset can be used, since __getitem__
loads this baseline file on init.

Also reports the resulting engagement label distribution that synthesize_engagement_label()
would assign to the same sample using this baseline — reusing the already-extracted ROI
temps rather than re-running detection.

Usage:
    python scripts/compute_population_baseline.py --sample-size 1000
    python scripts/compute_population_baseline.py --sample-size 0   # full dataset
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import normalize_to_grayscale, raw_to_celsius
from src.roi.extraction import CascadeLandmarkPipeline, extract_roi_temperatures
from src.roi.labeling import (
    CLASS_NAMES,
    ProxyThresholds,
    compute_population_baseline,
    proxy_vector,
    save_baseline,
    synthesize_engagement_label,
)

DETECTOR_NAMES = ["mediapipe", "haar_default", "lbp", "none"]


def main(raw_dir: str, sample_size: int, seed: int) -> None:
    root = Path(raw_dir)
    tiffs = [
        p
        for subject_dir in sorted(root.iterdir())
        if subject_dir.is_dir()
        for p in subject_dir.iterdir()
        if p.name.startswith("R") and p.suffix == ".tiff"
    ]
    if not tiffs:
        raise RuntimeError(f"No R*.tiff files found under {root}")

    random.seed(seed)
    if sample_size and sample_size < len(tiffs):
        tiffs = random.sample(tiffs, sample_size)

    pipeline = CascadeLandmarkPipeline()

    roi_samples = []
    detector_counts = Counter()
    for i, path in enumerate(tiffs):
        raw = np.array(Image.open(path))
        temp_c = raw_to_celsius(raw)
        gray = normalize_to_grayscale(temp_c)

        result = pipeline.detect(gray)
        detector_counts[result["detector_used"]] += 1
        if result["success"]:
            roi_samples.append(extract_roi_temperatures(temp_c, result["landmarks"]))

        if (i + 1) % 200 == 0:
            skipped = detector_counts["none"]
            print(f"  processed {i + 1}/{len(tiffs)} ({skipped} skipped so far, no face detected)")

    pipeline.close()
    if not roi_samples:
        raise RuntimeError("No faces detected in any sampled image — check landmark detection.")

    baseline = compute_population_baseline(roi_samples)
    save_baseline(baseline)

    thresholds = ProxyThresholds.from_config()
    label_counts = Counter()
    for roi_temps in roi_samples:
        proxy = proxy_vector(roi_temps, baseline, thresholds)
        label = synthesize_engagement_label(proxy[0], proxy[1])
        label_counts[CLASS_NAMES[label]] += 1

    total = len(tiffs)
    processed = len(roi_samples)
    skipped = total - processed

    print(f"\n=== Population baseline (from {processed}/{total} successfully-detected images) ===")
    for roi_name, temp in baseline.items():
        print(f"  {roi_name}: {temp:.3f} C")

    print(f"\n=== Detection summary ===")
    print(f"  processed: {processed}/{total} ({processed / total:.1%})")
    print(f"  skipped:   {skipped}/{total} ({skipped / total:.1%})")

    print(f"\n=== Detector breakdown ===")
    for name in DETECTOR_NAMES:
        count = detector_counts.get(name, 0)
        print(f"  {name}: {count} ({count / total:.1%})")

    print(f"\n=== Engagement label distribution (of {processed} labeled images) ===")
    for name in CLASS_NAMES:
        count = label_counts.get(name, 0)
        print(f"  {name}: {count} ({count / processed:.1%})")

    print("\nSaved to data/labels/population_baseline.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/Charlotte-ThermalFace")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.raw_dir, args.sample_size, args.seed)
