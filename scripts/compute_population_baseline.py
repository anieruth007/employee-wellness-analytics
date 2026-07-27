"""Compute the fixed population-baseline ROI temperatures used by src/roi/labeling.py's
threshold rules (mean nose_tip/forehead/periorbital/upper_lip temps across a sample of
neutral thermal frames). Must be run once before ThermalDataset can be used, since
__getitem__ loads this baseline file on init.

Usage:
    python scripts/compute_population_baseline.py --sample-size 1000
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import normalize_to_grayscale, raw_to_celsius
from src.roi.extraction import LandmarkDetector, extract_roi_temperatures, load_roi_landmarks
from src.roi.labeling import compute_population_baseline, save_baseline


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

    roi_landmarks = load_roi_landmarks()
    detector = LandmarkDetector()

    roi_samples = []
    skipped = 0
    for i, path in enumerate(tiffs):
        raw = np.array(Image.open(path))
        temp_c = raw_to_celsius(raw)
        gray = normalize_to_grayscale(temp_c)
        landmarks = detector.detect(gray)
        if landmarks is None:
            skipped += 1
            continue
        roi_samples.append(extract_roi_temperatures(temp_c, landmarks, roi_landmarks))
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(tiffs)} ({skipped} skipped, no face detected)")

    detector.close()
    if not roi_samples:
        raise RuntimeError("No faces detected in any sampled image — check landmark detection.")

    baseline = compute_population_baseline(roi_samples)
    save_baseline(baseline)
    print(f"\nBaseline computed from {len(roi_samples)} images ({skipped} skipped, no face detected):")
    print(baseline)
    print("Saved to data/labels/population_baseline.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/Charlotte-ThermalFace")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.raw_dir, args.sample_size, args.seed)
