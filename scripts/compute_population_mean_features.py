"""Compute the mean 512-dim ResNet18 backbone feature vector across the
Charlotte-ThermalFace population — the reference point for
src/scoring/confidence_score.py's cosine-similarity "measurement confidence" score.

Uses the same successfully-detected image set already cached in
data/labels/engagement_labels.json (no re-detection needed) and the same frozen,
room-temperature-pretrained backbone used everywhere else in the pipeline.

Usage:
    python scripts/compute_population_mean_features.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import EVAL_TRANSFORM, normalize_to_grayscale, raw_to_celsius, resize_for_cnn
from src.models.resnet_backbone import load_frozen_backbone

CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "labels" / "engagement_labels.json"
RAW_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "Charlotte-ThermalFace"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "labels" / "population_mean_features.npy"
BACKBONE_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "room_temp_backbone" / "backbone_only.pt"


def main(batch_size: int = 64):
    with open(CACHE_PATH) as f:
        records = json.load(f)
    detected = [r for r in records if r.get("normalized_roi") is not None]
    print(f"Computing mean features over {len(detected)} successfully-detected images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    backbone = load_frozen_backbone(BACKBONE_CHECKPOINT, device=device)

    all_features = []
    batch_tensors = []

    def flush(batch_tensors):
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            feats = backbone(batch)
        all_features.append(feats.cpu().numpy())

    for i, r in enumerate(detected):
        path = RAW_ROOT / r["path"]
        raw = np.array(Image.open(path))
        temp_c = raw_to_celsius(raw)
        gray_full_res = normalize_to_grayscale(temp_c)
        gray_48 = resize_for_cnn(gray_full_res, 48)
        image_tensor = EVAL_TRANSFORM(Image.fromarray(gray_48))
        batch_tensors.append(image_tensor)

        if len(batch_tensors) == batch_size:
            flush(batch_tensors)
            batch_tensors = []

        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{len(detected)}")

    flush(batch_tensors)

    all_features = np.concatenate(all_features, axis=0)
    mean_features = all_features.mean(axis=0)
    print(f"\nComputed mean feature vector: shape={mean_features.shape}, "
          f"norm={np.linalg.norm(mean_features):.3f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, mean_features)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
