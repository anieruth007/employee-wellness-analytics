"""Verify WeightedRandomSampler produces a balanced effective class distribution on the
training split, and report the computed class weights — run once after
scripts/precompute_labels_cache.py, before starting real training with train.py.

Usage:
    python scripts/verify_class_balance.py
"""
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch.utils.data import random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.thermal_dataset import ThermalDataset, build_weighted_sampler, compute_class_weights
from src.roi.labeling import CLASS_NAMES


def main():
    with open("configs/cnn_config.yaml") as f:
        cnn_cfg = yaml.safe_load(f)
    with open("configs/fusion_config.yaml") as f:
        fusion_cfg = yaml.safe_load(f)

    dataset = ThermalDataset(raw_dir=cnn_cfg["data"]["raw_dir"], input_size=cnn_cfg["model"]["input_size"])

    val_split = fusion_cfg["data"]["val_split"]
    test_split = fusion_cfg["data"]["test_split"]
    n = len(dataset)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(fusion_cfg["data"]["seed"])
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    print(f"Split sizes: train={len(train_set)} val={len(val_set)} test={len(test_set)}\n")

    train_indices = train_set.indices
    raw_counts = Counter(dataset.labels[i] for i in train_indices)
    print("Raw (pre-sampling) training-split class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        c = raw_counts.get(i, 0)
        print(f"  {name}: {c} ({c / len(train_indices):.1%})")

    num_classes = fusion_cfg["model"]["num_classes"]
    class_weights = compute_class_weights([dataset.labels[i] for i in train_indices], num_classes=num_classes)
    print("\nComputed class weights (inverse frequency, N / (num_classes * count)):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {class_weights[i]:.4f}")

    sampler = build_weighted_sampler(dataset.labels, train_indices, num_classes=num_classes)
    # Sampler yields positions WITHIN the subset (0..len(train_indices)-1), not absolute
    # dataset indices — map back through train_indices before looking up labels.
    drawn_relative = list(sampler)
    drawn_absolute = [train_indices[i] for i in drawn_relative]
    effective_counts = Counter(dataset.labels[i] for i in drawn_absolute)

    print(f"\nEffective class distribution after WeightedRandomSampler ({len(drawn_absolute)} draws, with replacement):")
    for i, name in enumerate(CLASS_NAMES):
        c = effective_counts.get(i, 0)
        print(f"  {name}: {c} ({c / len(drawn_absolute):.1%})")


if __name__ == "__main__":
    main()
