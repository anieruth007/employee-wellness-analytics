"""Charlotte-ThermalFace dataset — single thermal image per sample (no sequences).

Every R<id>.tiff across every subject folder is treated as one independent sample. ROI
proxy vectors and engagement labels are precomputed once via
scripts/precompute_labels_cache.py and loaded from data/labels/engagement_labels.json —
NOT recomputed here — since re-running the 3-step detection cascade on every
__getitem__ call would make training impractically slow.
"""
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data.preprocessing import normalize_to_grayscale, raw_to_celsius, resize_for_cnn

DEFAULT_LABELS_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "labels" / "engagement_labels.json"


def _index_samples(root: Path) -> List[Path]:
    """Must match scripts/precompute_labels_cache.py's _index_samples() exactly — the
    cache is positional, aligned to this ordering.
    """
    samples = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(subject_dir.iterdir()):
            if f.name.startswith("R") and f.suffix == ".tiff":
                samples.append(f)
    return samples


class ThermalDataset(Dataset):
    """__getitem__ returns (image_tensor[1,48,48], proxy_tensor[2], label:int).

    Requires data/labels/engagement_labels.json to already exist — run
    `python scripts/precompute_labels_cache.py` once first (after
    scripts/compute_population_baseline.py).
    """

    def __init__(
        self,
        raw_dir: str = "data/raw/Charlotte-ThermalFace",
        input_size: int = 48,
        labels_cache_path: Optional[Path] = None,
    ):
        self.root = Path(raw_dir)
        self.input_size = input_size

        cache_path = Path(labels_cache_path) if labels_cache_path else DEFAULT_LABELS_CACHE_PATH
        if not cache_path.exists():
            raise FileNotFoundError(
                f"{cache_path} not found — run `python scripts/precompute_labels_cache.py` "
                "first (after scripts/compute_population_baseline.py)."
            )
        with open(cache_path) as f:
            records = json.load(f)

        all_raw_files = _index_samples(self.root)
        if len(records) != len(all_raw_files):
            raise RuntimeError(
                f"{cache_path} has {len(records)} entries but {len(all_raw_files)} raw images "
                f"were found under {self.root} — the cache is stale, rerun "
                "scripts/precompute_labels_cache.py."
            )

        # Frames where every cascade stage failed to find a face have no real proxy
        # signal — they were previously defaulted to a neutral [0,0] proxy, which
        # silently inflated the Neutral class with detection failures rather than actual
        # neutral engagement. Excluded entirely rather than mislabeled.
        detected = [r for r in records if r["detector_used"] != "none"]
        self.num_excluded = len(records) - len(detected)

        self.samples: List[Path] = [self.root / r["path"] for r in detected]
        self.proxies: List[List[float]] = [r["proxy"] for r in detected]
        self.labels: List[int] = [r["label"] for r in detected]
        if not self.samples:
            raise RuntimeError(f"No successfully-detected samples found in {cache_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        tiff_path = self.samples[idx]
        raw = np.array(Image.open(tiff_path))
        temp_c = raw_to_celsius(raw)
        gray_full_res = normalize_to_grayscale(temp_c)

        gray_48 = resize_for_cnn(gray_full_res, self.input_size)
        image_tensor = torch.from_numpy(gray_48).float().unsqueeze(0) / 255.0
        proxy_tensor = torch.tensor(self.proxies[idx], dtype=torch.float32)
        label = self.labels[idx]

        return image_tensor, proxy_tensor, label


def compute_class_weights(labels: List[int], num_classes: int = 3) -> torch.Tensor:
    """Inverse class-frequency weights: weight[c] = N / (num_classes * count[c]).
    Used to derive WeightedRandomSampler per-sample weights (see build_weighted_sampler).
    Not also applied to the loss function — combining sampler-based oversampling with
    loss-weighting double-corrects for imbalance and risks overshooting into
    over-predicting the minority class.
    """
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    counts = counts.clamp(min=1)  # avoid div-by-zero if a class is entirely absent from this subset
    return len(labels) / (num_classes * counts)


def build_weighted_sampler(labels: List[int], indices: List[int], num_classes: int = 3) -> WeightedRandomSampler:
    """WeightedRandomSampler over a training-split subset, oversampling underrepresented
    classes (e.g. Disengaged) so each training epoch sees roughly balanced classes.

    `indices`: positions into the FULL dataset's `labels` list that make up the training
    split (e.g. `Subset.indices` from `torch.utils.data.random_split`).

    NOTE: the returned sampler yields indices in [0, len(indices)) — i.e. positions
    *within the subset*, not absolute dataset indices. This is exactly what
    `DataLoader(subset, sampler=...)` expects (Subset translates them internally), but if
    you ever consume the sampler directly, remember to map back via `indices[i]`.
    """
    subset_labels = [labels[i] for i in indices]
    class_weights = compute_class_weights(subset_labels, num_classes)
    sample_weights = [class_weights[label].item() for label in subset_labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
