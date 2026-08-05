"""Charlotte-ThermalFace dataset — single thermal image per sample (no sequences).

Every R<id>.tiff across every subject folder is treated as one independent sample. ROI
proxy vectors and engagement labels are precomputed once via
scripts/precompute_labels_cache.py and loaded from data/labels/engagement_labels.json —
NOT recomputed here — since re-running the 3-step detection cascade on every
__getitem__ call would make training impractically slow.
"""
import json
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data.preprocessing import EVAL_TRANSFORM, normalize_to_grayscale, raw_to_celsius, resize_for_cnn
from src.roi.labeling import raw_temperature_vector

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


def parse_room_condition(path: Path) -> int:
    """Room-temperature condition index (1-4, or 5 for S5's non-standard extra condition)
    encoded as the digit right after the subject number in the numeric filename ID —
    e.g. 'R21101.tiff' (subject S2) -> condition 1, 'R101102.tiff' (subject S10) ->
    condition 1. Holds regardless of the frame-number suffix, since it's always read
    right after the subject digits, unaffected by whatever follows.

    IMPORTANT: the subject-number width varies — S1-S9 are 1 digit, S10 is 2 digits — so
    the offset is derived from the actual subject folder name (path.parent.name), NOT a
    hardcoded position. An earlier version of this function assumed a fixed position 1,
    which silently misparsed all of S10's files as a bogus "condition 0". Verified
    empirically across all 10 subjects: 8/10 show exactly conditions {1,2,3,4} (~30
    sequences each); S1 is missing most of condition 1 (incomplete capture); S5 has an
    extra condition 5.
    """
    subject_digits = path.parent.name.lstrip("S")  # "S1" -> "1", "S10" -> "10"
    core = path.stem[1:]  # strip leading 'N' or 'R'
    return int(core[len(subject_digits)])


def _index_room_temp_samples(root: Path) -> "tuple[List[Path], List[int]]":
    """All R*.tiff files labeled by room-temperature condition (0-3), excluding condition
    5 (S5's non-standard extra condition — only the 4 standard conditions are used).
    """
    samples: List[Path] = []
    labels: List[int] = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(subject_dir.iterdir()):
            if f.name.startswith("R") and f.suffix == ".tiff":
                condition = parse_room_condition(f)
                if condition in (1, 2, 3, 4):
                    samples.append(f)
                    labels.append(condition - 1)
    return samples, labels


class RoomTempDataset(Dataset):
    """(image_tensor[1,48,48], room_temp_label:int) pairs, labeled directly from the
    room-temperature condition digit encoded in each filename (see parse_room_condition) —
    no face detection or ROI extraction needed, so this uses every image in the dataset
    (minus S5's non-standard 5th condition), not just the ~64% with a successfully
    detected face that ThermalDataset is limited to.

    Used to pretrain a CNN/ResNet backbone on a much larger, cleanly-labeled task before
    reusing its features for the noisier, proxy-labeled engagement classification.
    """

    def __init__(
        self,
        raw_dir: str = "data/raw/Charlotte-ThermalFace",
        input_size: int = 48,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(raw_dir)
        self.input_size = input_size
        self.transform = transform or EVAL_TRANSFORM
        self.samples, self.labels = _index_room_temp_samples(self.root)
        if not self.samples:
            raise RuntimeError(f"No labeled room-temperature samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        tiff_path = self.samples[idx]
        raw = np.array(Image.open(tiff_path))
        temp_c = raw_to_celsius(raw)
        gray_full_res = normalize_to_grayscale(temp_c)

        gray_48 = resize_for_cnn(gray_full_res, self.input_size)
        pil_image = Image.fromarray(gray_48, mode="L")
        image_tensor = self.transform(pil_image)
        label = self.labels[idx]

        return image_tensor, label


class ThermalDataset(Dataset):
    """__getitem__ returns (image_tensor[1,48,48], roi_features_tensor[5], label:int).

    roi_features is the raw [nose_temp, forehead_temp, periorbital_temp, upper_lip_temp,
    differential_index] vector (see src/roi/labeling.py::raw_temperature_vector) — NOT the
    binary N/C proxy, which is still computed (self.proxies) for label synthesis and
    dashboard explanations but is never fed to the classifier (target-leakage risk).

    Requires data/labels/engagement_labels.json to already exist — run
    `python scripts/precompute_labels_cache.py` once first (after
    scripts/compute_population_baseline.py). Must be a cache version that includes
    per-record "roi_temps" (raw ROI temperatures, not just the binary proxy).
    """

    def __init__(
        self,
        raw_dir: str = "data/raw/Charlotte-ThermalFace",
        input_size: int = 48,
        labels_cache_path: Optional[Path] = None,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(raw_dir)
        self.input_size = input_size
        # Defaults to no augmentation (EVAL_TRANSFORM) — pass preprocessing.TRAIN_TRANSFORM
        # explicitly for the training split. See train.py's build_dataloaders for how the
        # same train/val/test split is reproduced across two differently-transformed
        # ThermalDataset instances.
        self.transform = transform or EVAL_TRANSFORM

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

        if detected and "roi_temps" not in detected[0]:
            raise RuntimeError(
                f"{cache_path} predates raw ROI-temperature caching (no 'roi_temps' field) "
                "— rerun scripts/precompute_labels_cache.py to regenerate it."
            )

        self.samples: List[Path] = [self.root / r["path"] for r in detected]
        self.proxies: List[List[float]] = [r["proxy"] for r in detected]
        self.roi_features: List[List[float]] = [raw_temperature_vector(r["roi_temps"]) for r in detected]
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
        pil_image = Image.fromarray(gray_48, mode="L")
        image_tensor = self.transform(pil_image)
        roi_features_tensor = torch.tensor(self.roi_features[idx], dtype=torch.float32)
        label = self.labels[idx]

        return image_tensor, roi_features_tensor, label


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
