"""Charlotte-ThermalFace dataset — single thermal image per sample (no sequences).

Every R<id>.tiff across every subject folder is treated as one independent sample.
Landmark detection runs on the full-resolution normalized grayscale image; the CNN
branch separately gets a 48x48 resize of that same image.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.preprocessing import normalize_to_grayscale, raw_to_celsius, resize_for_cnn
from src.roi.extraction import CascadeLandmarkPipeline, extract_roi_temperatures
from src.roi.labeling import (
    DEFAULT_BASELINE_PATH,
    ProxyThresholds,
    load_baseline,
    proxy_vector,
    synthesize_engagement_label,
)


def _index_samples(root: Path):
    samples = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(subject_dir.iterdir()):
            if f.name.startswith("R") and f.suffix == ".tiff":
                samples.append(f)
    return samples


class ThermalDataset(Dataset):
    """__getitem__ returns (image_tensor[1,48,48], proxy_tensor[2], label:int)."""

    def __init__(
        self,
        raw_dir: str = "data/raw/Charlotte-ThermalFace",
        input_size: int = 48,
        baseline_path: Optional[Path] = None,
        thresholds: Optional[ProxyThresholds] = None,
    ):
        self.root = Path(raw_dir)
        self.samples = _index_samples(self.root)
        if not self.samples:
            raise RuntimeError(f"No R*.tiff files found under {self.root}")
        self.input_size = input_size
        self.baseline = load_baseline(baseline_path or DEFAULT_BASELINE_PATH)
        self.thresholds = thresholds or ProxyThresholds.from_config()
        self._pipeline = None  # lazy: not picklable across DataLoader worker processes

    def _get_pipeline(self) -> CascadeLandmarkPipeline:
        if self._pipeline is None:
            self._pipeline = CascadeLandmarkPipeline()
        return self._pipeline

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        tiff_path = self.samples[idx]
        raw = np.array(Image.open(tiff_path))
        temp_c = raw_to_celsius(raw)
        gray_full_res = normalize_to_grayscale(temp_c)

        result = self._get_pipeline().detect(gray_full_res)
        if result["success"]:
            roi_temps = extract_roi_temperatures(temp_c, result["landmarks"])
            proxy = proxy_vector(roi_temps, self.baseline, self.thresholds)
        else:
            # No face detected by any cascade stage: fall back to a neutral proxy rather
            # than dropping the sample.
            proxy = [0.0, 0.0]

        gray_48 = resize_for_cnn(gray_full_res, self.input_size)
        image_tensor = torch.from_numpy(gray_48).float().unsqueeze(0) / 255.0
        proxy_tensor = torch.tensor(proxy, dtype=torch.float32)
        label = synthesize_engagement_label(proxy[0], proxy[1])

        return image_tensor, proxy_tensor, label
