"""Shared thermal image preprocessing: calibration, grayscale normalization, CNN resize,
CLAHE contrast enhancement for landmark detection, and the train/eval tensor pipelines.

Used identically by training data loading (thermal_dataset.py) and live inference
(inference.py) so the model never sees a train/inference distribution mismatch — EVAL_TRANSFORM
(no augmentation) is used for both validation/test and live inference; TRAIN_TRANSFORM
(augmented) is used only for the training split.
"""
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def raw_to_celsius(raw_uint16: np.ndarray) -> np.ndarray:
    """Charlotte-ThermalFace raw TIFF values are centikelvin: °C = raw/100 - 273.15."""
    return raw_uint16.astype(np.float32) / 100.0 - 273.15


def normalize_to_grayscale(temp_c: np.ndarray) -> np.ndarray:
    """Min-max normalize a calibrated Celsius temperature array to 0-255 uint8."""
    t_min, t_max = float(temp_c.min()), float(temp_c.max())
    if t_max - t_min < 1e-6:
        return np.zeros_like(temp_c, dtype=np.uint8)
    scaled = (temp_c - t_min) / (t_max - t_min) * 255.0
    return scaled.astype(np.uint8)


def apply_clahe(gray_uint8: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization — boosts local contrast on the
    normalized grayscale thermal image before MediaPipe landmark detection. MediaPipe's
    face landmarker is trained on visible-spectrum faces and otherwise misses a large
    fraction of low-contrast thermal frames. Only used for the landmark-detection input;
    the CNN branch and ROI temperature extraction are unaffected.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_uint8)


def resize_for_cnn(gray_uint8: np.ndarray, size: int = 48) -> np.ndarray:
    """Resize a normalized grayscale image to the CNN's expected (size, size) input."""
    img = Image.fromarray(gray_uint8)
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)


class AddGaussianNoise:
    """Adds i.i.d. Gaussian noise to an already-normalized tensor. Training-only —
    used at the end of TRAIN_TRANSFORM, never in EVAL_TRANSFORM.
    """

    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std


# Applied to the 48x48 normalized grayscale PIL image (post resize_for_cnn), training
# split only. Geometric/photometric augmentation first (operates on the PIL image), then
# ToTensor+Normalize to match EVAL_TRANSFORM's distribution, then tensor-space augmentation
# (RandomErasing, Gaussian noise) that only makes sense post-tensor-conversion.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=12),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    AddGaussianNoise(std=0.02),
])

# No augmentation — used for validation, test, and live inference, so the model always
# sees the same normalization at eval time that it saw for the non-augmented half of training.
EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
