"""Shared thermal image preprocessing: calibration, grayscale normalization, CNN resize.

Used identically by training data loading (thermal_dataset.py) and live inference
(inference.py) so the model never sees a train/inference distribution mismatch.
"""
import numpy as np
from PIL import Image


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


def resize_for_cnn(gray_uint8: np.ndarray, size: int = 48) -> np.ndarray:
    """Resize a normalized grayscale image to the CNN's expected (size, size) input."""
    img = Image.fromarray(gray_uint8)
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)
