"""Pipeline 2 — facial landmark detection + ROI temperature extraction on thermal images.

Landmark detection runs on the normalized grayscale thermal image at its ORIGINAL
captured resolution (not the 48x48 image resized for the CNN branch) — 48x48 is too
small for MediaPipe FaceMesh to localize landmarks reliably.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "roi_thresholds.yaml"


def load_roi_landmarks(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return {name: spec["landmarks"] for name, spec in config["rois"].items()}


class LandmarkDetector:
    """Thin wrapper around MediaPipe FaceMesh, lazily imported since it's a heavy optional dep."""

    def __init__(self, min_detection_confidence: float = 0.5):
        import mediapipe as mp

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """gray_image: (H, W) uint8. Returns (468, 2) pixel coordinates, or None if no face found."""
        rgb = np.repeat(gray_image[..., None], 3, axis=2)
        result = self._face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        h, w = gray_image.shape
        landmarks = result.multi_face_landmarks[0].landmark
        return np.array([[lm.x * w, lm.y * h] for lm in landmarks])

    def close(self):
        self._face_mesh.close()


def extract_roi_temperatures(
    temp_array_c: np.ndarray,
    landmark_coords: np.ndarray,
    roi_landmarks: Optional[dict] = None,
    patch_radius: int = 2,
) -> dict:
    """temp_array_c: (H, W) float32 calibrated Celsius temperatures, same resolution as
    the image landmark_coords was detected on.

    Averages a small patch around each landmark (rather than a single pixel) to reduce
    sensitivity to sensor noise, then averages across all landmarks in a region.
    Returns {roi_name: mean_temp_c}.
    """
    if roi_landmarks is None:
        roi_landmarks = load_roi_landmarks()

    h, w = temp_array_c.shape
    roi_temps = {}
    for roi_name, indices in roi_landmarks.items():
        samples = []
        for idx in indices:
            x, y = landmark_coords[idx]
            x, y = int(round(x)), int(round(y))
            x0, x1 = max(0, x - patch_radius), min(w, x + patch_radius + 1)
            y0, y1 = max(0, y - patch_radius), min(h, y + patch_radius + 1)
            patch = temp_array_c[y0:y1, x0:x1]
            if patch.size:
                samples.append(float(patch.mean()))
        roi_temps[roi_name] = float(np.mean(samples)) if samples else float("nan")
    return roi_temps
