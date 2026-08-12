"""Pipeline 2 — facial landmark detection + ROI temperature extraction on thermal images.

Landmark detection runs on the normalized grayscale thermal image at its ORIGINAL
captured resolution (not the 48x48 image resized for the CNN branch) — 48x48 is too
small for any of these detectors to localize a face reliably.

3-step detection cascade (MediaPipe alone misses ~55% of thermal frames — trained on
visible-spectrum faces, not thermal-as-grayscale):
  1. MediaPipe FaceLandmarker on a CLAHE-enhanced image -> full 468-point landmark set
  2. OpenCV Haar Cascade (frontalface_default) -> bounding box -> estimated landmarks
  3. OpenCV LBP Cascade (frontalface_improved) -> bounding box -> estimated landmarks
  else: undetected, caller should skip the sample

(haarcascade_frontalface_alt_tree was tried and dropped — on a 100-image smoke test it
contributed zero additional detections beyond what MediaPipe + Haar default already
caught, so it wasn't worth its cost.)

Both landmark sources (MediaPipe indices, bbox-based estimation) are normalized into a
common `roi_points` representation — {roi_name: [(x, y), ...]} — so downstream ROI
temperature extraction doesn't need to know which detector produced them.
"""
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from src.data.preprocessing import apply_clahe

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "roi_thresholds.yaml"

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

CASCADE_DIR = Path(__file__).resolve().parents[2] / "models" / "cascades"
CASCADE_BASE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data"
CASCADE_FILES = {
    "haarcascade_frontalface_default.xml": f"{CASCADE_BASE_URL}/haarcascades/haarcascade_frontalface_default.xml",
    "lbpcascade_frontalface_improved.xml": f"{CASCADE_BASE_URL}/lbpcascades/lbpcascade_frontalface_improved.xml",
}


def load_roi_landmarks(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return {name: spec["landmarks"] for name, spec in config["rois"].items()}


def _ensure_model_downloaded(path: Path = MODEL_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def _ensure_cascade_downloaded(filename: str) -> Path:
    path = CASCADE_DIR / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CASCADE_FILES[filename], path)
    return path


class LandmarkDetector:
    """Thin wrapper around MediaPipe FaceLandmarker (Tasks API).

    mediapipe>=0.10 dropped the legacy `mp.solutions.face_mesh` API from its published
    wheels, so this uses `mp.tasks.vision.FaceLandmarker` instead, auto-downloading the
    model asset to models/face_landmarker.task on first use. Same underlying 468-point
    canonical face mesh topology, so landmark indices in configs/roi_thresholds.yaml are
    unaffected by the API switch.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        min_detection_confidence: float = 0.1,
        min_tracking_confidence: float = 0.1,
    ):
        import mediapipe as mp

        model_path = _ensure_model_downloaded(model_path)
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            # min_tracking_confidence only affects VIDEO/LIVE_STREAM mode; inert in IMAGE mode.
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """gray_image: (H, W) uint8. Returns (468, 2) pixel coordinates, or None if no face found."""
        rgb = np.repeat(gray_image[..., None], 3, axis=2)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        h, w = gray_image.shape
        landmarks = result.face_landmarks[0]
        return np.array([[lm.x * w, lm.y * h] for lm in landmarks])

    def close(self):
        self._landmarker.close()


class CascadeFaceDetector:
    """OpenCV Haar/LBP cascade fallback chain (cascade Steps 2-3), tried in order when
    MediaPipe fails to detect a face in a thermal image. Each stage's XML is auto-downloaded
    from the official OpenCV GitHub repo on first use, since the installed opencv-python
    build doesn't bundle cascade data files.
    """

    _STAGES = [
        ("haar_default", "haarcascade_frontalface_default.xml", dict(scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))),
        ("lbp", "lbpcascade_frontalface_improved.xml", dict(scaleFactor=1.05, minNeighbors=2)),
    ]

    def __init__(self):
        self._stages = [
            (name, cv2.CascadeClassifier(str(_ensure_cascade_downloaded(filename))), kwargs)
            for name, filename, kwargs in self._STAGES
        ]

    def detect(self, gray_image: np.ndarray) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]], float]:
        """Returns (detector_name, (x, y, w, h), confidence) for the first stage that finds
        a face (largest detection if multiple), or (None, None, 0.0) if all stages fail.
        `confidence` is OpenCV's cascade level-weight for the winning detection — a rough,
        detector-internal score, not comparable across detector types.
        """
        for name, classifier, kwargs in self._stages:
            faces, _, level_weights = classifier.detectMultiScale3(
                gray_image, outputRejectLevels=True, **kwargs
            )
            if len(faces) > 0:
                best = int(np.argmax(level_weights))
                x, y, w, h = faces[best]
                return name, (int(x), int(y), int(w), int(h)), float(level_weights[best])
        return None, None, 0.0

    def close(self):
        pass  # cv2.CascadeClassifier has no explicit teardown


def estimate_landmarks_from_bbox(x: float, y: float, w: float, h: float) -> Dict[str, Tuple[float, float]]:
    """Estimate key ROI landmark positions from a face bounding box, as proportional
    coordinates within the face region. Used for cascade Steps 2-4, where OpenCV only
    gives a bounding box, not per-point landmarks.
    """
    return {
        "nose_tip": (x + w * 0.50, y + h * 0.55),
        "forehead": (x + w * 0.50, y + h * 0.15),
        "periorbital_l": (x + w * 0.30, y + h * 0.38),
        "periorbital_r": (x + w * 0.70, y + h * 0.38),
        "upper_lip": (x + w * 0.50, y + h * 0.72),
    }


def _bbox_to_roi_points(bbox: Tuple[int, int, int, int]) -> Dict[str, List[Tuple[float, float]]]:
    x, y, w, h = bbox
    pts = estimate_landmarks_from_bbox(x, y, w, h)
    return {
        "nose_tip": [pts["nose_tip"]],
        "forehead": [pts["forehead"]],
        "periorbital": [pts["periorbital_l"], pts["periorbital_r"]],
        "upper_lip": [pts["upper_lip"]],
    }


def _mediapipe_landmarks_to_roi_points(
    landmark_coords: np.ndarray, roi_landmarks: dict
) -> Dict[str, List[Tuple[float, float]]]:
    return {
        roi_name: [tuple(landmark_coords[idx]) for idx in indices]
        for roi_name, indices in roi_landmarks.items()
    }


class CascadeLandmarkPipeline:
    """The full 3-step detection cascade. Owns one MediaPipe detector + one OpenCV cascade
    chain; `detect()` tries them in order and returns a uniform result dict:

        {
            "success": bool,
            "detector_used": "mediapipe" | "haar_default" | "lbp" | "none",
            "landmarks": {roi_name: [(x, y), ...]} or None,
            "confidence": float,
            "bbox": (x, y, w, h) or None,
        }

    `landmarks`, when present, is ready to pass straight to extract_roi_temperatures().
    """

    def __init__(self, min_detection_confidence: float = 0.1, min_tracking_confidence: float = 0.1):
        self._mediapipe = LandmarkDetector(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._cascade = CascadeFaceDetector()
        self._roi_landmarks = load_roi_landmarks()

    def detect(self, gray_image: np.ndarray) -> dict:
        # Step 1: MediaPipe on a CLAHE-enhanced copy.
        mp_landmarks = self._mediapipe.detect(apply_clahe(gray_image))
        if mp_landmarks is not None:
            # bbox from the full 468-point landmark set's extent (not just our 4 ROI
            # points) — needed for ambient-temperature background masking, which requires
            # a real face-covering box regardless of which detector found the face.
            x_min, y_min = mp_landmarks.min(axis=0)
            x_max, y_max = mp_landmarks.max(axis=0)
            bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
            return {
                "success": True,
                "detector_used": "mediapipe",
                "landmarks": _mediapipe_landmarks_to_roi_points(mp_landmarks, self._roi_landmarks),
                "confidence": 1.0,  # FaceLandmarker's Tasks API doesn't expose a scalar detection score
                "bbox": bbox,
            }

        # Steps 2-4: OpenCV Haar/LBP cascade chain on the plain (non-CLAHE) grayscale image.
        detector_name, bbox, confidence = self._cascade.detect(gray_image)
        if bbox is not None:
            return {
                "success": True,
                "detector_used": detector_name,
                "landmarks": _bbox_to_roi_points(bbox),
                "confidence": confidence,
                "bbox": bbox,
            }

        return {"success": False, "detector_used": "none", "landmarks": None, "confidence": 0.0, "bbox": None}

    def close(self):
        self._mediapipe.close()
        self._cascade.close()


def compute_ambient_temperature(
    temp_array_c: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.25,
) -> float:
    """Within-image ambient-temperature reference: mean temperature of background
    (non-face) pixels, used to normalize ROI temperatures against the specific room the
    photo was taken in, instead of a fixed population baseline that may have been
    captured at a different ambient temperature.

    1. Build a face mask from `bbox`, padded by `margin` on each side (a tight bbox would
       still leave face/hair pixels just outside it counted as "background").
    2. Invert it -> background/non-face pixel mask.
    3. Return the mean temperature of those background pixels.

    Falls back to the whole image's mean if the padded bbox covers it entirely (e.g. a
    tight close-up crop with no visible background) — rare, but division-by-empty-set
    otherwise.
    """
    h, w = temp_array_c.shape
    x, y, bw, bh = bbox
    pad_x, pad_y = int(bw * margin), int(bh * margin)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)

    mask = np.ones((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = False  # face region (padded) excluded
    background_pixels = temp_array_c[mask]

    if background_pixels.size == 0:
        return float(temp_array_c.mean())
    return float(background_pixels.mean())


def extract_roi_temperatures(
    temp_array_c: np.ndarray,
    roi_points: Dict[str, List[Tuple[float, float]]],
    patch_radius: int = 2,
) -> dict:
    """temp_array_c: (H, W) float32 calibrated Celsius temperatures, same resolution as
    the image roi_points was detected on.

    roi_points: {roi_name: [(x, y), ...]}, as produced by CascadeLandmarkPipeline.detect()
    (works identically whether the points came from MediaPipe indices or bbox estimation).

    Averages a small patch around each point (rather than a single pixel) to reduce
    sensitivity to sensor noise, then averages across all points in a region.
    Returns {roi_name: mean_temp_c}.
    """
    h, w = temp_array_c.shape
    roi_temps = {}
    for roi_name, points in roi_points.items():
        samples = []
        for px, py in points:
            x, y = int(round(px)), int(round(py))
            x0, x1 = max(0, x - patch_radius), min(w, x + patch_radius + 1)
            y0, y1 = max(0, y - patch_radius), min(h, y + patch_radius + 1)
            patch = temp_array_c[y0:y1, x0:x1]
            if patch.size:
                samples.append(float(patch.mean()))
        roi_temps[roi_name] = float(np.mean(samples)) if samples else float("nan")
    return roi_temps
