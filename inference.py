"""Single-image inference: one thermal capture -> Stress Index, Cognitive Load Index,
Wellness Score, and Measurement Confidence — all continuous 0-100 scores, no
classification labels (see project v2.0.docx: the project was reframed from an engagement
classifier to a continuous physiological monitoring system).

Accepts two source formats:
  - FLIR E8 radiometric JPEG (.jpg/.jpeg) -> via flirimageextractor (production path)
  - Charlotte-ThermalFace-style raw 16-bit TIFF (.tiff/.tif) -> direct calibration (test/
    demo path — useful before/without a physical FLIR E8 camera)

Usage:
    python inference.py path/to/flir_capture.jpg
    python inference.py path/to/sample.tiff
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.preprocessing import EVAL_TRANSFORM, normalize_to_grayscale, raw_to_celsius, resize_for_cnn
from src.models.resnet_backbone import load_frozen_backbone
from src.roi.extraction import CascadeLandmarkPipeline, compute_ambient_temperature, extract_roi_temperatures
from src.roi.labeling import ambient_normalized_roi_temps
from src.scoring.cognitive_load_index import compute_cognitive_load_index
from src.scoring.confidence_score import compute_confidence_score, load_population_mean_features
from src.scoring.stress_index import compute_stress_index
from src.scoring.wellness_score import compute_wellness_score

RESEARCH_GROUNDING = {
    "stress_basis": "Fernandez et al. (2024, Anxiety Stress & Coping) - nose tip temperature "
                     "as a stress index; Gioia et al. (2023, Sensors) - differential index as "
                     "an autonomic marker.",
    "cognitive_basis": "Forehead thermal elevation under cognitive load - Frontiers in "
                        "Psychiatry (2025).",
}


def extract_temperature_array(image_path: str) -> np.ndarray:
    """Raw per-pixel Celsius temperature array from a thermal image file.

    NOTE (FLIR path): verify `process_image`/`get_thermal_np` against your installed
    flirimageextractor version's README — method names have varied slightly across
    releases of this package.
    """
    suffix = Path(image_path).suffix.lower()
    if suffix in (".tiff", ".tif"):
        raw = np.array(Image.open(image_path))
        return raw_to_celsius(raw)

    from flirimageextractor import FlirImageExtractor  # heavy optional dep, lazy import

    fie = FlirImageExtractor()
    fie.process_image(image_path)
    return fie.get_thermal_np().astype(np.float32)


def _stress_level(stress_index: float) -> str:
    """Bin boundaries are a design choice (not specified in the reframe doc) — evenly
    split 0-100 into quartiles. Easy to retune once real capture data gives a feel for
    where meaningful cutoffs actually are.
    """
    if stress_index < 25:
        return "Low"
    if stress_index < 50:
        return "Moderate"
    if stress_index < 75:
        return "Elevated"
    return "High"


def _cognitive_state(cognitive_load_index: float) -> str:
    if cognitive_load_index < 25:
        return "Low"
    if cognitive_load_index < 50:
        return "Moderate"
    if cognitive_load_index < 75:
        return "High"
    return "Very High"


def _wellness_flag(wellness_score: float) -> str:
    """Higher wellness_score = better, per its formula — bins run the opposite direction
    from stress_level/cognitive_state.
    """
    if wellness_score >= 75:
        return "Good"
    if wellness_score >= 50:
        return "Moderate"
    if wellness_score >= 25:
        return "Needs attention"
    return "Alert"


def _recommendation(stress_level: str, cognitive_state: str) -> str:
    """Rule-based text combining stress_level and cognitive_state — mirrors the four
    corner cases the wellness_score formula itself was validated against.
    """
    high_stress = stress_level in ("Elevated", "High")
    high_cognitive = cognitive_state in ("High", "Very High")

    if high_stress and high_cognitive:
        return (f"Stress markers {stress_level.lower()} alongside {cognitive_state.lower()} "
                "cognitive load — pushing through despite strain. Consider a short break.")
    if high_stress:
        return f"Stress markers {stress_level.lower()}. Consider a short break."
    if high_cognitive:
        return "Calm and focused — a productive working state."
    return "Relaxed, low engagement signals. No action needed, but worth checking in if this persists."


def build_interpretation(stress_index: float, cognitive_load_index: float, wellness_score: float) -> dict:
    stress_level = _stress_level(stress_index)
    cognitive_state = _cognitive_state(cognitive_load_index)
    return {
        "stress_level": stress_level,
        "cognitive_state": cognitive_state,
        "wellness_flag": _wellness_flag(wellness_score),
        "recommendation": _recommendation(stress_level, cognitive_state),
    }


def run_inference(image_path: str) -> dict:
    """Returns the full scoring JSON (see project v2.0.docx's output schema) plus a few
    internal fields (prefixed `_`) carrying pixel data the dashboard needs for the
    annotated-image display — not part of the documented API, stripped by the CLI's
    `json.dumps` call below.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_frozen_backbone(device=device)

    temp_c = extract_temperature_array(image_path)
    gray_full_res = normalize_to_grayscale(temp_c)

    pipeline = CascadeLandmarkPipeline()
    result = pipeline.detect(gray_full_res)
    pipeline.close()
    if not result["success"]:
        raise RuntimeError("No face detected in the captured thermal image (all cascade stages failed) — retake the shot.")

    roi_temps = extract_roi_temperatures(temp_c, result["landmarks"])
    ambient_temp = compute_ambient_temperature(temp_c, result["bbox"])
    normalized = ambient_normalized_roi_temps(roi_temps, ambient_temp)
    differential = normalized["nose_tip"] - normalized["periorbital"]

    stress_index = compute_stress_index(normalized["nose_tip"], differential)
    cognitive_load_index = compute_cognitive_load_index(normalized["forehead"])
    wellness_score = compute_wellness_score(stress_index, cognitive_load_index)

    gray_48 = resize_for_cnn(gray_full_res, 48)
    image_tensor = EVAL_TRANSFORM(Image.fromarray(gray_48)).unsqueeze(0).to(device)
    with torch.no_grad():
        current_features = backbone(image_tensor)[0].cpu().numpy()
    population_mean_features = load_population_mean_features()
    confidence = compute_confidence_score(current_features, population_mean_features)

    return {
        "stress_index": round(stress_index, 1),
        "cognitive_load_index": round(cognitive_load_index, 1),
        "wellness_score": round(wellness_score, 1),
        "measurement_confidence": round(confidence, 1),
        "ambient_temp": round(ambient_temp, 1),
        "roi_temps": {k: round(v, 1) for k, v in roi_temps.items()},
        "normalized_deltas": {
            "nose_delta": round(normalized["nose_tip"], 1),
            "forehead_delta": round(normalized["forehead"], 1),
            "periorbital_delta": round(normalized["periorbital"], 1),
            "differential": round(differential, 1),
        },
        "interpretation": build_interpretation(stress_index, cognitive_load_index, wellness_score),
        "research_grounding": RESEARCH_GROUNDING,
        "detector_used": result["detector_used"],
        "_gray_image": gray_full_res,
        "_bbox": result["bbox"],
        "_landmarks": result["landmarks"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to a FLIR E8 radiometric JPEG or a raw 16-bit thermal TIFF")
    args = parser.parse_args()

    result = run_inference(args.image_path)
    public_result = {k: v for k, v in result.items() if not k.startswith("_")}
    print(json.dumps(public_result, indent=2))
