"""Demonstration of steps 1-5 (normalization bounds + the 4 scoring modules) on a real
FLIR E8 capture — NOT wired into inference.py/dashboard yet (steps 6-7, deferred pending
review of this output, per the reframe doc).

Usage:
    python scripts/demo_scoring_flir3643.py
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
from src.roi.extraction import CascadeLandmarkPipeline, compute_ambient_temperature, extract_roi_temperatures
from src.roi.labeling import ambient_normalized_roi_temps, differential_index
from src.scoring.cognitive_load_index import compute_cognitive_load_index
from src.scoring.confidence_score import compute_confidence_score, load_population_mean_features
from src.scoring.stress_index import compute_stress_index
from src.scoring.wellness_score import compute_wellness_score

IMAGE_PATH = "data/raw/FLIR_E8_collected/FLIR3643.jpg"


def main():
    from flirimageextractor import FlirImageExtractor

    fie = FlirImageExtractor()
    fie.process_image(IMAGE_PATH)
    temp_c = fie.get_thermal_np().astype(np.float32)
    gray_full_res = normalize_to_grayscale(temp_c)

    pipeline = CascadeLandmarkPipeline()
    result = pipeline.detect(gray_full_res)
    pipeline.close()
    if not result["success"]:
        raise RuntimeError("No face detected.")

    roi_temps = extract_roi_temperatures(temp_c, result["landmarks"])
    ambient_temp = compute_ambient_temperature(temp_c, result["bbox"])
    normalized = ambient_normalized_roi_temps(roi_temps, ambient_temp)
    differential = normalized["nose_tip"] - normalized["periorbital"]

    stress_index = compute_stress_index(differential)
    cognitive_load_index = compute_cognitive_load_index(normalized["forehead"])
    wellness_score = compute_wellness_score(stress_index, cognitive_load_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_frozen_backbone(device=device)
    gray_48 = resize_for_cnn(gray_full_res, 48)
    image_tensor = EVAL_TRANSFORM(Image.fromarray(gray_48)).unsqueeze(0).to(device)
    with torch.no_grad():
        current_features = backbone(image_tensor)[0].cpu().numpy()
    population_mean_features = load_population_mean_features()
    confidence = compute_confidence_score(current_features, population_mean_features)

    output = {
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
        "detector_used": result["detector_used"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
