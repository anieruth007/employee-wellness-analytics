"""Single-image inference: one FLIR E8 thermal capture -> engagement classification,
wellness score, N/C personality proxy, and a natural-language explanation.

Usage:
    python inference.py path/to/flir_capture.jpg
"""
import argparse

import numpy as np
import torch
import yaml
from PIL import Image

from src.data.preprocessing import EVAL_TRANSFORM, normalize_to_grayscale, resize_for_cnn
from src.models.fusion_model import EngagementModel, FusionClassifier
from src.models.thermal_cnn import ThermalCNNEncoder
from src.roi.extraction import CascadeLandmarkPipeline, extract_roi_temperatures
from src.roi.labeling import CLASS_NAMES, DEFAULT_BASELINE_PATH, ProxyThresholds, load_baseline, proxy_vector


def extract_flir_temperature(image_path: str) -> np.ndarray:
    """Extract the raw per-pixel Celsius temperature array from a FLIR E8 radiometric JPEG.

    NOTE: verify `process_image`/`get_thermal_np` against your installed flirimageextractor
    version's README — method names have varied slightly across releases of this package.
    """
    from flirimageextractor import FlirImageExtractor  # heavy optional dep, lazy import

    fie = FlirImageExtractor()
    fie.process_image(image_path)
    return fie.get_thermal_np().astype(np.float32)


def load_model(cnn_cfg: dict, fusion_cfg: dict, checkpoint_path: str, device: torch.device) -> EngagementModel:
    encoder = ThermalCNNEncoder(feature_dim=cnn_cfg["model"]["feature_dim"])
    classifier = FusionClassifier(
        expression_dim=fusion_cfg["model"]["expression_dim"],
        hidden_dims=tuple(fusion_cfg["model"]["hidden_dims"]),
        num_classes=fusion_cfg["model"]["num_classes"],
        dropout=fusion_cfg["model"]["dropout"],
    )
    model = EngagementModel(encoder, classifier).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def explain(label: str, n_proxy: float, c_proxy: float) -> str:
    n_desc = "Elevated stress markers detected (High N proxy)" if n_proxy >= 0.5 else "Stable physiological markers (Low N proxy)"
    c_desc = "reduced cognitive engagement (Low C proxy)" if c_proxy < 0.5 else "sustained cognitive engagement (High C proxy)"
    return f"{n_desc} with {c_desc} — indicating {label.lower()} classification."


def run_inference(image_path: str, checkpoint_path: str = "checkpoints/fusion/best_model.pt", input_size: int = 48) -> dict:
    with open("configs/cnn_config.yaml") as f:
        cnn_cfg = yaml.safe_load(f)
    with open("configs/fusion_config.yaml") as f:
        fusion_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cnn_cfg, fusion_cfg, checkpoint_path, device)

    temp_c = extract_flir_temperature(image_path)
    gray_full_res = normalize_to_grayscale(temp_c)

    pipeline = CascadeLandmarkPipeline()
    result = pipeline.detect(gray_full_res)
    pipeline.close()
    if not result["success"]:
        raise RuntimeError("No face detected in the captured thermal image (all cascade stages failed) — retake the shot.")

    thresholds = ProxyThresholds.from_config()
    baseline = load_baseline(DEFAULT_BASELINE_PATH)

    roi_temps = extract_roi_temperatures(temp_c, result["landmarks"])
    proxy = proxy_vector(roi_temps, baseline, thresholds)

    gray_48 = resize_for_cnn(gray_full_res, input_size)
    # Same EVAL_TRANSFORM (ToTensor + Normalize, no augmentation) used for val/test during
    # training — must match exactly, or the model sees a different input distribution here
    # than it was trained/validated on.
    image_tensor = EVAL_TRANSFORM(Image.fromarray(gray_48, mode="L")).unsqueeze(0).to(device)

    with torch.no_grad():
        expression_vec = model.cnn_encoder(image_tensor)
        probs = model.fusion_classifier.predict_proba(expression_vec)[0]

    pred_idx = int(probs.argmax())
    label = CLASS_NAMES[pred_idx]
    # P(Disengaged) only — "Burned Out" is also a concerning state this doesn't capture.
    # Left as-is (matches the original single-scalar wellness score design); revisit if a
    # wellness score that reflects both concerning classes is wanted.
    wellness_score = float(probs[0])

    return {
        "engagement": label,
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "wellness_score": wellness_score,
        "n_proxy": proxy[0],
        "c_proxy": proxy[1],
        "roi_temps_c": roi_temps,
        "explanation": explain(label, proxy[0], proxy[1]),
        "detector_used": result["detector_used"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to a FLIR E8 radiometric JPEG")
    parser.add_argument("--checkpoint", default="checkpoints/fusion/best_model.pt")
    args = parser.parse_args()

    result = run_inference(args.image_path, args.checkpoint)
    for k, v in result.items():
        print(f"{k}: {v}")
