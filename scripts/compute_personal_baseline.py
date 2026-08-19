"""Compute a per-session personal resting baseline from a set of neutral/resting thermal
captures of one subject.

The production scoring pipeline (see project v2.0.docx, and the ThermaWell v2.0 reframe)
normalizes ROI temperatures against ambient (background) temperature, then scales against
P5/P95 bounds fit on the Charlotte-ThermalFace population. That population-level scaling
saturates on real FLIR E8 captures of a single subject (see
scripts/validate_collected_faces.py: neutral scored higher stress than the stressed
condition on a 16-image self-collected set) because ambient-normalized deltas for a real
subject sit in a much narrower, individually-shifted range than the population bounds
assume.

A personal baseline re-centers each subsequent capture against THIS subject's own resting
state, rather than the population — matching the original threshold-based design intent
(a fixed +-0.5 / +-0.3 degC cutoff only makes sense once deltas are re-centered near zero;
against raw ambient-normalized deltas of 5-15 degC that cutoff never triggers).

Usage:
    python scripts/compute_personal_baseline.py \
        --images data/raw/FLIR_E8_collected/neutral/FLIR4147.jpg data/raw/FLIR_E8_collected/neutral/FLIR4149.jpg ... \
        --session-name lab_session1

    # or, simpler, point it at a folder of resting-condition captures:
    python scripts/compute_personal_baseline.py --folder data/raw/FLIR_E8_collected/neutral --session-name lab_session1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import extract_temperature_array
from src.data.preprocessing import normalize_to_grayscale
from src.roi.extraction import CascadeLandmarkPipeline, compute_ambient_temperature, extract_roi_temperatures
from src.roi.labeling import ambient_normalized_roi_temps


def compute_normalized_deltas_for_image(image_path: str) -> dict:
    temp_c = extract_temperature_array(image_path)
    gray_full_res = normalize_to_grayscale(temp_c)

    pipeline = CascadeLandmarkPipeline()
    result = pipeline.detect(gray_full_res)
    pipeline.close()
    if not result["success"]:
        raise RuntimeError(f"No face detected in {image_path} — exclude it from the baseline set.")

    roi_temps = extract_roi_temperatures(temp_c, result["landmarks"])
    ambient_temp = compute_ambient_temperature(temp_c, result["bbox"])
    normalized = ambient_normalized_roi_temps(roi_temps, ambient_temp)
    differential = normalized["nose_tip"] - normalized["periorbital"]

    return {
        "nose_delta": normalized["nose_tip"],
        "forehead_delta": normalized["forehead"],
        "periorbital_delta": normalized["periorbital"],
        "differential": differential,
        "ambient_temp": ambient_temp,
    }


def compute_personal_baseline(image_paths: list) -> dict:
    per_image = [compute_normalized_deltas_for_image(p) for p in image_paths]
    n = len(per_image)

    baseline = {
        "nose_delta_mean": sum(r["nose_delta"] for r in per_image) / n,
        "forehead_delta_mean": sum(r["forehead_delta"] for r in per_image) / n,
        "periorbital_delta_mean": sum(r["periorbital_delta"] for r in per_image) / n,
        "differential_mean": sum(r["differential"] for r in per_image) / n,
        "ambient_temp_mean": sum(r["ambient_temp"] for r in per_image) / n,
        "n_images": n,
        "source_images": [Path(p).name for p in image_paths],
    }
    return baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="*", help="Explicit list of resting-condition image paths")
    parser.add_argument("--folder", help="Folder of resting-condition images (all .jpg inside used)")
    parser.add_argument("--session-name", required=True, help="e.g. lab_session1 -> data/labels/personal_baseline_lab_session1.json")
    args = parser.parse_args()

    if args.folder:
        image_paths = sorted(str(p) for p in Path(args.folder).glob("*.jpg"))
    elif args.images:
        image_paths = args.images
    else:
        parser.error("Provide either --folder or --images")

    print(f"Computing personal baseline from {len(image_paths)} images:")
    for p in image_paths:
        print(f"  {p}")

    baseline = compute_personal_baseline(image_paths)

    out_path = Path("data/labels") / f"personal_baseline_{args.session_name}.json"
    out_path.write_text(json.dumps(baseline, indent=2))

    print("\nBaseline:")
    print(json.dumps(baseline, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
