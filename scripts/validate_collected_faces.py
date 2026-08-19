"""Validate the scoring pipeline against a small self-collected FLIR E8 set with three
self-labeled conditions (neutral / engaged / stressed) in data/raw/FLIR_E8_collected/.

This is a sanity-check / calibration-diagnostic run, NOT a retraining step — there is no
trainable classifier left in the v2.0 pipeline (see project v2.0.docx). What this checks:
  - does every image get a face detected?
  - do the continuous scores move in a sane direction across the three self-labeled
    conditions (e.g. stressed sessions skewing higher stress_index than neutral)?
  - are scores saturating at the P5/P95 normalization bounds (a sign those bounds, fit on
    Charlotte-ThermalFace, may not transfer to this camera/room/subject)?

Usage:
    python scripts/validate_collected_faces.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import run_inference

ROOT = Path("data/raw/FLIR_E8_collected")
CONDITIONS = ["neutral", "engaged", "stressed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--personal-baseline", default=None,
                         help="Path to a data/labels/personal_baseline_*.json file to score in personal-baseline mode")
    parser.add_argument("--out", default="c:/tmp/validate_collected_faces.json")
    args = parser.parse_args()

    rows = []
    failures = []

    for condition in CONDITIONS:
        folder = ROOT / condition
        images = sorted(folder.glob("*.jpg"))
        for img_path in images:
            try:
                result = run_inference(str(img_path), personal_baseline_path=args.personal_baseline)
            except Exception as e:
                failures.append((condition, img_path.name, str(e)))
                continue

            row = {
                "condition": condition,
                "file": img_path.name,
                "stress_index": result["stress_index"],
                "cognitive_load_index": result["cognitive_load_index"],
                "wellness_score": result["wellness_score"],
                "measurement_confidence": result["measurement_confidence"],
                "ambient_temp": result["ambient_temp"],
                "nose_delta": result["normalized_deltas"]["nose_delta"],
                "forehead_delta": result["normalized_deltas"]["forehead_delta"],
                "differential": result["normalized_deltas"]["differential"],
                "detector_used": result["detector_used"],
            }
            log_line = (f"[{condition}] {img_path.name}: stress={result['stress_index']:.1f} "
                        f"cognitive={result['cognitive_load_index']:.1f} "
                        f"wellness={result['wellness_score']:.1f} "
                        f"conf={result['measurement_confidence']:.1f} "
                        f"ambient={result['ambient_temp']:.1f} "
                        f"detector={result['detector_used']}")

            if result.get("personal_baseline_used"):
                row["relative_nose"] = result["relative_deltas"]["relative_nose"]
                row["relative_forehead"] = result["relative_deltas"]["relative_forehead"]
                row["ambient_drift_from_baseline"] = result["ambient_drift_from_baseline"]
                row["ambient_drift_flag"] = result["ambient_drift_flag"]
                log_line += (f" | rel_nose={row['relative_nose']:+.2f} rel_forehead={row['relative_forehead']:+.2f} "
                             f"ambient_drift={row['ambient_drift_from_baseline']:+.2f}"
                             f"{' <<< DRIFT >1degC' if row['ambient_drift_flag'] else ''}")

            rows.append(row)
            print(log_line)

    print("\n=== Failures ===")
    if not failures:
        print("(none)")
    else:
        for condition, name, err in failures:
            print(f"[{condition}] {name}: {err}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"rows": rows, "failures": failures}, indent=2))
    print(f"\nSaved {len(rows)} results ({len(failures)} failures) to {out_path}")


if __name__ == "__main__":
    main()
