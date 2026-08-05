"""Recompute engagement labels in the existing data/labels/engagement_labels.json cache
using the current synthesize_engagement_label() rule, WITHOUT re-running face detection.

Proxy vectors ([N_proxy, C_proxy]) are unaffected by a change to the label-synthesis
rule — only the proxy -> label mapping changes — so this just re-derives labels from the
already-cached proxies. Far faster than rerunning scripts/precompute_labels_cache.py
(which re-does the ~15-minute 3-step detection cascade over the full dataset).

Usage:
    python scripts/relabel_cache.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.roi.labeling import CLASS_NAMES, synthesize_engagement_label

CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "labels" / "engagement_labels.json"


def main(cache_path: Path = CACHE_PATH) -> None:
    with open(cache_path) as f:
        records = json.load(f)

    label_counts = Counter()
    for r in records:
        r["label"] = synthesize_engagement_label(r["proxy"][0], r["proxy"][1])
        label_counts[CLASS_NAMES[r["label"]]] += 1

    with open(cache_path, "w") as f:
        json.dump(records, f)

    total = len(records)
    print(f"Relabeled {total} cached records using the current synthesize_engagement_label() rule.")
    print("\nLabel distribution:")
    for name in CLASS_NAMES:
        count = label_counts.get(name, 0)
        print(f"  {name}: {count} ({count / total:.1%})")


if __name__ == "__main__":
    main()
