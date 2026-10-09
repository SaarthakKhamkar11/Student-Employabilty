"""Download the team's labels from the labelling page into data/benchmark/ for the evaluation engine.

    .venv/bin/python scripts/export_labels.py
    .venv/bin/python -m src.embeddings.model_evaluation_engine --labels data/benchmark/human_labels.csv

Writes human_labels.csv (student_id, job_id, relevance_label, label_source, split) and
human_labels.manifest.json (provenance + agreement, read by the circular-evaluation check).
Only students whose 12 jobs are all graded are exported.
"""

import json
import os
import sys
import urllib.request

API = os.environ.get("API", "http://127.0.0.1:8000")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "benchmark")


def fetch(path: str) -> bytes:
    with urllib.request.urlopen(API + path, timeout=30) as resp:
        return resp.read()


if __name__ == "__main__":
    manifest = json.loads(fetch("/labelling/api/export.manifest.json"))
    if manifest["pairs"] == 0:
        sys.exit("No fully graded students yet: nothing to export.")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "human_labels.csv"), "wb") as fh:
        fh.write(fetch("/labelling/api/export.csv"))
    with open(os.path.join(OUT, "human_labels.manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    agreement = manifest["agreement"]
    print(f"exported {manifest['pairs']} pairs from {manifest['students_exported']} students "
          f"({manifest['students_incomplete']} not finished yet), {manifest['labellers']} labeller(s)")
    print(f"agreement on {agreement['pairs_with_two_or_more_labels']} double-graded pairs: exact "
          f"{agreement['exact_agreement']}, quadratic-weighted kappa {agreement['quadratic_weighted_kappa']}")
