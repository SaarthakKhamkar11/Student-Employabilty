"""One fixed per-student split shared by every benchmark (spec §7, §25, §26).

A student's split is a pure function of a salt and their student_id (a SHA-256 bucket), so:
  * every model and every label set sees the same split: no model gets a different test set;
  * adding students never moves an existing student into another split;
  * a student can never sit in two splits (the "same student across splits" leakage in spec §26).

Students are the queries; jobs are the shared corpus being ranked, so jobs are not split.

Usage:  python -m src.evaluation.splits     (writes data/benchmark/student_splits.csv)
"""

import hashlib
import os

import pandas as pd

SPLIT_VERSION = "student-sha256-v1"
SALT = "phase6-relevance-benchmark"
RATIOS = (("train", 0.6), ("validation", 0.2), ("test", 0.2))
SPLITS = tuple(name for name, _ in RATIOS)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCHMARK_DIR = os.path.join(ROOT, "data", "benchmark")
DEFAULT_STUDENTS = os.path.join(ROOT, "data", "processed", "students_unified.csv")


def split_for(student_id: str, salt: str = SALT, ratios=RATIOS) -> str:
    bucket = int.from_bytes(hashlib.sha256(f"{salt}:{student_id}".encode()).digest()[:8], "big") / 2**64
    edge = 0.0
    for name, share in ratios:
        edge += share
        if bucket < edge:
            return name
    return ratios[-1][0]


def assign_splits(student_ids) -> pd.DataFrame:
    ids = sorted(set(student_ids))
    return pd.DataFrame({"student_id": ids, "split": [split_for(s) for s in ids]})


if __name__ == "__main__":
    students = pd.read_csv(DEFAULT_STUDENTS, dtype=str, keep_default_na=False)
    table = assign_splits(students["student_id"])
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    out = os.path.join(BENCHMARK_DIR, "student_splits.csv")
    table.to_csv(out, index=False)
    merged = table.merge(students[["student_id", "source_dataset"]], on="student_id")
    print(f"{SPLIT_VERSION}: {table['split'].value_counts().to_dict()} -> {out}")
    print(merged.groupby(["source_dataset", "split"]).size().unstack(fill_value=0).to_string())
