"""Relevance-label loader with the Phase 6 hard rules (spec §8, §26).

Labels CSV:  student_id, job_id, relevance_label, label_source, split
             0 = irrelevant, 1 = weak match, 2 = moderate match, 3 = strong match.
Provenance:  <labels>.manifest.json next to the CSV, written by whatever produced the grades
             (the labelling page export, src/evaluation/weak_labels.py, ...). It records who or
             what assigned the grades. Without it, independence from the evaluated models can't
             be confirmed, so the circular-evaluation check reports LOW validity.

load_labels() fails loudly: a single LabelValidationError lists every invalid row.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field

import pandas as pd

from src.evaluation.retrieval_metrics import mean_kendall_tau
from src.evaluation.splits import SPLITS, split_for

LABEL_SOURCES = ("human_reviewed", "weakly_labelled", "synthetic_clearly_marked")  # spec §8 order of preference
REQUIRED_COLUMNS = ("student_id", "job_id", "relevance_label", "label_source", "split")
GRADES = (0, 1, 2, 3)
STAMPS = {
    "synthetic_clearly_marked": "SYNTHETIC LABELS — NOT REAL GROUND TRUTH",
    "weakly_labelled": "WEAK LABELS — RULE-BASED, NOT HUMAN GROUND TRUTH",
}
# Mean per-student Kendall tau at or above this means the labels look like a function of a
# model's scores. Independent human grades of semantic fit do not track one model this closely.
LEAKAGE_TAU = 0.9


class LabelValidationError(ValueError):
    pass


@dataclass
class LabelSet:
    path: str
    frame: pd.DataFrame
    manifest: dict | None
    sha256: str
    warnings: list[str] = field(default_factory=list)

    def source_breakdown(self) -> dict[str, int]:
        return {s: int(n) for s, n in self.frame["label_source"].value_counts().items()}

    def split_breakdown(self) -> dict[str, dict[str, int]]:
        g = self.frame.groupby("split")
        return {s: {"pairs": int(len(d)), "students": int(d["student_id"].nunique())} for s, d in g}

    def label_distribution(self) -> dict[int, int]:
        counts = self.frame["relevance_label"].value_counts()
        return {int(k): int(counts.get(k, 0)) for k in GRADES}

    def preferred_source(self) -> str:
        return next(s for s in LABEL_SOURCES if s in set(self.frame["label_source"]))

    def select(self, source: str, splits) -> pd.DataFrame:
        f = self.frame
        return f[(f["label_source"] == source) & f["split"].isin(list(splits))]


def manifest_path(labels_path: str) -> str:
    root, _ = os.path.splitext(labels_path)
    return root + ".manifest.json"


def _fail(problems: list[str], path: str) -> None:
    shown = "\n  ".join(problems[:20])
    more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
    raise LabelValidationError(f"{path}: {len(problems)} invalid label row(s)\n  {shown}{more}")


def load_labels(path: str, students_csv: str, jobs_csv: str, extra_job_ids=()) -> LabelSet:
    """Read and validate a labels CSV. Any invalid row stops the load."""
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise LabelValidationError(f"{path}: missing required columns {missing}")
    if df.empty:
        raise LabelValidationError(f"{path}: no label rows")
    students = set(pd.read_csv(students_csv, dtype=str, keep_default_na=False, usecols=["student_id"])["student_id"])
    jobs = set(pd.read_csv(jobs_csv, dtype=str, keep_default_na=False, usecols=["job_id"])["job_id"]) | set(extra_job_ids)

    problems = []
    for i, row in enumerate(df.itertuples(index=False), start=2):  # line 1 is the header
        where = f"line {i} ({row.student_id}, {row.job_id})"
        if row.student_id not in students:
            problems.append(f"{where}: student_id not in students_unified.csv")
        if row.job_id not in jobs:
            problems.append(f"{where}: job_id not in jobs.csv or the jobs table")
        if row.relevance_label.strip() not in {str(g) for g in GRADES}:
            problems.append(f"{where}: relevance_label {row.relevance_label!r} is not one of 0, 1, 2, 3")
        if row.label_source not in LABEL_SOURCES:
            problems.append(f"{where}: label_source {row.label_source!r} is not one of {list(LABEL_SOURCES)}")
        if row.split not in SPLITS:
            problems.append(f"{where}: split {row.split!r} is not one of {list(SPLITS)}")
        elif row.student_id in students and row.split != split_for(row.student_id):
            problems.append(f"{where}: split {row.split!r} differs from the student's fixed split "
                            f"{split_for(row.student_id)!r} (src/evaluation/splits.py)")
    dupes = df[df.duplicated(["student_id", "job_id", "label_source"], keep=False)]
    for (s, j), _ in dupes.groupby(["student_id", "job_id"]):
        problems.append(f"({s}, {j}): duplicate rows for the same pair and label_source")
    if problems:
        _fail(problems, path)

    df["relevance_label"] = df["relevance_label"].str.strip().astype(int)
    manifest = None
    mpath = manifest_path(path)
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as fh:
            manifest = json.load(fh)
        declared = manifest.get("label_source")
        if declared and set(df["label_source"]) != {declared}:
            raise LabelValidationError(f"{mpath}: declares label_source {declared!r} but the CSV contains "
                                       f"{sorted(set(df['label_source']))}")
    return LabelSet(path=path, frame=df, manifest=manifest, sha256=digest)


def circularity_check(labelset: LabelSet, source: str, evaluated_models: list[str],
                      model_scores: dict[str, dict[tuple[str, str], float]] | None = None) -> dict:
    """Spec §8 hard rule: the labels must not come from any model being evaluated.

    Checks the provenance manifest and, when model scores are given, whether any evaluated
    model's scores reproduce the labels almost exactly (mean per-student Kendall tau).
    """
    reasons, low = [], False
    manifest = labelset.manifest
    if manifest is None:
        low = True
        reasons.append("no provenance manifest: independence from the evaluated models cannot be confirmed")
    else:
        used = set(manifest.get("models_used_for_labels", []))
        generated_by = str(manifest.get("generated_by", ""))
        if generated_by.startswith("model:"):
            used.add(generated_by.split(":", 1)[1])
        overlap = sorted(used & set(evaluated_models))
        if overlap:
            low = True
            reasons.append(f"labels were generated with evaluated model(s) {overlap}")
        if "models_used_for_labels" not in manifest:
            low = True
            reasons.append("manifest does not state which models (if any) produced the grades")
    taus = {}
    if model_scores:
        subset = labelset.frame[labelset.frame["label_source"] == source]
        for name, scores in model_scores.items():
            taus[name] = mean_kendall_tau(subset, scores)[0]
            if taus[name] is not None and taus[name] >= LEAKAGE_TAU:
                low = True
                reasons.append(f"{name}'s scores reproduce the labels (mean per-student Kendall tau "
                               f"{taus[name]:.2f} >= {LEAKAGE_TAU}): suspected model-generated labels")

    if low:
        line = "evaluation validity: LOW — labels not independently sourced"
        level = "LOW"
    elif source == "synthetic_clearly_marked":
        line, level = "evaluation validity: LOW — synthetic labels, not real ground truth", "LOW"
    elif source == "weakly_labelled":
        line = "evaluation validity: LIMITED — weak rule-based labels, independent of the evaluated models"
        level = "LIMITED"
    else:
        line, level = "evaluation validity: OK — human labels, independent of the evaluated models", "OK"
    return {"level": level, "line": line, "reasons": reasons,
            "generated_by": (manifest or {}).get("generated_by", "unknown"),
            "pool_models": (manifest or {}).get("pool", {}).get("models", []),
            "mean_per_student_kendall_tau": {k: (round(v, 3) if v is not None else None) for k, v in taus.items()}}
