"""Ranking metrics for student -> job retrieval, computed on judged candidates only.

Each student's judged jobs are ranked by a model's score and compared with the graded labels.
Ties are broken by job_id, so a tie can't favour any model. Only judged jobs are ranked
("condensed lists", Sakai 2007): an unjudged job never counts as irrelevant, so a model that
did not help build the labelling pool is not penalised for retrieving unjudged jobs.

Definitions (K configurable; defaults 5 and 10):
  relevant      label >= threshold (default 2: moderate or strong match)
  Precision@K   relevant jobs in the top K / K
  Recall@K      relevant jobs in the top K / relevant jobs judged for the student
  Hit Rate@K    1 if any relevant job is in the top K, else 0
  MRR           1 / rank of the first relevant job
  NDCG@K        DCG@K / ideal DCG@K, gain 2^label - 1, discount 1 / log2(rank + 1)
  Kendall tau   rank correlation of scores with the graded labels, per student, then averaged
A student with no relevant judged job carries no signal for these metrics: excluded and counted.
Every metric is a mean over students; per-student values are kept for confidence intervals.
"""

import math

import numpy as np
import pandas as pd

NEG_INF = float("-inf")  # a pair a model could not score (an entity with no text) ranks last


def rank(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda job: (-scores[job], job))


def _dcg(grades) -> float:
    return sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(grades, start=1))


def query_metrics(ranked: list[str], labels: dict[str, int], ks=(5, 10), threshold: int = 2) -> dict | None:
    relevant = sum(1 for g in labels.values() if g >= threshold)
    if relevant == 0:
        return None
    grades = [labels[j] for j in ranked]
    first = next((i for i, g in enumerate(grades, start=1) if g >= threshold), None)
    out = {"mrr": 1.0 / first if first else 0.0}
    ideal = sorted(labels.values(), reverse=True)
    for k in ks:
        hits = sum(1 for g in grades[:k] if g >= threshold)
        out[f"precision@{k}"] = hits / k
        out[f"recall@{k}"] = hits / relevant
        out[f"hit_rate@{k}"] = 1.0 if hits else 0.0
        out[f"ndcg@{k}"] = _dcg(grades[:k]) / _dcg(ideal[:k])
    return out


def mean_kendall_tau(labels: pd.DataFrame, scores: dict[tuple[str, str], float]) -> tuple[float | None, dict]:
    """Mean per-student Kendall tau-b between model scores and graded labels (rank-based, so raw
    cosine magnitudes, which differ between model families, do not matter)."""
    from scipy.stats import kendalltau

    per_student = {}
    for sid, group in labels.groupby("student_id"):
        y = group["relevance_label"].to_numpy()
        s = np.array([scores.get((sid, j), np.nan) for j in group["job_id"]], dtype=float)
        keep = np.isfinite(s)
        if keep.sum() < 3 or len(set(y[keep])) < 2 or len(set(s[keep])) < 2:
            continue
        tau = kendalltau(s[keep], y[keep]).statistic
        if not np.isnan(tau):
            per_student[sid] = float(tau)
    return (float(np.mean(list(per_student.values()))) if per_student else None), per_student


def evaluate(scores: dict[tuple[str, str], float], labels: pd.DataFrame, ks=(5, 10), threshold: int = 2) -> dict:
    """labels: student_id, job_id, relevance_label (one label source, evaluation splits only)."""
    per_student, excluded, pool_sizes, unscored = {}, [], [], 0
    for sid, group in labels.groupby("student_id"):
        judged = dict(zip(group["job_id"], group["relevance_label"].astype(int)))
        pool_sizes.append(len(judged))
        model = {}
        for job in judged:
            value = scores.get((sid, job))
            if value is None or not np.isfinite(value):
                unscored += 1
                value = NEG_INF
            model[job] = value
        metrics = query_metrics(rank(model), judged, ks, threshold)
        if metrics is None:
            excluded.append(sid)
        else:
            per_student[sid] = metrics
    names = sorted({m for v in per_student.values() for m in v})
    tau, tau_per_student = mean_kendall_tau(labels, scores)
    return {
        "mean": {m: float(np.mean([v[m] for v in per_student.values()])) for m in names} | {"kendall_tau": tau},
        "per_student": per_student,
        "kendall_tau_per_student": tau_per_student,
        "students_evaluated": len(per_student),
        "students_excluded_no_relevant": len(excluded),
        "pairs": int(len(labels)),
        "unscored_pairs": unscored,
        "pool_size": {"min": min(pool_sizes, default=0), "max": max(pool_sizes, default=0)},
        # K >= a student's candidate count: Precision@K and Recall@K can't separate models there
        "degenerate_k": [k for k in ks if pool_sizes and min(pool_sizes) <= k],
    }


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float] | None:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, (n_boot, arr.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_difference(a: dict[str, float], b: dict[str, float], n_boot: int = 2000, seed: int = 42) -> dict | None:
    """Mean of a - b over the students both models were evaluated on, with a paired bootstrap 95% CI.
    'distinguishable' is False when the interval contains 0."""
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    diffs = [a[s] - b[s] for s in common]
    ci = bootstrap_ci(diffs, n_boot, seed)
    return {"mean_difference": float(np.mean(diffs)), "ci95": ci, "students": len(common),
            "distinguishable": bool(ci and (ci[0] > 0 or ci[1] < 0))}
