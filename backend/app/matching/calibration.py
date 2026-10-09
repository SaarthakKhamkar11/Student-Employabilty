"""Calibration of the similarity features for the configured embedding model (docs/adr/0014).

Raw cosine is model-specific (Phase 6), so each similarity feature is used as a percentile: where a
pair's value falls among the same similarity over a reference set (every processed student x every
canonical job, for that model and version). A percentile of 0.9 means "more similar than 90% of
student-job pairs". Mid-rank is used for ties.

The transferable-skill threshold is calibrated the same way: two skill names are related when their
similarity is in the top (100 - percentile)% of all pairs of distinct skills in the data AND at least
min_similarity (config/matching_baseline_v1.yaml `features.transferable`).

Build after any (re)embedding, on the host (it writes into data/benchmark, mounted read-only in Docker):
    DATABASE_URL=postgresql://<user>:<password>@127.0.0.1:5432/<db> python -m app.matching.calibration
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse

from app.core.config import REPO_ROOT
from app.embeddings.provider import EmbeddingProvider, Vector
from app.matching.entities import JobView, StudentView
from app.matching.similarity import RETRIEVAL_PAIRS, SIGNALS
from app.profile.skills import skill_key

CALIBRATION_DIR = REPO_ROOT / "data" / "benchmark" / "matching_calibration"
N_QUANTILES = 1001
FEATURES = ("semantic_similarity", *SIGNALS)


class CalibrationMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class Calibration:
    model_name: str
    model_version: str
    quantiles: dict[str, tuple[float, ...]]
    transferable_threshold: float
    reference_pairs: dict[str, int]

    def percentile(self, feature: str, value: float | None) -> float | None:
        if value is None:
            return None
        q = np.asarray(self.quantiles[feature])
        lo, hi = np.searchsorted(q, value, side="left"), np.searchsorted(q, value, side="right")
        return round(float((lo + hi) / 2 / len(q)), 4)


def path_for(model_name: str, model_version: str) -> Path:
    return CALIBRATION_DIR / f"{model_name}-{model_version}.json"


def load_calibration(model_name: str, model_version: str) -> Calibration:
    path = path_for(model_name, model_version)
    if not path.exists():
        raise CalibrationMissing(f"no matching calibration for {model_name} {model_version}: "
                                 "run python -m app.matching.calibration")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Calibration(model_name=data["model_name"], model_version=data["model_version"],
                       quantiles={k: tuple(v) for k, v in data["quantiles"].items()},
                       transferable_threshold=float(data["transferable"]["threshold"]),
                       reference_pairs=data["reference_pairs"])


@lru_cache(maxsize=4)
def cached_calibration(model_name: str, model_version: str) -> Calibration:
    return load_calibration(model_name, model_version)


def _matrix(provider: EmbeddingProvider, texts: list[str], dimension: int):
    """Rows of L2-normalised vectors (sparse for sparse models); all-zero rows mark "no text / no signal"."""
    keep = [i for i, t in enumerate(texts) if t and t.strip()]
    vectors = provider.embed([texts[i] for i in keep]) if keep else []
    rows, cols, vals, present = [], [], [], np.zeros(len(texts), dtype=bool)
    dense = None
    for i, v in zip(keep, vectors):
        if v is None:
            continue
        present[i] = True
        if v.indices is None:
            dense = dense if dense is not None else np.zeros((len(texts), dimension), dtype=np.float32)
            dense[i] = v.values
        else:
            rows += [i] * len(v.indices)
            cols += v.indices
            vals += v.values
    if dense is not None:
        return dense, present
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(texts), dimension), dtype=np.float32), present


def _cos(a, b) -> np.ndarray:
    product = a @ b.T
    return np.asarray(product.todense() if sparse.issparse(product) else product, dtype=np.float64)


def build_calibration(students: list[StudentView], jobs: list[JobView], provider: EmbeddingProvider,
                      transferable: dict, skill_names: list[str]) -> dict:
    version, dimension = provider.describe()
    s_keys = {s for s, _, _ in RETRIEVAL_PAIRS} | {s for s, _ in SIGNALS.values()}
    j_keys = {j for _, j, _ in RETRIEVAL_PAIRS} | {j for _, j in SIGNALS.values()}
    sm = {k: _matrix(provider, [s.texts.get(k, "") for s in students], dimension) for k in s_keys}
    jm = {k: _matrix(provider, [j.texts.get(k, "") for j in jobs], dimension) for k in j_keys}

    values: dict[str, np.ndarray] = {}
    num = np.zeros((len(students), len(jobs)))
    den = np.zeros_like(num)
    for s_sec, j_sec, weight in RETRIEVAL_PAIRS:
        (a, pa), (b, pb) = sm[s_sec], jm[j_sec]
        both = np.outer(pa, pb)
        num += np.where(both, weight * _cos(a, b), 0.0)
        den += np.where(both, weight, 0.0)
    values["semantic_similarity"] = (num[den > 0] / den[den > 0])
    for name, (s_key, j_key) in SIGNALS.items():
        (a, pa), (b, pb) = sm[s_key], jm[j_key]
        values[name] = _cos(a, b)[np.outer(pa, pb)]

    grid = np.linspace(0, 1, N_QUANTILES)
    quantiles = {k: np.quantile(v, grid).round(6).tolist() if v.size else [0.0] * N_QUANTILES
                 for k, v in values.items()}

    # transferable threshold over all pairs of distinct skill names in the data
    names = sorted({skill_key(n): n for n in skill_names if skill_key(n)}.values())
    mat, present = _matrix(provider, names, dimension)
    iu = np.triu_indices(len(names), k=1)
    pair_sims = _cos(mat, mat)[iu]
    mask = present[iu[0]] & present[iu[1]]
    pair_values = pair_sims[mask]
    pct = float(transferable["percentile"])
    threshold = max(float(np.percentile(pair_values, pct)) if pair_values.size else 1.0,
                    float(transferable["min_similarity"]))
    related = np.flatnonzero(mask & (pair_sims >= threshold))
    related = related[np.argsort(-pair_sims[related], kind="stable")]
    above = [(names[iu[0][k]], names[iu[1][k]], round(float(pair_sims[k]), 3)) for k in related]
    return {
        "model_name": provider.model_name, "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": f"{len(students)} processed students x {len(jobs)} canonical jobs",
        "reference_pairs": {k: int(v.size) for k, v in values.items()},
        "quantiles": quantiles,
        "transferable": {"percentile": pct, "min_similarity": float(transferable["min_similarity"]),
                         "threshold": round(threshold, 6), "skills": len(names),
                         "pairs": int(mask.sum()), "pairs_related": len(above),
                         "examples_strongest": above[:10], "examples_at_threshold": above[-10:]},
    }


if __name__ == "__main__":
    from app.core.config import get_settings
    from app.db.matching_repository import build_matching_repository
    from app.embeddings.provider import build_embedding_provider
    from app.matching.config import get_matching_config

    settings = get_settings()
    repo = build_matching_repository(settings)
    if repo is None:
        raise SystemExit("DATABASE_URL is not set")
    provider = build_embedding_provider(settings)
    cfg = get_matching_config()
    t0 = time.time()
    students = [v for v, _ in repo.students()]
    jobs = [j for j in repo.jobs() if not j.near_duplicate_of]
    skill_names = [s for v in students for s in v.skills] + [s for j in jobs for s, _ in j.skills]
    result = build_calibration(students, jobs, provider, cfg.features["transferable"], skill_names)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    out = path_for(result["model_name"], result["model_version"])
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    t = result["transferable"]
    print(f"calibration for {result['model_name']} {result['model_version']} from {result['reference']} "
          f"in {time.time() - t0:.0f}s -> {out}")
    print(f"transferable threshold {t['threshold']} (p{t['percentile']} of {t['pairs']} skill pairs, floor "
          f"{t['min_similarity']}): {t['pairs_related']} related pairs")
    print("strongest:", t["examples_strongest"][:5])
    print("at threshold:", t["examples_at_threshold"][-5:])
