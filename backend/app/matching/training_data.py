"""The Phase 8 feature matrix: features-v1 for every labelled (student, job) pair (docs/adr/0015).

    DATABASE_URL=postgresql://<user>:<password>@<host>:5432/<db> python -m app.matching.training_data

Pairs: every weak-label pair (all splits; only train is used for training) plus every labelling-pool pair
(validation/test, judged by the team on /labelling), so the human-label run needs no database. Features are
computed by Phase 7's own stage functions (similarity -> features), exactly as the Matching Agent computes
them, over each student's judged jobs. Hard filters are a business rule, not a quality signal, so they are not
applied (as in Phase 7's evaluate_model).

Output: data/benchmark/matching_features/<feature_version>-<embedding_version>.csv.gz (ids, split, and the 20
contract features; no PII, no protected attribute) + .manifest.json (what produced it).
"""

import hashlib
import json
import time
from datetime import datetime, timezone

import pandas as pd

from app.core.config import REPO_ROOT, get_settings
from app.db.matching_repository import build_matching_repository
from app.embeddings.provider import build_embedding_provider
from app.matching import pipeline as P
from app.matching.config import get_matching_config
from app.matching.feature_builder import FEATURE_VERSION
from app.matching.features import FEATURE_NAMES, validate_feature_names
from app.tools.matching_tools import build_context
from src.evaluation.splits import split_for

BENCHMARK = REPO_ROOT / "data" / "benchmark"
OUT_DIR = BENCHMARK / "matching_features"
SOURCES = {"weak_labels": BENCHMARK / "weak_labels.csv", "labelling_pool": BENCHMARK / "labelling_pool.csv"}
ID_COLUMNS = ("student_id", "job_id", "split", "retrieval_scored")


def feature_settings_sha(cfg) -> str:
    """The parts of the matching config that change a feature value (not weights, not registration)."""
    body = {"features": cfg.features, "branch_rules": [(p.pattern, f) for p, f in cfg.branch_rules],
            "role_families": {k: v.pattern for k, v in cfg.role_families.items()}}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def labelled_pairs() -> pd.DataFrame:
    frames = [pd.read_csv(path, dtype=str, usecols=["student_id", "job_id"]) for path in SOURCES.values()]
    return pd.concat(frames).drop_duplicates().sort_values(["student_id", "job_id"]).reset_index(drop=True)


def student_features(ctx: P.MatchingContext, student_id: str, jobs: list[str]) -> list[dict]:
    stage2 = P.retrieval(ctx, "student", student_id, jobs, {}, len(jobs))
    scored = {r["id"]: r["score"] for r in stage2["top"]}
    sims = P.similarity(ctx, "student", student_id, jobs, scored)
    feats = P.features(ctx, "student", student_id, jobs, sims["percentiles"])["features"]
    return [{"student_id": student_id, "job_id": j, "split": split_for(student_id), "retrieval_scored": j in scored,
             **feats[j]} for j in jobs]


def build(ctx: P.MatchingContext) -> tuple[pd.DataFrame, dict]:
    pairs = labelled_pairs()
    have_students = {s.student_id for s, _ in ctx.repo.students()}
    have_jobs = {j.job_id for j in ctx.repo.jobs()}
    missing = pairs[~pairs["student_id"].isin(have_students) | ~pairs["job_id"].isin(have_jobs)]
    pairs = pairs.drop(missing.index)
    rows, t0 = [], time.time()
    groups = list(pairs.groupby("student_id"))
    for i, (sid, group) in enumerate(groups, start=1):
        rows += student_features(ctx, sid, sorted(group["job_id"]))
        if i % 25 == 0 or i == len(groups):
            print(f"  {i}/{len(groups)} students, {len(rows):,} pairs, {time.time() - t0:.0f} s", flush=True)
    frame = pd.DataFrame(rows, columns=[*ID_COLUMNS, *FEATURE_NAMES])
    validate_feature_names(frame.columns[len(ID_COLUMNS):])
    version, _ = ctx.provider.describe()
    manifest = {
        "feature_version": FEATURE_VERSION, "features": list(FEATURE_NAMES),
        "embedding_model": ctx.provider.model_name, "embedding_version": version,
        "calibration": ctx.calibration().model_version,
        "dataset_version": P.dataset_version(ctx.processed_dir),
        "feature_settings_sha256": feature_settings_sha(ctx.cfg),
        "sources": {k: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for k, p in SOURCES.items()},
        "pairs": int(len(frame)), "students": int(frame["student_id"].nunique()), "jobs": int(frame["job_id"].nunique()),
        "pairs_without_stored_profile_or_posting": int(len(missing)),
        "pairs_not_scored_by_retrieval": int((~frame["retrieval_scored"]).sum()),
        "missing_share_per_feature": {f: round(float(frame[f].isna().mean()), 4) for f in FEATURE_NAMES},
        "seconds": round(time.time() - t0, 1),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return frame, manifest


def output_path(feature_version: str, embedding_version: str):
    return OUT_DIR / f"{feature_version}-{embedding_version.replace('/', '_')}.csv.gz"  # HF versions hold a "/"


def main() -> None:
    settings = get_settings()
    repo = build_matching_repository(settings)
    if repo is None:
        raise SystemExit("DATABASE_URL is not set")
    provider = build_embedding_provider(settings)
    ctx = build_context(repo, provider, settings, cfg=get_matching_config())
    frame, manifest = build(ctx)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path(FEATURE_VERSION, manifest["embedding_version"])
    frame.to_csv(out, index=False, float_format="%.6g", compression={"method": "gzip", "mtime": 0})
    out.with_suffix("").with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"written {out} ({manifest['pairs']:,} pairs, {manifest['students']} students, {manifest['jobs']} jobs; "
          f"{manifest['pairs_not_scored_by_retrieval']} not scored by retrieval; {manifest['seconds']} s)")


if __name__ == "__main__":
    main()
