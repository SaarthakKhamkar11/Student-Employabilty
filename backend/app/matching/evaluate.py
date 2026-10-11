"""Evaluate, benchmark and register the matching configuration (spec §24, §25; docs/adr/0014).

    DATABASE_URL=postgresql://<user>:<password>@127.0.0.1:5432/<db> \\
        python -m app.matching.evaluate --labels data/benchmark/weak_labels.csv --register

1. Evaluate weighted_baseline_v1 on the labelled pairs of the validation split (never train): MRR, NDCG@K,
   P@K, R@K and Hit Rate with 95% bootstrap intervals, next to "semantic similarity alone" (the retrieval
   score) as a reference. The Phase 6 circular-evaluation check runs on the scores. Weak labels come from the
   same role families branch_match uses, so their numbers are a sanity check, never a selection.
2. Benchmark the full Stage 1-6 pipeline, per stage, on a sample of validation students.
3. Record the run as an experiment (experiment_runs / experiment_metrics) and in results/matching_evaluation/.
4. --register: write that experiment_id and the config hash into config/matching_baseline_v1.yaml, so every
   prediction can name it (spec §24). Editing the config changes the hash, and predictions are refused until
   the edited config is registered again.
"""

import argparse
import json
import platform
import re
import statistics
import time
from datetime import datetime, timezone

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from app.core.config import REPO_ROOT, get_settings
from app.db.matching_repository import build_matching_repository
from app.embeddings.provider import build_embedding_provider
from app.matching import pipeline as P
from app.matching.config import MATCHING_CONFIG_PATH, get_matching_config
from app.matching.feature_builder import FEATURE_VERSION
from app.matching.predictor import BaselinePredictor
from app.tools.matching_tools import build_context, score_labelled
from src.evaluation.relevance_labels import STAMPS, circularity_check, load_labels
from src.evaluation.retrieval_metrics import bootstrap_ci, evaluate, paired_difference

RESULTS = REPO_ROOT / "results" / "matching_evaluation"
STUDENTS_CSV = REPO_ROOT / "data" / "processed" / "students_unified.csv"
JOBS_CSV = REPO_ROOT / "data" / "processed" / "jobs.csv"
SPLIT, KS, THRESHOLD, SEED = "validation", (5, 10), 2, 42


def evaluate_on_labels(ctx: P.MatchingContext, labels_path: str) -> dict:
    labelset = load_labels(labels_path, str(STUDENTS_CSV), str(JOBS_CSV))
    source = labelset.preferred_source()
    frame = labelset.select(source, [SPLIT])
    have = {s.student_id for s, _ in ctx.repo.students()}
    frame = frame[frame["student_id"].isin(have)]
    pairs = frame[["student_id", "job_id", "relevance_label"]].to_dict("records")
    reference: dict = {}
    t0 = time.time()
    baseline = score_labelled(ctx, pairs, KS, THRESHOLD, reference=reference)
    seconds = time.time() - t0
    scores = baseline.pop("scores")
    semantic = evaluate(reference, frame, KS, THRESHOLD)
    pairs_set = set(zip(frame["student_id"], frame["job_id"]))
    validity = circularity_check(labelset, source, ["weighted_baseline_v1", "semantic_only"],
                                 {"weighted_baseline_v1": {k: v for k, v in scores.items() if k in pairs_set},
                                  "semantic_only": {k: v for k, v in reference.items() if k in pairs_set}})
    per = {"weighted_baseline_v1": baseline["per_student"], "semantic_only": semantic["per_student"]}
    ci = {m: {k: bootstrap_ci([v[k] for v in per[m].values()], 2000, SEED) for k in ("mrr", "ndcg@5", "ndcg@10")}
          for m in per}
    diff = paired_difference({s: v["ndcg@5"] for s, v in per["weighted_baseline_v1"].items()},
                             {s: v["ndcg@5"] for s, v in per["semantic_only"].items()}, 2000, SEED)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("per_student", "kendall_tau_per_student", "scores")}  # noqa: E731
    return {"labels": {"path": labels_path, "sha256": labelset.sha256, "source": source, "split": SPLIT,
                       "pairs": int(len(frame)), "students": int(frame["student_id"].nunique()),
                       "stamp": STAMPS.get(source), "validity": validity},
            "weighted_baseline_v1": strip(baseline), "semantic_only": strip(semantic), "ci95": ci,
            "baseline_minus_semantic_ndcg@5": diff, "seconds": round(seconds, 1)}


def benchmark(ctx: P.MatchingContext, n: int = 10) -> dict:
    from src.evaluation.splits import split_for

    students = [s.student_id for s, _ in ctx.repo.students() if split_for(s.student_id) == SPLIT][:n]
    runs = [P.run_pipeline(ctx, "jobs_for_student", s, None, 10, with_versions=False)["timings_ms"]
            for s in students]
    summary = lambda v: {"median": round(statistics.median(v), 1), "p95": round(float(np.percentile(v, 95)), 1)}  # noqa: E731
    return {"students": students, "stage_ms": {k: summary([r[k] for r in runs]) for k in runs[0]},
            "total_ms": summary([sum(r.values()) for r in runs])}


def record(dsn: str, run: dict) -> None:
    ev = run.get("evaluation")
    rows = []
    if ev:
        for model in ("weighted_baseline_v1", "semantic_only"):
            for metric, value in ev[model]["mean"].items():
                if value is not None:
                    rows.append((model, run["matching_model_version"] if model != "semantic_only" else
                                 run["embedding_version"], metric, SPLIT, float(value), "measured"))
    for stage, v in run["benchmark"]["stage_ms"].items():
        rows.append(("weighted_baseline_v1", run["matching_model_version"], f"latency_ms_{stage}", "n/a",
                     float(v["median"]), "measured"))
    with psycopg.connect(dsn, connect_timeout=5) as conn, conn.transaction():
        conn.execute(
            "INSERT INTO experiment_runs (experiment_id, kind, started_at, finished_at, dataset_version, dataset_size, "
            "train_split, validation_split, test_split, random_seed, feature_version, embedding_model, embedding_version, "
            "prediction_model, hyperparameters, metrics, runtime_seconds, memory, hardware, labels, notes, limitations, "
            "report) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (run["experiment_id"], "evaluation", run["started_at"], run["finished_at"], run["dataset_version"],
             Jsonb(run["dataset_size"]), Jsonb({"evaluated": False}), Jsonb({"evaluated": bool(ev)}),
             Jsonb({"evaluated": False}), SEED, FEATURE_VERSION, run["embedding_model"], run["embedding_version"],
             run["matching_model_version"], Jsonb(run["config"]),
             Jsonb({k: run[k] for k in ("evaluation", "benchmark") if run.get(k)}), run["runtime_seconds"],
             Jsonb({}), Jsonb(run["hardware"]), Jsonb((ev or {}).get("labels") or {"provided": False}),
             run["notes"], run["limitations"], Jsonb(run)))
        conn.cursor().executemany(
            "INSERT INTO experiment_metrics (experiment_id, model_name, model_version, metric, split, value, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)", [(run["experiment_id"], *r) for r in rows])


def register(experiment_id: str, sha256: str) -> None:
    text = MATCHING_CONFIG_PATH.read_text(encoding="utf-8")
    text, n1 = re.subn(r"(?m)^(  experiment_id: ).*$", rf"\g<1>{experiment_id}", text)
    text, n2 = re.subn(r"(?m)^(  config_sha256: ).*$", rf"\g<1>{sha256}", text)
    if (n1, n2) != (1, 1):
        raise RuntimeError("could not find the registration block in the matching config")
    MATCHING_CONFIG_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=None, help="labels CSV (validated; validation split only)")
    ap.add_argument("--register", action="store_true", help="name this experiment in the config's registration")
    ap.add_argument("--benchmark-students", type=int, default=10)
    args = ap.parse_args()
    settings = get_settings()
    repo = build_matching_repository(settings)
    if repo is None or not settings.database_url:
        raise SystemExit("DATABASE_URL is not set")
    provider = build_embedding_provider(settings)
    cfg = get_matching_config()
    ctx = build_context(repo, provider, settings, cfg=cfg, predictor=BaselinePredictor(cfg))  # always Baseline 3
    version, _ = provider.describe()
    started = datetime.now(timezone.utc)
    t0 = time.time()
    evaluation = evaluate_on_labels(ctx, args.labels) if args.labels else None
    bench = benchmark(ctx, args.benchmark_students)
    tag = (evaluation["labels"]["source"].split("_")[0]) if evaluation else "nolabels"
    run = {
        "experiment_id": f"match-{started:%Y%m%dT%H%M%SZ}-{tag}", "kind": "evaluation",
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 1), "dataset_version": P.dataset_version(str(settings.processed_data_dir)),
        "dataset_size": {"students_with_profiles": len(repo.students()), "jobs_with_postings": len(repo.jobs())},
        "embedding_model": provider.model_name, "embedding_version": version,
        "matching_model": cfg.matching_model, "matching_model_version": cfg.matching_model_version,
        "feature_version": FEATURE_VERSION,
        "config": {"path": str(MATCHING_CONFIG_PATH.relative_to(REPO_ROOT)), "sha256": cfg.sha256,
                   "weights": cfg.weights, "top_n": cfg.top_n, "eligibility": cfg.eligibility, "features": cfg.features},
        "calibration": ctx.calibration().model_version, "evaluation": evaluation, "benchmark": bench,
        "hardware": {"machine": platform.machine(), "system": f"{platform.system()} {platform.release()}",
                     "python": platform.python_version()},
        "notes": ("Registers weighted_baseline_v1 (interim, placeholder weights pending Phase 8). Quality is on the "
                  "labels given, validation split only; the benchmark times the full Stage 1-6 pipeline."),
        "limitations": ("Weak labels share role families with branch_match and reflect the student's stated domain, as "
                        "career_interest_alignment does: a sanity check, not a selection. Synthetic templated job data. "
                        "Laptop timings with Docker running."),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{run['experiment_id']}.json"
    out.write_text(json.dumps(run, indent=1, default=str), encoding="utf-8")
    record(settings.database_url.get_secret_value(), run)
    if args.register:
        register(run["experiment_id"], cfg.sha256)
    print(json.dumps({k: run[k] for k in ("experiment_id", "matching_model_version", "runtime_seconds")}, indent=1))
    if evaluation:
        e = evaluation
        print(f"labels: {e['labels']['source']} {e['labels']['split']} {e['labels']['pairs']} pairs, "
              f"{e['labels']['students']} students | {e['labels']['validity']['line']}")
        for model in ("weighted_baseline_v1", "semantic_only"):
            m = e[model]["mean"]
            print(f"  {model:22s} NDCG@5 {m['ndcg@5']:.3f} (CI {e['ci95'][model]['ndcg@5'][0]:.3f}-"
                  f"{e['ci95'][model]['ndcg@5'][1]:.3f})  MRR {m['mrr']:.3f}  P@5 {m['precision@5']:.3f}  "
                  f"R@10 {m['recall@10']:.3f}  tau {m['kendall_tau']:.3f}")
        d = e["baseline_minus_semantic_ndcg@5"]
        print(f"  baseline - semantic NDCG@5: {d['mean_difference']:+.3f} (95% CI {d['ci95'][0]:+.3f} to "
              f"{d['ci95'][1]:+.3f}), {'distinguishable' if d['distinguishable'] else 'not distinguishable'}")
        print(f"  Kendall tau vs labels: {e['labels']['validity']['mean_per_student_kendall_tau']}")
    print(f"benchmark ({len(bench['students'])} students, full pipeline): {bench['stage_ms']} total {bench['total_ms']}")
    print(f"written {out}" + ("; registered in config" if args.register else ""))


if __name__ == "__main__":
    main()
