"""Matching Model Selection Engine (Phase 8, spec §10, §13): one table, transparent weights, sensitivity analysis.

    python -m src.scoring.matching_model_selection_engine
        trains every candidate (each in its own process) and measures cost/engineering/explainability; quality
        reads "Not yet evaluated" without labels. Artifacts: models/matching/<experiment_id>/
    python -m src.scoring.matching_model_selection_engine --labels data/benchmark/human_labels.csv --reuse-models <id>
        evaluates the SAME trained artifacts on the human labels (validation split) and applies the selection rule
        fixed in docs/adr/0015. --activate then writes the winner into config/matching_model.yaml.
    --labels data/benchmark/weak_labels.csv --reuse-models <id>   a stamped weak-label diagnostic, never a selection
    --record-db   store the experiment in Postgres (EXPERIMENTS_DATABASE_URL or DATABASE_URL)
    --record-existing results/matching_model_evaluation/<id>.json   store a finished run without re-running it

Rules: every model (6 trained candidates + Baseline 3) is scored on the identical pairs of the identical split;
train is never evaluated; test is held back for Phase 14. No model is preferred in code: the winner is whatever
the weights in config/matching_model_selection_weights.yaml produce from the measured numbers.
"""

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from app.matching.features import FEATURE_NAMES, PROTECTED_ATTRIBUTES, validate_feature_names
from src.evaluation import NOT_YET_EVALUATED
from src.evaluation.relevance_labels import STAMPS, circularity_check, load_labels
from src.evaluation.retrieval_metrics import bootstrap_ci, evaluate, paired_difference
from src.evaluation.splits import SPLIT_VERSION
from src.scoring.matching_models import (BASELINE_KEY, CANDIDATES_PATH, complexity_points, feature_matrix,
                                         load_candidates, read_features, sha256)

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "config" / "matching_model_selection_weights.yaml"
ACTIVE_CONFIG = ROOT / "config" / "matching_model.yaml"
FEATURES_DIR = ROOT / "data" / "benchmark" / "matching_features"
WEAK_LABELS = ROOT / "data" / "benchmark" / "weak_labels.csv"
STUDENTS_CSV = ROOT / "data" / "processed" / "students_unified.csv"
JOBS_CSV = ROOT / "data" / "processed" / "jobs.csv"
MODELS_DIR = ROOT / "models" / "matching"
RESULTS = ROOT / "results"

QUALITY = ("f1", "ndcg", "mrr")
COST = ("latency", "training_time", "memory", "complexity", "explainability")
HIGHER_IS_BETTER = {"f1", "ndcg", "mrr", "explainability"}
FLOORS = {"latency": 1e-3, "training_time": 1e-3, "memory": 0.1}  # keeps log10 finite for Baseline 3's zeros


# ---------------------------------------------------------------- config

def load_weights(path: Path = WEIGHTS) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    criteria = set(QUALITY) | set(COST)
    for name, scenario in cfg["scenarios"].items():
        weights = scenario["weights"]
        if set(weights) != criteria:
            raise ValueError(f"scenario {name!r} must weight exactly {sorted(criteria)}")
        if any(not str(w.get("reason", "")).strip() for w in weights.values()):
            raise ValueError(f"scenario {name!r}: every weight needs a one-line reason")
        total = sum(w["weight"] for w in weights.values())
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"scenario {name!r}: weights sum to {total}, not 1")
    if cfg["evaluation"]["selection_split"] == "train":
        raise ValueError("selection_split can't be train: models are trained on it")
    cfg["sha256"] = sha256(path)
    return cfg


def latest_features() -> Path:
    files = sorted(FEATURES_DIR.glob("*.csv.gz"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit("no feature matrix: run `python -m app.matching.training_data` first")
    return files[-1]


# ---------------------------------------------------------------- multi-criteria decision

def normalise(criteria: dict[str, dict], method: str = "min_max", log_scale=()) -> dict:
    """{criterion: {model: score}}, 1 = best. A criterion with any missing value is left out."""
    out = {}
    for c in sorted({c for v in criteria.values() for c in v}):
        values = {m: criteria[m].get(c) for m in criteria}
        if any(v is None for v in values.values()):
            continue
        x = {m: (math.log10(max(v, FLOORS.get(c, 1e-9))) if c in log_scale else v) for m, v in values.items()}
        sign = 1.0 if c in HIGHER_IS_BETTER else -1.0
        if method == "z_score":
            arr = np.array(list(x.values()))
            sd = arr.std()
            out[c] = {m: (sign * (v - arr.mean()) / sd if sd > 0 else 0.0) for m, v in x.items()}
        else:
            lo, hi = min(x.values()), max(x.values())
            out[c] = {m: (1.0 if hi == lo else ((v - lo) if sign > 0 else (hi - v)) / (hi - lo)) for m, v in x.items()}
    return out


def overall_scores(normalised: dict, weights: dict[str, float]) -> dict[str, float] | None:
    if any(w > 0 and c not in normalised for c, w in weights.items()):
        return None
    models = next(iter(normalised.values())).keys()
    return {m: float(sum(w * normalised[c][m] for c, w in weights.items() if w > 0)) for m in models}


def pick_winner(scores: dict[str, float] | None, eligible: list[str]) -> str | None:
    if not scores:
        return None
    pool = [m for m in eligible if m in scores]
    return max(pool, key=lambda m: (scores[m], m)) if pool else None


def random_weight_stability(normalised, balanced, eligible, n, concentration, seed) -> dict | None:
    names = list(balanced)
    if overall_scores(normalised, balanced) is None or not eligible:
        return None
    rng = np.random.default_rng(seed)
    wins = {m: 0 for m in eligible}
    for draw in rng.dirichlet(np.array([balanced[c] for c in names]) * concentration, size=n):
        wins[pick_winner(overall_scores(normalised, dict(zip(names, draw))), eligible)] += 1
    return {m: round(100 * k / n, 1) for m, k in sorted(wins.items(), key=lambda kv: -kv[1])}


def explainability_points(meas: dict, interpretability: int) -> tuple[int, list[str]]:
    """Rubric in config/matching_model_candidates.yaml, from measurements."""
    e, points, reasons = meas["explainability"], 0, []
    if e["additivity_max_abs_error"] is not None and e["additivity_max_abs_error"] < 1e-4:
        points += 2
        reasons.append(f"+2 match-level trace verified on {e['pairs_checked']} pairs "
                       f"(max error {e['additivity_max_abs_error']:.1e})")
    else:
        reasons.append("+0 contributions did not add up to the output")
    if interpretability >= 4:
        points += 2
        reasons.append(f"+2 readable global structure (interpretability {interpretability}/5)")
    else:
        reasons.append(f"+0 post-hoc only (interpretability {interpretability}/5)")
    if e["single_pair_explanation_ms"]["median"] < 10:
        points += 1
        reasons.append(f"+1 explanation {e['single_pair_explanation_ms']['median']:.2f} ms per pair")
    else:
        reasons.append(f"+0 explanation {e['single_pair_explanation_ms']['median']:.1f} ms per pair (>= 10 ms)")
    return points, reasons


# ---------------------------------------------------------------- training (one isolated process per model)

def train_all(features_path: Path, folder: Path, keys: list[str]) -> dict[str, dict]:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "backend"), str(ROOT),
                                                         os.environ.get("PYTHONPATH", "")]))
    out = {}
    for key in keys:
        print(f"  training + measuring {key} (own process)", flush=True)
        t = time.time()
        subprocess.run([sys.executable, "-m", "src.scoring.matching_models", "measure", "--key", key,
                        "--features", str(features_path), "--labels", str(WEAK_LABELS), "--out", str(folder)],
                       check=True, env=env, cwd=ROOT)
        path = folder / f"{key}.measurement.json"
        out[key] = json.loads(path.read_text(encoding="utf-8"))
        path.unlink()
        print(f"    done in {time.time() - t:.0f} s", flush=True)
    return out


def write_manifest(folder: Path, experiment_id: str, measurements: dict, feature_manifest: dict, cands: dict,
                   features_path: Path) -> dict:
    manifest = {
        "experiment_id": experiment_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_version": feature_manifest["feature_version"], "features": list(FEATURE_NAMES),
        "embedding_model": feature_manifest["embedding_model"], "embedding_version": feature_manifest["embedding_version"],
        "dataset_version": feature_manifest["dataset_version"],
        "feature_matrix": {"file": features_path.name, "sha256": sha256(features_path)},
        "training": cands["training"], "candidates_sha256": cands["sha256"],
        "weak_labels_sha256": sha256(WEAK_LABELS),
        "models": {k: m["artifact"] for k, m in measurements.items() if k != BASELINE_KEY},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    slim = {k: {kk: vv for kk, vv in m.items() if kk != "scores"} for k, m in measurements.items()}
    (folder / "measurements.json").write_text(json.dumps(slim, indent=1), encoding="utf-8")
    return manifest


def rescore(folder: Path, manifest: dict, features: pd.DataFrame, measurements: dict) -> None:
    """--reuse-models: score the held-out pairs with the saved artifacts (hash-checked), not a retrained model."""
    from src.scoring.matching_models import load, measure_baseline

    held = features[features["split"] != "train"].reset_index(drop=True)
    X, ids = feature_matrix(held), held["student_id"] + "|" + held["job_id"]
    for key, art in manifest["models"].items():
        model = load(folder / art["file"], key, art["format"], art["sha256"])
        measurements[key]["scores"] = dict(zip(ids, model.score(X).round(8).tolist()))
    measurements[BASELINE_KEY]["scores"] = measure_baseline(features, folder)["scores"]


# ---------------------------------------------------------------- quality

def classification(scores: dict[str, float], labels: pd.DataFrame, threshold: int, decision: float) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score, recall_score,
                                 roc_auc_score)

    keys = labels["student_id"] + "|" + labels["job_id"]
    s = keys.map(scores)
    scored = s.notna().to_numpy()
    y = (labels["relevance_label"].astype(int) >= threshold).to_numpy()[scored]
    p = s.to_numpy(dtype=float)[scored]
    pred = p >= decision
    both = len(set(y)) == 2
    return {"pairs": int(scored.sum()), "unscored_pairs": int((~scored).sum()), "positives": int(y.sum()),
            "accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "pr_auc": float(average_precision_score(y, p)) if both else None}


def quality_for(measurements: dict, eval_labels: pd.DataFrame, ev: dict, decision: float) -> dict:
    out = {}
    for key, m in measurements.items():
        pair_scores = {tuple(k.split("|")): v for k, v in m["scores"].items()}
        ranking = evaluate(pair_scores, eval_labels, tuple(ev["ks"]), ev["relevance_threshold"])
        cls = classification(m["scores"], eval_labels, ev["relevance_threshold"],
                             0.5 if key == BASELINE_KEY else decision)
        out[key] = {"ranking": ranking, "classification": cls, "pair_scores": pair_scores}
    return out


# ---------------------------------------------------------------- the run

def run(features_path: Path, labels_path: str | None, reuse: str | None, label_source: str | None = None) -> dict:
    started = time.time()
    started_at = datetime.now(timezone.utc)
    cfg, cands = load_weights(), load_candidates()
    ev = cfg["evaluation"]
    k, threshold, split = ev["primary_k"], ev["relevance_threshold"], ev["selection_split"]
    features = read_features(features_path)
    feature_manifest = json.loads(features_path.with_suffix("").with_suffix(".manifest.json").read_text(encoding="utf-8"))

    # the feature-exclusion guard, re-run against THIS phase's training data
    columns = [c for c in features.columns if c not in ("student_id", "job_id", "split", "retrieval_scored")]
    validate_feature_names(columns)
    exclusion = {"columns_given_to_models": list(FEATURE_NAMES), "extra_columns": sorted(set(columns) - set(FEATURE_NAMES)),
                 "protected_present": sorted(c for c in features.columns
                                             if c.lower() in PROTECTED_ATTRIBUTES or "gender" in c.lower()),
                 "passed": True}

    tag = (label_source or "") if labels_path else "nolabels"
    labelset, source, eval_labels = None, None, None
    labels_info = {"provided": False, "path": None}
    if labels_path:
        labelset = load_labels(labels_path, str(STUDENTS_CSV), str(JOBS_CSV))
        source = label_source or labelset.preferred_source()
        tag = source.split("_")[0]
        eval_labels = labelset.select(source, [split])
        if eval_labels.empty:
            raise ValueError(f"no {source} labels in the {split!r} split: nothing to evaluate")
        have = set(features["student_id"] + "|" + features["job_id"])
        missing = eval_labels[~(eval_labels["student_id"] + "|" + eval_labels["job_id"]).isin(have)]
        if len(missing):
            raise ValueError(f"{len(missing)} labelled pairs have no features: rebuild the feature matrix")
        labels_info = {"provided": True, "path": os.path.relpath(labels_path, ROOT), "sha256": labelset.sha256,
                       "pairs": int(len(labelset.frame)), "source_breakdown": labelset.source_breakdown(),
                       "split_breakdown": labelset.split_breakdown(), "label_distribution": labelset.label_distribution(),
                       "used_source": source, "used_split": split, "used_pairs": int(len(eval_labels)),
                       "used_students": int(eval_labels["student_id"].nunique()),
                       "used_label_distribution": {int(g): int((eval_labels["relevance_label"] == g).sum())
                                                   for g in (0, 1, 2, 3)},
                       "train_rows_evaluated": 0, "stamp": STAMPS.get(source)}
    experiment_id = f"mms-{started_at:%Y%m%dT%H%M%SZ}-{tag}"

    # train (or reuse) -----------------------------------------------------------------------------------
    keys = list(cands["candidates"])
    if reuse:
        folder = MODELS_DIR / reuse
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["feature_matrix"]["sha256"] != sha256(features_path):
            raise ValueError("the feature matrix changed since these models were trained: retrain")
        measurements = json.loads((folder / "measurements.json").read_text(encoding="utf-8"))
        rescore(folder, manifest, features, measurements)
        artifacts_experiment = reuse
    else:
        folder = MODELS_DIR / experiment_id
        folder.mkdir(parents=True, exist_ok=True)
        measurements = train_all(features_path, folder, [*keys, BASELINE_KEY])
        manifest = write_manifest(folder, experiment_id, measurements, feature_manifest, cands, features_path)
        artifacts_experiment = experiment_id
    models = [*keys, BASELINE_KEY]

    # quality ----------------------------------------------------------------------------------------------
    quality, validity, stats = {}, None, {}
    if eval_labels is not None:
        quality = quality_for(measurements, eval_labels, ev, cands["training"]["decision_threshold"])
        pairs = set(zip(eval_labels["student_id"], eval_labels["job_id"]))
        validity = circularity_check(labelset, source, models,
                                     {m: {p: v for p, v in q["pair_scores"].items() if p in pairs}
                                      for m, q in quality.items()})
        labels_info["validity"] = validity
        boot, seed = ev["bootstrap_resamples"], ev["seed"]
        per = {m: quality[m]["ranking"]["per_student"] for m in models}
        primary = f"ndcg@{k}"
        for m in models:
            stats[m] = {metric: bootstrap_ci([v[metric] for v in per[m].values()], boot, seed)
                        for metric in ("mrr", "ndcg@5", f"ndcg@{k}")}
        stats["vs_baseline"] = {m: paired_difference({s: v[primary] for s, v in per[m].items()},
                                                     {s: v[primary] for s, v in per[BASELINE_KEY].items()}, boot, seed)
                                for m in keys}

    def qv(m, name):
        if m not in quality:
            return None
        if name == "f1":
            return quality[m]["classification"]["f1"]
        return quality[m]["ranking"]["mean"].get("mrr" if name == "mrr" else f"{name}@{k}")

    # criteria, scores, scenarios -------------------------------------------------------------------------------
    engineering, criteria = {}, {}
    for m in models:
        meas = measurements[m]
        spec = cands["candidates"].get(m) or cands["baseline"]
        cpts, creasons = complexity_points(spec if m != BASELINE_KEY else {"library": "scikit-learn"})
        epts, ereasons = explainability_points(meas, int(spec["interpretability"]["score"]))
        engineering[m] = {"complexity": {"points": cpts, "reasons": creasons},
                          "explainability": {"points": epts, "reasons": ereasons},
                          "interpretability": spec["interpretability"], "maintainability": spec["maintainability"]}
        criteria[m] = {"f1": qv(m, "f1"), "ndcg": qv(m, "ndcg"), "mrr": qv(m, "mrr"),
                       "latency": meas["inference"]["batch_50_ms"]["median"],
                       "training_time": meas["training"]["median_seconds"],
                       "memory": meas["training"]["memory_added_mb"], "complexity": cpts, "explainability": epts}
    normalised = normalise(criteria, cfg["normalization"]["method"], cfg["normalization"]["log_scale"])
    gate = cfg["gate"]["primary_quality_metric"]
    if quality and cfg["gate"]["must_beat_baseline"]:
        eligible = [m for m in keys if criteria[m][gate] > criteria[BASELINE_KEY][gate]]
    else:
        eligible = list(keys)
    scenarios = {}
    for name, sc in cfg["scenarios"].items():
        weights = {c: v["weight"] for c, v in sc["weights"].items()}
        overall = overall_scores(normalised, weights)
        scenarios[name] = {"description": sc["description"], "weights": weights,
                           "reasons": {c: v["reason"] for c, v in sc["weights"].items()}, "overall": overall,
                           "ranking": sorted(overall, key=lambda m: -overall[m]) if overall else None,
                           "winner": pick_winner(overall, eligible)}
    winners = {n: s["winner"] for n, s in scenarios.items()}
    known = [w for w in winners.values() if w]
    sens = cfg["sensitivity"]
    stability = random_weight_stability(normalised, scenarios["balanced"]["weights"], eligible,
                                        sens["random_weightings"], sens["concentration"], sens["seed"])
    # The simpler-model check for every scenario's winner, and always for XGBoost (never chosen by default).
    simple = None
    if quality:
        compared = dict.fromkeys([w for w in winners.values() if w] + (["xgboost"] if "xgboost" in keys else []))
        simple = {m: simple_model_check(cfg, criteria, quality, m, k, ev) for m in compared}

    hw = {"machine": platform.machine(), "system": f"{platform.system()} {platform.release()}",
          "python": platform.python_version(), "cpu_count": os.cpu_count(),
          "libraries": library_versions()}
    report = {
        "experiment_id": experiment_id, "kind": "matching_selection",
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - started, 1),
        "artifacts_experiment": artifacts_experiment, "artifacts_dir": os.path.relpath(MODELS_DIR / artifacts_experiment, ROOT),
        "dataset_version": feature_manifest["dataset_version"], "feature_version": feature_manifest["feature_version"],
        "embedding_model": feature_manifest["embedding_model"], "embedding_version": feature_manifest["embedding_version"],
        "feature_matrix": {"file": os.path.relpath(features_path, ROOT), "sha256": sha256(features_path),
                           "pairs": feature_manifest["pairs"],
                           "missing_share_per_feature": feature_manifest["missing_share_per_feature"]},
        "exclusion_check": exclusion,
        "splits": {"version": SPLIT_VERSION, "training": cands["training"], "selection_split": split,
                   "train_evaluated": False, "test": "held back for Phase 14"},
        "target": {"formulation": "binary classification: suitable = relevance_label >= 2 (docs/adr/0015)",
                   "train_label_distribution": train_distribution(features)},
        "config": {"weights_file": os.path.relpath(WEIGHTS, ROOT), "weights_version": cfg["version"],
                   "weights_sha256": cfg["sha256"], "candidates_file": os.path.relpath(CANDIDATES_PATH, ROOT),
                   "candidates_version": cands["version"], "candidates_sha256": cands["sha256"],
                   "evaluation": ev, "normalization": cfg["normalization"], "gate": cfg["gate"],
                   "sensitivity": sens, "close_enough_margin": cfg["close_enough_margin"],
                   "simple_models": cfg["simple_models"]},
        "labels": labels_info, "validity": (validity or {}).get("line", NOT_YET_EVALUATED),
        "models": {m: {"display": measurements[m]["display"], "library": measurements[m]["library"],
                       "params": measurements[m]["params"] if m != BASELINE_KEY else "weights in config/matching_baseline_v1.yaml",
                       "role": "baseline (floor)" if m == BASELINE_KEY else "candidate"} for m in models},
        "measurements": {m: {kk: vv for kk, vv in measurements[m].items() if kk != "scores"} for m in models},
        "quality": {m: {"classification": q["classification"],
                        "ranking": {kk: vv for kk, vv in q["ranking"].items()
                                    if kk not in ("per_student", "kendall_tau_per_student")}}
                    for m, q in quality.items()},
        "confidence": stats, "engineering": engineering, "criteria": criteria, "normalised": normalised,
        "eligible": eligible, "scenarios": scenarios, "winner_changes": (len(set(known)) > 1) if known else None,
        "random_weight_stability": stability, "simple_model_check": simple, "hardware": hw,
    }
    report["recommendation"] = recommend(report)
    report["markdown"] = render_markdown(report)
    return report


def train_distribution(features: pd.DataFrame) -> dict:
    weak = pd.read_csv(WEAK_LABELS, dtype=str)
    weak = weak[weak["split"] == "train"]
    counts = weak["relevance_label"].astype(int).value_counts()
    return {int(g): int(counts.get(g, 0)) for g in (0, 1, 2, 3)}


def library_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in ("scikit-learn", "xgboost-cpu", "xgboost", "lightgbm", "shap", "numpy", "pandas"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def simple_model_check(cfg, criteria, quality, winner, k, ev) -> dict | None:
    """How close is the best simple model (Logistic Regression / Decision Tree) to the winner?"""
    if winner is None:
        return None
    simple = max(cfg["simple_models"], key=lambda m: criteria[m]["ndcg"])
    if simple == winner:
        return {"simple": simple, "winner": winner, "same": True}
    per = lambda m: {s: v[f"ndcg@{k}"] for s, v in quality[m]["ranking"]["per_student"].items()}  # noqa: E731
    diff = paired_difference(per(winner), per(simple), ev["bootstrap_resamples"], ev["seed"]) or {}
    gap = criteria[winner]["ndcg"] - criteria[simple]["ndcg"]
    close = gap <= cfg["close_enough_margin"] or not diff.get("distinguishable", False)
    return {"simple": simple, "winner": winner, "same": False, "ndcg_gap": gap, "paired": diff, "close_enough": close,
            "f1_gap": criteria[winner]["f1"] - criteria[simple]["f1"],
            "latency_ratio": criteria[winner]["latency"] / max(criteria[simple]["latency"], 1e-9),
            "complexity_gap": criteria[winner]["complexity"] - criteria[simple]["complexity"],
            "explainability_gap": criteria[winner]["explainability"] - criteria[simple]["explainability"]}


def recommend(r: dict) -> dict:
    """The selection rule fixed in docs/adr/0015 before any human label existed.

    Only human_reviewed labels with evaluation validity OK can select. The Balanced winner among models that beat
    Baseline 3 on NDCG@K is selected, unless Baseline 3 scores higher overall AND the winner's NDCG gain over it is
    not statistically distinguishable: then the evidence says keep Baseline 3, which is a valid outcome."""
    labels, win = r["labels"], r["scenarios"]["balanced"]["winner"]
    keep = "Keep weighted_baseline_v1 (Baseline 3) as the active matching model"
    if not labels["provided"]:
        return {"decision": "no_selection", "model": None,
                "line": f"Insufficient evidence: no relevance labels. {keep} until the human-label run."}
    level, source = labels["validity"]["level"], labels["used_source"]
    if source != "human_reviewed":
        return {"decision": "no_selection", "model": None,
                "line": f"Not a selection: {source} labels can't select a model (evaluation validity {level}). "
                        f"Provisional leader on these labels: {_name(r, win) if win else 'none'}. {keep} until "
                        "the human-label run."}
    if level != "OK":
        return {"decision": "no_selection", "model": None,
                "line": f"Not a selection: evaluation validity {level} ({'; '.join(labels['validity']['reasons'])}). {keep}."}
    if win is None:
        return {"decision": "keep_baseline", "model": BASELINE_KEY,
                "line": f"{keep}: no trained model beat it on NDCG@{r['config']['evaluation']['primary_k']}."}
    overall = r["scenarios"]["balanced"]["overall"]
    gain = (r["confidence"].get("vs_baseline") or {}).get(win) or {}
    if overall[BASELINE_KEY] > overall[win] and not gain.get("distinguishable"):
        return {"decision": "keep_baseline", "model": BASELINE_KEY,
                "line": f"{keep}: {_name(r, win)} leads the candidates, but its NDCG gain over Baseline 3 is not "
                        f"statistically distinguishable and Baseline 3 scores higher overall "
                        f"({overall[BASELINE_KEY]:.3f} vs {overall[win]:.3f})."}
    return {"decision": "select", "model": win,
            "line": f"Select {_name(r, win)} as the matching model (Balanced winner, Overall Score {overall[win]:.3f}); "
                    f"activate with --activate."}


# ---------------------------------------------------------------- report

def _name(r: dict, key: str | None) -> str:
    return NOT_YET_EVALUATED if key is None else r["models"][key]["display"]


def _fmt(v, digits=3) -> str:
    return NOT_YET_EVALUATED if v is None else f"{v:.{digits}f}"


def render_markdown(r: dict) -> str:
    ev = r["config"]["evaluation"]
    k = ev["primary_k"]
    labels, stamp = r["labels"], r["labels"].get("stamp")
    models = list(r["models"])
    out = [f"# Matching model selection: {r['experiment_id']}", ""]
    if not labels["provided"]:
        out += ["> **NO RELEVANCE LABELS YET: every quality metric below reads \"Not yet evaluated\", no Overall "
                "Score is computed, and no model is selected.** The team's human labels (`/labelling`) are what "
                "Phase 8 selects on; weak labels can't (docs/adr/0014, 0015). Cost, engineering and "
                "explainability below are measured.", ""]
    if stamp:
        out += [f"> **{stamp}. NOT A SELECTION.** The models were trained on weak labels from the same rules, so "
                "high scores here partly mean \"learned the rules back\" (branch_match and career_interest_alignment "
                "encode the same role families and stated domain: docs/adr/0014).", ""]
    out += [f"Generated by `src/scoring/matching_model_selection_engine.py` on {r['finished_at']} "
            f"({r['runtime_seconds']} s). Weights `{r['config']['weights_file']}` ({r['config']['weights_version']}, "
            f"sha256 {r['config']['weights_sha256'][:12]}); candidates `{r['config']['candidates_file']}` "
            f"({r['config']['candidates_version']}, sha256 {r['config']['candidates_sha256'][:12]}). "
            f"Trained artifacts: `{r['artifacts_dir']}` (experiment `{r['artifacts_experiment']}`). "
            f"Dataset {r['dataset_version']}; {r['feature_version']} on {r['embedding_model']} {r['embedding_version']}.", ""]

    t = r["splits"]["training"]
    out += ["## Setup", "",
            f"- **Target:** {r['target']['formulation']}.",
            f"- **Training rows:** {t['label_source']} labels, `{t['split']}` split only: "
            f"{r['measurements'][models[0]]['training']['rows']:,} pairs, "
            f"{r['measurements'][models[0]]['training']['positives']:,} suitable. Grade distribution on the train "
            f"split: {r['target']['train_label_distribution']}. Same rows, seed {t['seed']}, class balance "
            f"`{t['class_balance']}`, {t['threads']} thread, for every model.",
            f"- **Features:** the {len(FEATURE_NAMES)} features-v1 contract features, the same for every model. "
            f"Feature-exclusion check against this training data: "
            f"**{'passed' if r['exclusion_check']['passed'] else 'FAILED'}** (protected columns present: "
            f"{r['exclusion_check']['protected_present'] or 'none'}; columns outside the contract: "
            f"{r['exclusion_check']['extra_columns'] or 'none'}).",
            f"- **Evaluation:** `{r['splits']['selection_split']}` split only, identical pairs for every model; train "
            f"is never evaluated; test is {r['splits']['test']}. Decision threshold {t['decision_threshold']} for "
            "Accuracy/Precision/Recall/F1; ranking metrics rank each student's judged jobs by the model's score.", ""]

    out += ["## Labels", ""]
    if not labels["provided"]:
        out += ["None provided: quality **Not yet evaluated**. Rerun with "
                f"`--labels data/benchmark/human_labels.csv --reuse-models {r['artifacts_experiment']}` once the team "
                "has graded the pool.", ""]
    else:
        out += [f"- File `{labels['path']}` (sha256 {labels['sha256'][:12]}): {labels['pairs']:,} pairs; sources "
                f"{labels['source_breakdown']}; splits {labels['split_breakdown']}.",
                f"- Evaluated: **{labels['used_source']}**, **{labels['used_split']}** only: {labels['used_pairs']:,} "
                f"pairs, {labels['used_students']} students, grades {labels['used_label_distribution']}.",
                f"- **{labels['validity']['line']}**"]
        out += [f"  - {reason}" for reason in labels["validity"]["reasons"]]
        out.append("")

    overall = r["scenarios"]["balanced"]["overall"]
    out += [f"## Matching Model Selection Engine (Balanced weights, K = {k})", ""]
    if stamp:
        out += [f"**{stamp}**", ""]
    out += ["| Model | F1 | NDCG@10 | MRR | Latency | Training Time | Memory | Complexity | Explainability | Overall Score |",
            "|" + "---|" * 10]
    for m in models:
        c = r["criteria"][m]
        score = NOT_YET_EVALUATED if overall is None else f"{overall[m]:.3f}" + (
            " (baseline)" if m == BASELINE_KEY else "" if m in r["eligible"] else " (below baseline)")
        out.append(f"| {_name(r, m)} | {_fmt(c['f1'])} | {_fmt(c['ndcg'])} | {_fmt(c['mrr'])} | "
                   f"{c['latency']:.2f} ms | {c['training_time']:.2f} s | {c['memory']:.1f} MB | {c['complexity']} pt | "
                   f"{c['explainability']}/5 | {score} |")
    out += ["", "Latency: one student x 50 retrieved jobs scored by the loaded artifact, median of 30 trials. Training "
            "time: median of 3 fits, single thread, after an untimed warm-up fit (library imports). Memory: peak resident memory added by the first measured fit (own process). "
            "Complexity and Explainability: point rubrics in `config/matching_model_candidates.yaml` (details below). "
            "Normalisation: min-max across these 7 models, latency/training time/memory on log10.", ""]

    if r["quality"]:
        out += ["## Quality (full suite)", "",
                "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | MRR (95% CI) | NDCG@5 | NDCG@10 (95% CI) | P@5 | P@10 | R@5 | R@10 | vs Baseline 3, NDCG@10 |",
                "|" + "---|" * 15]
        for m in models:
            cl, rk, ci = r["quality"][m]["classification"], r["quality"][m]["ranking"]["mean"], r["confidence"][m]
            vs = r["confidence"]["vs_baseline"].get(m) if m != BASELINE_KEY else None
            vs_text = "—" if not vs else (f"{vs['mean_difference']:+.3f} ({vs['ci95'][0]:+.3f} to {vs['ci95'][1]:+.3f})"
                                         + ("" if vs["distinguishable"] else ", not distinguishable"))
            cim = lambda name: f" ({ci[name][0]:.3f}-{ci[name][1]:.3f})" if ci.get(name) else ""  # noqa: E731
            out.append(f"| {_name(r, m)} | {cl['accuracy']:.3f} | {cl['precision']:.3f} | {cl['recall']:.3f} | "
                       f"{cl['f1']:.3f} | {_fmt(cl['roc_auc'])} | {_fmt(cl['pr_auc'])} | {rk['mrr']:.3f}{cim('mrr')} | "
                       f"{rk['ndcg@5']:.3f} | {rk['ndcg@10']:.3f}{cim('ndcg@10')} | {rk['precision@5']:.3f} | "
                       f"{rk['precision@10']:.3f} | {rk['recall@5']:.3f} | {rk['recall@10']:.3f} | {vs_text} |")
        q0 = r["quality"][models[0]]
        out += ["", f"Pairs: {q0['classification']['pairs']:,} ({q0['classification']['positives']:,} suitable); "
                f"students evaluated {q0['ranking']['students_evaluated']}, excluded (no relevant job) "
                f"{q0['ranking']['students_excluded_no_relevant']}. CIs: bootstrap over students "
                f"({ev['bootstrap_resamples']} resamples). Baseline 3's F1 uses suitability >= 0.5.", ""]
        if q0["ranking"]["degenerate_k"]:
            out += [f"At K in {q0['ranking']['degenerate_k']}, K >= some students' judged jobs, so P@K and R@K can't "
                    "separate models there; NDCG and MRR still can.", ""]
    else:
        out += ["## Quality (full suite)", "",
                "Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MRR, NDCG@K, Precision@K, Recall@K: "
                f"**{NOT_YET_EVALUATED}** for every model (no relevance labels).", ""]

    out += ["## Performance (measured)", "",
            "| Model | Training time, 3 fits (s) | Memory added / peak (MB) | Single pair: median / p95 (ms) | 50 pairs: median / p95 (ms) | Throughput (pairs/s) | Model size on disk | Format |",
            "|" + "---|" * 8]
    for m in models:
        me = r["measurements"][m]
        tr, inf, art = me["training"], me["inference"], me["artifact"]
        peak = "—" if tr["memory_peak_mb"] is None else f"{tr['memory_peak_mb']:.0f}"
        out.append(f"| {_name(r, m)} | {', '.join(f'{s:.2f}' for s in tr['seconds'])} | {tr['memory_added_mb']:.1f} / {peak} | "
                   f"{inf['single_pair_ms']['median']:.3f} / {inf['single_pair_ms']['p95']:.3f} | "
                   f"{inf['batch_50_ms']['median']:.3f} / {inf['batch_50_ms']['p95']:.3f} | "
                   f"{inf['throughput_pairs_per_s']:,.0f} | {art['size_bytes'] / 1024:,.1f} KB | {art['format']} |")
    hw = r["hardware"]
    out += ["", f"Hardware: {hw['system']} ({hw['machine']}), {hw['cpu_count']} logical CPUs, Python {hw['python']}; "
            f"{hw['libraries']}. Throughput over {r['measurements'][models[0]]['inference']['throughput_rows']:,} "
            "held-out pairs (median of 5). Absolute numbers are specific to this machine; the order is what transfers.", ""]

    out += ["## Engineering", "",
            "| Model | Complexity (points) | Reproducible (same seed, refit) | Maintainability (1-5) | Interpretability (1-5) | Scalability (EXTRAPOLATED) |",
            "|---|---|---|---|---|---|"]
    for m in models:
        e, me = r["engineering"][m], r["measurements"][m]
        rep, sc = me["reproducibility"], me["scalability"]
        rep_text = ("yes (bit-identical)" if rep["deterministic"] else f"no (max diff {rep['max_abs_diff_between_refits']:.1e})")
        sc_text = ("nothing to train" if not sc["measured"] else
                   f"~{sc['training_seconds_1M_pairs']:,.0f} s to train on 1M pairs, ~{sc['training_seconds_10M_pairs']:,.0f} s "
                   f"on 10M (time ∝ rows^{sc['exponent']}, fitted to {', '.join(str(p['rows']) for p in sc['measured'])} rows)")
        out.append(f"| {_name(r, m)} | {e['complexity']['points']}: {'; '.join(e['complexity']['reasons'])} | {rep_text} | "
                   f"{e['maintainability']['score']}: {e['maintainability']['reason']} | "
                   f"{e['interpretability']['score']}: {e['interpretability']['reason']} | {sc_text} |")
    out += ["", "Interpretability rubric: 5 = the whole model is readable (coefficients / weights); 4 = readable with "
            "effort (one shallow tree); 3 = partly (few trees); 2 = post-hoc explanations only (ensembles); 1 = opaque "
            "even post hoc. Maintainability rubric: 5 = pure scikit-learn/config, no tuning surface; 4 = pure "
            "scikit-learn with a large or preprocessing-dependent model; 3 = an extra native dependency and model "
            "format; 2 = several; 1 = custom code to keep alive.", ""]

    out += ["## Explainability (checked per model)", "",
            "| Model | SHAP / explainer actually used | Match-level trace (contributions add up to the output) | Explanation cost per pair | Points | Top features (mean abs contribution) |",
            "|---|---|---|---|---|---|"]
    for m in models:
        ex, e = r["measurements"][m]["explainability"], r["engineering"][m]["explainability"]
        top = ", ".join(f"{f} {v:.3f}" for f, v in list(ex["mean_abs_contribution"].items())[:4]) or "the weights (config)"
        ok = ex["additivity_max_abs_error"] < 1e-4
        out.append(f"| {_name(r, m)} | {ex['explainer']} | {'yes' if ok else 'NO'}: max error "
                   f"{ex['additivity_max_abs_error']:.1e} on {ex['pairs_checked']} validation pairs ({ex['output_space']}) | "
                   f"{ex['single_pair_explanation_ms']['median']:.2f} ms | {e['points']}/5: {'; '.join(e['reasons'])} | {top} |")
    out += ["", "Feature-level explanation quality: Logistic Regression's contributions are coefficient x (value - mean) "
            "in log-odds, so the same feature always pushes the same way; a single tree's path can be read as rules; "
            "for the ensembles TreeSHAP is exact for the model but describes 200 trees, so a contribution can change "
            "sign between similar pairs. scikit-learn models are explained on their imputed inputs; a feature's "
            "missing-indicator contribution is added to the feature's own.", ""]

    out += ["## Multi-criteria decision and sensitivity analysis", "",
            f"Gate: a trained model must beat Baseline 3 on NDCG@{k} to be eligible. Overall Score = sum of weight x "
            "normalised criterion. **The winner may differ by scenario**: each scenario's table is shown in full.", ""]
    for name, sc in r["scenarios"].items():
        w = sc["weights"]
        out += [f"### {name.replace('_', ' ').title()}: {sc['description']}",
                "Weights: " + ", ".join(f"{c} {w[c]}" for c in (*QUALITY, *COST)) + ".", ""]
        if sc["overall"] is None:
            out += [f"Ranking: {NOT_YET_EVALUATED} (needs relevance labels).", ""]
            continue
        out += ["| Rank | Model | Overall Score | Eligible |", "|---|---|---|---|"]
        for rank, m in enumerate(sc["ranking"], start=1):
            status = "baseline" if m == BASELINE_KEY else ("yes" if m in r["eligible"] else "no (below baseline)")
            out.append(f"| {rank} | {_name(r, m)} | {sc['overall'][m]:.3f} | {status} |")
        out += ["", f"Winner: **{_name(r, sc['winner']) if sc['winner'] else 'none (no eligible model)'}**", ""]
    if r["winner_changes"] is None:
        out += [f"Does the winner change across scenarios? **{NOT_YET_EVALUATED}**.", ""]
    else:
        winners = {n: s["winner"] for n, s in r["scenarios"].items()}
        out += [f"Does the winner change across scenarios? **{'YES' if r['winner_changes'] else 'No'}** ("
                + "; ".join(f"{n}: {_name(r, w) if w else 'none'}" for n, w in winners.items()) + ").", ""]
    if r["random_weight_stability"]:
        s = r["config"]["sensitivity"]
        out += [f"Random weightings ({s['random_weightings']} Dirichlet draws centred on Balanced, concentration "
                f"{s['concentration']}, seed {s['seed']}): share won by each eligible model: "
                + ", ".join(f"{_name(r, m)} {p}%" for m, p in r["random_weight_stability"].items()) + ".", ""]
    if r["simple_model_check"]:
        out += [f"**Simpler model check** (best of {', '.join(_name(r, m) for m in r['config']['simple_models'])} by "
                f"NDCG@{k}; \"close enough\" = within {r['config']['close_enough_margin']} or not statistically "
                "distinguishable):", ""]
        for model, sm in r["simple_model_check"].items():
            if sm["same"]:
                out.append(f"- {_name(r, model)}: is itself a simple model.")
                continue
            d = sm["paired"]
            ci = (f" (paired 95% CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}, "
                  f"{'distinguishable' if d['distinguishable'] else 'not distinguishable'})") if d.get("ci95") else ""
            out.append(f"- {_name(r, model)} vs {_name(r, sm['simple'])}: {sm['ndcg_gap']:+.3f} NDCG@{k}{ci}, "
                       f"{sm['f1_gap']:+.3f} F1; {sm['latency_ratio']:.1f}x the latency, {sm['complexity_gap']:+d} "
                       f"complexity and {sm['explainability_gap']:+d} explainability points. "
                       + ("**Close enough: the simpler model is a reasonable choice; weigh the complexity "
                          "trade-off explicitly.**" if sm["close_enough"]
                          else "Not close: the extra complexity buys measurable quality here."))
        out.append("")
    out += [f"**Decision (rule fixed in docs/adr/0015): {r['recommendation']['line']}**", ""]
    lgb = "lightgbm" in r["models"]
    out += ["## Not evaluated", "", "- None: all six candidates ran" + ("" if lgb else " (LightGBM: Not evaluated — dependency unavailable)") + ".", ""]
    out += ["## Limitations", "",
            "- Training labels are weak (rule-based): every model learns the weak-label rules, including their "
            "leakage through branch_match and career_interest_alignment. Only human labels can show whether that "
            "generalises to real fit.",
            "- Hyperparameters are fixed, not tuned, so a model's rank could change with tuning; tuning needs "
            "human-labelled training data that doesn't exist yet.",
            "- Suitability scores are class-balanced probabilities: a ranking score, not a calibrated probability.",
            "- The job corpus is synthetic and templated. Costs are from one machine.",
            "- Quality is on the validation split only; test is held back for Phase 14."]
    if stamp:
        out += ["", f"> **{stamp}**"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- outputs

def write_outputs(report: dict) -> dict[str, str]:
    folder = RESULTS / "matching_model_evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / report["experiment_id"]
    base.with_suffix(".json").write_text(json.dumps({k: v for k, v in report.items() if k != "markdown"}, indent=1,
                                                    default=str), encoding="utf-8")
    base.with_suffix(".md").write_text(report["markdown"], encoding="utf-8")
    source = report["labels"].get("used_source")
    latest = RESULTS / ("matching_model_evaluation" + (f"_{source}" if source else ""))
    shutil.copyfile(base.with_suffix(".json"), latest.with_suffix(".json"))
    shutil.copyfile(base.with_suffix(".md"), latest.with_suffix(".md"))
    return {"json": str(base.with_suffix(".json")), "markdown": str(base.with_suffix(".md")), "latest": str(latest.with_suffix(".md"))}


def _set(text: str, key: str, value: str) -> str:
    text, n = re.subn(rf"(?m)^({key}: ).*$", rf"\g<1>{value}", text)
    if n != 1:
        raise RuntimeError(f"{ACTIVE_CONFIG}: no `{key}:` line")
    return text


def register_artifacts(experiment_id: str) -> bool:
    """After a training run: make its artifacts the ones a trained model would be loaded from. Refused while a
    trained model is active (that would silently swap the active model's artifacts)."""
    text = ACTIVE_CONFIG.read_text(encoding="utf-8")
    active = re.search(r"(?m)^active: (.*)$", text).group(1).strip()
    if active != BASELINE_KEY:
        print(f"NOT registering {experiment_id}: {active} is active; select again on human labels first")
        return False
    ACTIVE_CONFIG.write_text(_set(text, "artifacts_experiment", experiment_id), encoding="utf-8")
    return True


def activate(report: dict) -> None:
    """Write the selected model into config/matching_model.yaml (only after a 'select' decision)."""
    rec = report["recommendation"]
    if rec["decision"] != "select":
        raise SystemExit(f"--activate refused: the decision is {rec['decision']!r}, not 'select'")
    text = ACTIVE_CONFIG.read_text(encoding="utf-8")
    primary = (yaml.safe_load(text) or {}).get("embedding_model")
    values = (("active", rec["model"]), ("artifacts_experiment", report["artifacts_experiment"]),
              ("selection_experiment", report["experiment_id"]))
    if primary in (None, report.get("embedding_model", primary)):
        for key, value in values:
            text = _set(text, key, value)
    else:  # a run on another embedding model sets that model's fallback entry, never the primary choice
        head, sep, block = text.partition(f"\n  {report['embedding_model']}:\n")
        if not sep:
            raise SystemExit(f"--activate refused: {ACTIVE_CONFIG} has no fallback entry for "
                             f"{report['embedding_model']!r}")
        for key, value in values:
            block, n = re.subn(rf"(?m)^(    {key}: ).*$", rf"\g<1>{value}", block, count=1)
            if n != 1:
                raise RuntimeError(f"{ACTIVE_CONFIG}: no `{key}:` line under fallback.{report['embedding_model']}")
        text = head + sep + block
    ACTIVE_CONFIG.write_text(text, encoding="utf-8")


def record_experiment(dsn: str, report: dict) -> None:
    """experiment_runs / experiment_metrics (spec §25), as Phase 6 and Phase 7 do."""
    import psycopg
    from psycopg.types.json import Jsonb

    ev, labels = report["config"]["evaluation"], report["labels"]
    rows = []
    for m in report["models"]:
        version = report["measurements"][m]["artifact"]["sha256"][:12]
        q = report["quality"].get(m)
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            v = q["classification"][metric] if q else None
            rows.append((m, version, metric, ev["selection_split"], v, "measured" if v is not None else "not_yet_evaluated"))
        for metric in ("mrr", "ndcg@5", "ndcg@10", "precision@5", "precision@10", "recall@5", "recall@10"):
            v = q["ranking"]["mean"].get(metric) if q else None
            rows.append((m, version, metric, ev["selection_split"], v, "measured" if v is not None else "not_yet_evaluated"))
        c = report["criteria"][m]
        for metric in ("latency", "training_time", "memory", "complexity", "explainability"):
            rows.append((m, version, metric, "n/a", float(c[metric]), "measured"))
        overall = report["scenarios"]["balanced"]["overall"]
        rows.append((m, version, "overall_score_balanced", "n/a", overall[m] if overall else None,
                     "measured" if overall else "not_yet_evaluated"))
    data = {k: v for k, v in report.items() if k != "markdown"}
    with psycopg.connect(dsn, connect_timeout=5) as conn, conn.transaction():
        conn.execute(
            "INSERT INTO experiment_runs (experiment_id, kind, started_at, finished_at, dataset_version, dataset_size, "
            "train_split, validation_split, test_split, random_seed, feature_version, embedding_model, embedding_version, "
            "prediction_model, hyperparameters, metrics, runtime_seconds, memory, hardware, labels, notes, limitations, "
            "report) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (report["experiment_id"], report["kind"], report["started_at"], report["finished_at"],
             report["dataset_version"], Jsonb({"feature_matrix_pairs": report["feature_matrix"]["pairs"]}),
             Jsonb({"evaluated": False, "used_for_training": report["splits"]["training"]}),
             Jsonb({"evaluated": labels["provided"]}), Jsonb({"evaluated": False, "note": "held back for Phase 14"}),
             ev["seed"], report["feature_version"], report["embedding_model"], report["embedding_version"],
             "comparison: " + ", ".join(report["models"]),
             Jsonb({m: report["models"][m]["params"] for m in report["models"]}),
             Jsonb({"quality": report["quality"], "criteria": report["criteria"]}), report["runtime_seconds"],
             Jsonb({m: report["measurements"][m]["training"]["memory_added_mb"] for m in report["models"]}),
             Jsonb(report["hardware"]), Jsonb(labels | {"validity_line": report["validity"]}),
             f"Phase 8 matching-model selection. Artifacts: {report['artifacts_dir']}. {report['recommendation']['line']}",
             "Weak-label training; fixed hyperparameters; validation split only; one machine. See the report.",
             Jsonb(json.loads(json.dumps(data, default=str)))))
        conn.cursor().executemany(
            "INSERT INTO experiment_metrics (experiment_id, model_name, model_version, metric, split, value, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)", [(report["experiment_id"], *row) for row in rows])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Phase 8 Matching Model Selection Engine")
    ap.add_argument("--features", default=None, help="feature matrix (default: newest in data/benchmark/matching_features)")
    ap.add_argument("--labels", default=None, help="labels CSV (student_id,job_id,relevance_label,label_source,split)")
    ap.add_argument("--label-source", default=None)
    ap.add_argument("--reuse-models", default=None, metavar="EXPERIMENT_ID",
                    help="evaluate the artifacts in models/matching/<id>/ instead of training new ones")
    ap.add_argument("--record-db", action="store_true")
    ap.add_argument("--activate", action="store_true", help="write a 'select' decision into config/matching_model.yaml")
    ap.add_argument("--record-existing", default=None, metavar="REPORT_JSON",
                    help="store an already written report in Postgres and exit")
    args = ap.parse_args(argv)
    dsn = os.environ.get("EXPERIMENTS_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if args.record_existing:
        if not dsn:
            raise SystemExit("--record-existing needs EXPERIMENTS_DATABASE_URL or DATABASE_URL")
        existing = json.loads(Path(args.record_existing).read_text(encoding="utf-8"))
        record_experiment(dsn, existing)
        raise SystemExit(f"recorded {existing['experiment_id']} in Postgres")
    if not args.labels:
        print("No labels: quality metrics will read 'Not yet evaluated' (cost, engineering, explainability are measured).")
    report = run(Path(args.features) if args.features else latest_features(), args.labels, args.reuse_models,
                 args.label_source)
    paths = write_outputs(report)
    if not args.reuse_models and register_artifacts(report["experiment_id"]):
        print(f"registered artifacts {report['experiment_id']} in {ACTIVE_CONFIG} (active model unchanged)")
    if args.record_db:
        if not dsn:
            raise SystemExit("--record-db needs EXPERIMENTS_DATABASE_URL or DATABASE_URL")
        record_experiment(dsn, report)
        print(f"recorded {report['experiment_id']} in Postgres")
    if args.activate:
        activate(report)
        print(f"activated {report['recommendation']['model']} in {ACTIVE_CONFIG}")
    print(report["markdown"])
    print(f"written: {paths}")


if __name__ == "__main__":
    main()
