"""Model Evaluation Engine (Phase 6, spec §9): one table, transparent weights, sensitivity analysis.

Every model is evaluated on the same benchmark: the same students, jobs, texts and split. Quality
comes only from relevance labels on the selection split (validation). Train rows are never
evaluated, and test is held back for Phase 14. Cost comes from isolated, repeated measurements
(src/embeddings/embedding_model_benchmark.py). Raw cosine similarity is never used as quality:
its magnitude is not comparable across model families (the report shows each model's range).

Usage:
  python -m src.embeddings.model_evaluation_engine                   no labels: quality "Not yet evaluated"
  python -m src.embeddings.model_evaluation_engine --labels data/benchmark/human_labels.csv
  python -m src.embeddings.model_evaluation_engine --labels data/benchmark/weak_labels.csv
    --device mps|cpu   scoring device (default mps when available: scores agree with CPU to about 1e-7)
    --reuse-cost       reuse results/embedding_cost_measurements.json instead of measuring again
    --record-db        also store the experiment in Postgres (reads EXPERIMENTS_DATABASE_URL)
    --record-existing results/embedding_evaluation/<id>.json   store a finished run without re-running it
Outputs: results/embedding_evaluation/<experiment_id>.{md,json}, plus a copy of the latest run per label
source at results/embedding_model_evaluation[_<source>].{md,json}.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from src.embeddings.embedding_model_benchmark import (CANDIDATES, DEFAULT_JOBS, DEFAULT_STUDENTS, FLOOR,
                                                      MODEL_CONFIGS, NOT_RUN_MODELS, PROTOCOLS, SECTION_PAIRS,
                                                      SECTION_WEIGHTS, as_pair_dict, complexity, dependency_footprint,
                                                      hardware, load_benchmark_data, load_encoder, measure_isolated,
                                                      score_matrix)
from src.evaluation import NOT_YET_EVALUATED
from src.evaluation.relevance_labels import STAMPS, circularity_check, load_labels
from src.evaluation.retrieval_metrics import bootstrap_ci, evaluate, paired_difference
from src.evaluation.splits import SPLIT_VERSION, SPLITS, split_for

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS = os.path.join(ROOT, "config", "model_selection_weights.yaml")
RESULTS = os.path.join(ROOT, "results")
COST_CACHE = os.path.join(RESULTS, "embedding_cost_measurements.json")
TEXT_VIEW_VERSION = "text-view-v2"  # benchmark texts after the I-3/I-4 fixes
NOT_FEASIBLE = NOT_RUN_MODELS["custom"]["status"]

QUALITY = ("mrr", "ndcg", "recall", "precision")
COST = ("latency", "memory", "size", "dimension", "complexity", "scalability")
HIGHER_IS_BETTER = set(QUALITY)
RUNNABLE = (FLOOR, *CANDIDATES)


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_config(path: str = WEIGHTS) -> dict:
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
        raise ValueError("selection_split can't be train: the spec forbids evaluating on train rows")
    cfg["sha256"] = _sha(path)
    return cfg


# ---------------------------------------------------------------- multi-criteria decision

def normalise(criteria: dict[str, dict[str, float | None]], method: str = "min_max", log_scale=()) -> dict:
    """{criterion: {model: score}} where 1 is best (min-max) or a direction-aligned z-score.
    A criterion with any missing value is left out (it can't be compared)."""
    out = {}
    names = sorted({c for v in criteria.values() for c in v})
    for c in names:
        values = {m: criteria[m].get(c) for m in criteria}
        if any(v is None for v in values.values()):
            continue
        x = {m: (math.log10(v) if c in log_scale else v) for m, v in values.items()}
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
    """Weighted sum over criteria. None when a weighted criterion couldn't be normalised (e.g. no labels)."""
    if any(w > 0 and c not in normalised for c, w in weights.items()):
        return None
    models = next(iter(normalised.values())).keys()
    return {m: float(sum(w * normalised[c][m] for c, w in weights.items() if w > 0)) for m in models}


def pick_winner(scores: dict[str, float] | None, eligible: list[str]) -> str | None:
    if not scores:
        return None
    pool = [m for m in eligible if m in scores]
    return max(pool, key=lambda m: (scores[m], m)) if pool else None


def random_weight_stability(normalised: dict, balanced: dict[str, float], eligible: list[str], n: int,
                            concentration: float, seed: int) -> dict[str, float] | None:
    names = list(balanced)
    if overall_scores(normalised, balanced) is None or not eligible:
        return None
    rng = np.random.default_rng(seed)
    alpha = np.array([balanced[c] for c in names]) * concentration
    wins = {m: 0 for m in eligible}
    for draw in rng.dirichlet(alpha, size=n):
        winner = pick_winner(overall_scores(normalised, dict(zip(names, draw))), eligible)
        wins[winner] += 1
    return {m: round(100 * k / n, 1) for m, k in sorted(wins.items(), key=lambda kv: -kv[1])}


# ---------------------------------------------------------------- measurements

def measure_costs(reuse: bool) -> dict:
    if reuse and os.path.exists(COST_CACHE):
        with open(COST_CACHE, encoding="utf-8") as fh:
            costs = json.load(fh)
        for key in RUNNABLE:  # a model that is the same network as a measured one (cost_as) reuses its cost
            same = MODEL_CONFIGS[key].get("cost_as")
            if key not in costs["models"] and same in costs["models"]:
                encoder = load_encoder(key, "cpu")
                costs["models"][key] = {**costs["models"][same], "model": key, "version": encoder.version,
                                        "revision": encoder.revision,
                                        "cost_basis": f"same network as {same} (fine-tuned weights): its measured "
                                                      f"cost is reused, not re-measured"}
                del encoder
        return costs
    costs = {"measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "hardware": hardware(),
             "models": {}}
    for key in RUNNABLE:
        print(f"  measuring cost: {key} (isolated process, offline)")
        costs["models"][key] = measure_isolated(key)
    os.makedirs(RESULTS, exist_ok=True)
    with open(COST_CACHE, "w", encoding="utf-8") as fh:
        json.dump(costs, fh, indent=2)
    return costs


def engineering(key: str, cost: dict) -> dict:
    cfg = MODEL_CONFIGS[key]
    deps = dependency_footprint(cfg["runtime_packages"])
    throughput = cost["cpu"]["throughput_texts_per_s"]
    return {
        "local_feasibility": ("yes: ran fully offline on CPU (HF_HUB_OFFLINE=1)" if cost.get("offline")
                              else "not verified offline"),
        "dependencies": deps,
        "preprocessing": ("query prefix " + repr(cfg["query_prefix"]) if cfg["query_prefix"] else "plain query text")
                         + "; " + ("passage prefix " + repr(cfg["passage_prefix"]) if cfg["passage_prefix"] else "plain passage text"),
        "complexity": complexity(key),
        "truncation": cost["truncation"],
        "scalability_extrapolated": {
            "cpu_minutes_per_100k_jobs": round(100_000 / throughput / 60, 1),
            "storage_mb_per_100k_vectors": round(cost["bytes_per_vector"] * 100_000 / 1024**2, 1),
            "basis": f"extrapolated linearly from the measured CPU throughput ({throughput} texts/s, batch of "
                     f"{cost['cpu']['batch_latency_ms']['batch_size']}); not measured at that scale",
        },
        "reproducibility": {"revision": cost["revision"],
                            "deterministic_on_cpu": cost["deterministic_cpu_max_abs_diff"] == 0.0,
                            "cpu_max_abs_diff_between_runs": cost["deterministic_cpu_max_abs_diff"],
                            "cpu_vs_mps_max_abs_diff": (cost["mps"].get("cpu_vs_mps_max_abs_diff")
                                                        if isinstance(cost["mps"], dict) else None)},
    }


def score_all(device: str, protocols, extra_job_ids=()) -> tuple[dict, dict, dict]:
    """Scores for every student x distinct job (plus any other labelled job), per model and protocol, and each
    model's raw cosine range."""
    students, jobs, all_jobs = load_benchmark_data()
    extra = all_jobs[all_jobs["job_id"].isin(set(extra_job_ids) - set(jobs["job_id"]))]
    jobs = pd.concat([jobs, extra], ignore_index=True)
    scores, versions, ranges = {p: {} for p in protocols}, {}, {}
    for key in RUNNABLE:
        print(f"  scoring: {key}")
        encoder = load_encoder(key, "cpu" if MODEL_CONFIGS[key]["kind"] == "tfidf" else device)
        versions[key] = encoder.version
        for protocol in protocols:
            sids, jids, m = score_matrix(encoder, students, jobs, protocol)
            scores[protocol][key] = as_pair_dict(sids, jids, m)
            if protocol == "full_text":
                v = m[~np.isnan(m)]
                ranges[key] = {"mean": round(float(v.mean()), 3), "sd": round(float(v.std()), 3),
                               "p5": round(float(np.percentile(v, 5)), 3), "p95": round(float(np.percentile(v, 95)), 3)}
        del encoder
    return scores, versions, ranges


# ---------------------------------------------------------------- the run

def run(labels_path: str | None, device: str, reuse_cost: bool, label_source: str | None = None,
        config_path: str = WEIGHTS) -> dict:
    started = time.time()
    started_at = datetime.now(timezone.utc)
    cfg = load_config(config_path)
    ev = cfg["evaluation"]
    ks, k, threshold, split = ev["ks"], ev["primary_k"], ev["relevance_threshold"], ev["selection_split"]
    students, jobs, all_jobs = load_benchmark_data()
    split_counts = students["student_id"].map(split_for).value_counts().to_dict()

    # labels (validated loudly) ---------------------------------------------------------------
    labels_info = {"provided": False, "path": None}
    eval_labels, source = None, None
    labelset = None
    if labels_path:
        labelset = load_labels(labels_path, DEFAULT_STUDENTS, DEFAULT_JOBS)
        source = label_source or labelset.preferred_source()
        eval_labels = labelset.select(source, [split])
        labels_info = {
            "provided": True, "path": os.path.relpath(labels_path, ROOT), "sha256": labelset.sha256,
            "pairs": int(len(labelset.frame)), "students": int(labelset.frame["student_id"].nunique()),
            "source_breakdown": labelset.source_breakdown(), "split_breakdown": labelset.split_breakdown(),
            "label_distribution": labelset.label_distribution(), "used_source": source,
            "used_split": split, "used_pairs": int(len(eval_labels)),
            "used_students": int(eval_labels["student_id"].nunique()),
            "train_rows_evaluated": 0, "stamp": STAMPS.get(source),
            "manifest": labelset.manifest,
        }
        if eval_labels.empty:
            raise ValueError(f"no {source} labels in the {split!r} split: nothing to evaluate")

    # cost -------------------------------------------------------------------------------------
    costs = measure_costs(reuse_cost)
    for key in RUNNABLE:
        if "error" in costs["models"].get(key, {"error": "missing"}):
            raise RuntimeError(f"cost measurement failed for {key}: {costs['models'].get(key)}")
    eng = {key: engineering(key, costs["models"][key]) for key in RUNNABLE}

    # quality ------------------------------------------------------------------------------------
    protocols = PROTOCOLS if eval_labels is not None else ("full_text",)
    scores, versions, ranges = score_all(device, protocols,
                                         eval_labels["job_id"].unique() if eval_labels is not None else ())
    quality = {p: {} for p in protocols}
    validity = None
    if eval_labels is not None:
        for p in protocols:
            for key in RUNNABLE:
                quality[p][key] = evaluate(scores[p][key], eval_labels, ks, threshold)
        label_pairs = set(zip(eval_labels["student_id"], eval_labels["job_id"]))
        validity = circularity_check(labelset, source, list(RUNNABLE),
                                     {m: {pair: v for pair, v in scores["full_text"][m].items() if pair in label_pairs}
                                      for m in RUNNABLE})
        labels_info["validity"] = validity

    primary = ev["primary_protocol"]
    q = quality.get(primary, {})

    def qmetric(key, name, kk=k):
        if key not in q:
            return None
        metric = "mrr" if name == "mrr" else f"{name}@{kk}"
        return q[key]["mean"].get(metric)

    # criteria matrix, eligibility, scores ------------------------------------------------------
    criteria = {}
    for key in RUNNABLE:
        c = costs["models"][key]
        criteria[key] = {
            "mrr": qmetric(key, "mrr"), "ndcg": qmetric(key, "ndcg"), "recall": qmetric(key, "recall"),
            "precision": qmetric(key, "precision"),
            "latency": c["cpu"]["single_pair_latency_ms"]["median"], "memory": max(c["memory_mb"]["peak"], 1.0),
            "size": max(c["size_mb"], 0.01), "dimension": c["bytes_per_vector"],
            # +1 keeps the value positive should complexity ever be put on the log scale; the order is unchanged
            "complexity": eng[key]["complexity"]["points"] + 1,
            "scalability": eng[key]["scalability_extrapolated"]["cpu_minutes_per_100k_jobs"],
        }
    normalised = normalise(criteria, cfg["normalization"]["method"], cfg["normalization"]["log_scale"])
    gate_metric = cfg["gate"]["primary_quality_metric"]
    if q:
        floor_q = criteria[FLOOR][gate_metric]
        eligible = [m for m in CANDIDATES if not cfg["gate"]["must_beat_floor"] or criteria[m][gate_metric] > floor_q]
    else:
        eligible = list(CANDIDATES)

    scenarios = {}
    for name, sc in cfg["scenarios"].items():
        weights = {c: v["weight"] for c, v in sc["weights"].items()}
        overall = overall_scores(normalised, weights)
        ranking = (sorted(overall, key=lambda m: -overall[m]) if overall else None)
        scenarios[name] = {"description": sc["description"], "weights": weights,
                           "reasons": {c: v["reason"] for c, v in sc["weights"].items()},
                           "overall": overall, "ranking": ranking, "winner": pick_winner(overall, eligible)}
    winners = {n: s["winner"] for n, s in scenarios.items()}
    known = [w for w in winners.values() if w]
    sens = cfg["sensitivity"]
    stability = random_weight_stability(normalised, scenarios["balanced"]["weights"], eligible,
                                        sens["random_weightings"], sens["concentration"], sens["seed"])

    # local deployment: cost-only composite among offline-capable (and, with labels, eligible) candidates
    local_w = {c: v["weight"] for c, v in cfg["local_deployment"].items()}
    local_scores = overall_scores({c: normalised[c] for c in local_w}, local_w)
    local_pool = [m for m in eligible if costs["models"][m].get("offline")]

    # statistics: CIs and paired comparisons -----------------------------------------------------
    stats = {}
    if q:
        boot = ev["bootstrap_resamples"]
        primary_metric = f"{gate_metric}@{k}"
        for key in RUNNABLE:
            per = q[key]["per_student"]
            stats[key] = {m: bootstrap_ci([v[m] for v in per.values()], boot, ev["seed"])
                          for m in ("mrr", f"ndcg@{k}", f"recall@{k}", f"precision@{k}")}
        by_quality = sorted(CANDIDATES, key=lambda m: -criteria[m][gate_metric])
        per_metric = {m: {s: v[primary_metric] for s, v in q[m]["per_student"].items()} for m in RUNNABLE}
        stats["best_vs_runner_up"] = {"best": by_quality[0], "runner_up": by_quality[1], "metric": primary_metric,
                                      **(paired_difference(per_metric[by_quality[0]], per_metric[by_quality[1]],
                                                           boot, ev["seed"]) or {})}
        stats["vs_floor"] = {m: paired_difference(per_metric[m], per_metric[FLOOR], boot, ev["seed"]) for m in CANDIDATES}

    # sensitivity to the scoring protocol: the Balanced decision recomputed with the other protocol's quality
    protocol_check = None
    other = [p for p in quality if p != primary and quality[p]]
    if q and other:
        p2 = other[0]
        crit2 = {m: dict(criteria[m]) for m in RUNNABLE}
        for m in RUNNABLE:
            mean2 = quality[p2][m]["mean"]
            crit2[m].update(mrr=mean2["mrr"], ndcg=mean2[f"ndcg@{k}"], recall=mean2[f"recall@{k}"],
                            precision=mean2[f"precision@{k}"])
        norm2 = normalise(crit2, cfg["normalization"]["method"], cfg["normalization"]["log_scale"])
        elig2 = [m for m in CANDIDATES
                 if not cfg["gate"]["must_beat_floor"] or crit2[m][gate_metric] > crit2[FLOOR][gate_metric]]
        overall2 = overall_scores(norm2, scenarios["balanced"]["weights"])
        protocol_check = {"protocol": p2, "eligible": elig2, "overall": overall2, "winner": pick_winner(overall2, elig2)}

    # the cheapest candidate on the local-deployment criteria, before the quality gate is applied
    offline = [m for m in CANDIDATES if costs["models"][m].get("offline")]
    cheapest_local = max(offline, key=lambda m: local_scores[m]) if local_scores and offline else None
    best = best_model_lines(criteria, q, k, gate_metric, eligible, scenarios, winners, stability, local_scores,
                            local_pool, stats, costs, cheapest_local)

    finished_at = datetime.now(timezone.utc)
    tag = (source or "nolabels").split("_")[0]
    experiment_id = f"emb-{started_at:%Y%m%dT%H%M%SZ}-{tag}"
    report = {
        "experiment_id": experiment_id, "kind": "embedding_selection",
        "started_at": started_at.isoformat(timespec="seconds"), "finished_at": finished_at.isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - started, 1),
        "dataset_version": f"students:{_sha(DEFAULT_STUDENTS)[:12]} jobs:{_sha(DEFAULT_JOBS)[:12]}",
        "dataset_size": {"students": int(len(students)), "jobs": int(len(all_jobs)), "distinct_jobs": int(len(jobs))},
        "splits": {"version": SPLIT_VERSION, "students": {s: int(split_counts.get(s, 0)) for s in SPLITS},
                   "selection_split": split, "train_evaluated": False},
        "config": {"weights_file": os.path.relpath(config_path, ROOT), "weights_version": cfg["version"],
                   "weights_sha256": cfg["sha256"], "evaluation": ev, "normalization": cfg["normalization"],
                   "gate": cfg["gate"], "text_view": TEXT_VIEW_VERSION, "section_weights": SECTION_WEIGHTS,
                   "section_pairs": SECTION_PAIRS, "scoring_device": device, "sensitivity": cfg["sensitivity"]},
        "labels": labels_info, "validity": (validity or {}).get("line", NOT_YET_EVALUATED),
        "models": {key: {"slot": MODEL_CONFIGS[key]["spec_slot"], "display": MODEL_CONFIGS[key]["display"],
                         "version": versions[key], "role": "floor" if key == FLOOR else "candidate"} for key in RUNNABLE},
        "not_run": {k: v for k, v in NOT_RUN_MODELS.items() if not (k == "custom" and "e5_domain" in RUNNABLE)},
        "criteria": criteria, "normalised": normalised, "eligible": eligible,
        "quality": {p: {m: {kk: v for kk, v in r.items() if kk not in ("per_student", "kendall_tau_per_student")}
                        for m, r in quality[p].items()} for p in quality},
        "confidence": stats, "costs": costs, "engineering": eng, "cosine_ranges": ranges,
        "scenarios": scenarios, "winner_changes": len(set(known)) > 1 if known else None,
        "protocol_sensitivity": protocol_check,
        "random_weight_stability": stability, "local_deployment": {"weights": local_w, "scores": local_scores},
        "best_models": best,
    }
    report["recommendation"] = recommend(report)
    report["markdown"] = render_markdown(report)
    return report


def _name(key: str | None) -> str:
    if key is None:
        return NOT_YET_EVALUATED
    return f"{MODEL_CONFIGS[key]['spec_slot']}. {MODEL_CONFIGS[key]['display']}" if key != FLOOR else MODEL_CONFIGS[key]["display"]


def _vs_floor(stats, model, k) -> str:
    d = (stats.get("vs_floor") or {}).get(model)
    if not d or not d.get("ci95"):
        return ""
    verdict = "distinguishable" if d["distinguishable"] else "not distinguishable"
    return (f"; vs the TF-IDF floor {d['mean_difference']:+.3f} NDCG@{k} (95% CI {d['ci95'][0]:+.3f} to "
            f"{d['ci95'][1]:+.3f}), {verdict}")


def best_model_lines(criteria, q, k, gate_metric, eligible, scenarios, winners, stability, local_scores, local_pool,
                     stats, costs, cheapest_local=None) -> dict[str, str]:
    lines = {}
    floor = criteria[FLOOR]
    if q:
        if eligible:
            top = max(eligible, key=lambda m: (criteria[m][gate_metric], criteria[m]["mrr"]))
            ci = stats[top].get(f"{gate_metric}@{k}")
            ci_text = f" (95% CI {ci[0]:.3f}-{ci[1]:.3f})" if ci else ""
            cmp = stats.get("best_vs_runner_up", {})
            if cmp.get("ci95"):
                verdict = ("distinguishable from the runner-up" if cmp["distinguishable"]
                           else "NOT distinguishable from the runner-up at this sample size")
                cmp_text = (f"; {verdict} ({_name(cmp['runner_up'])}, difference {cmp['mean_difference']:+.3f}, "
                            f"95% CI {cmp['ci95'][0]:+.3f} to {cmp['ci95'][1]:+.3f})")
            else:
                cmp_text = "; too few students to compare it with the runner-up"
            lines["quality"] = (f"{_name(top)}: NDCG@{k} {criteria[top][gate_metric]:.3f}{ci_text}{cmp_text}"
                                f"{_vs_floor(stats, top, k)}")
        else:
            lines["quality"] = f"none: no embedding model beat the TF-IDF floor (NDCG@{k} {floor[gate_metric]:.3f})"
    else:
        lines["quality"] = f"{NOT_YET_EVALUATED} (no relevance labels)"
    fastest = min(CANDIDATES, key=lambda m: criteria[m]["latency"])
    lines["speed"] = (f"{_name(fastest)}: {criteria[fastest]['latency']:.1f} ms per student-job pair on CPU "
                      f"(median of {costs['models'][fastest]['cpu']['single_pair_latency_ms']['trials']} warm trials; "
                      f"TF-IDF floor {floor['latency']:.1f} ms)")
    lightest = min(CANDIDATES, key=lambda m: criteria[m]["memory"])
    lines["memory"] = (f"{_name(lightest)}: {criteria[lightest]['memory']:.0f} MB peak added memory "
                       f"(TF-IDF floor {floor['memory']:.0f} MB)")
    excluded = ""
    if q and cheapest_local and cheapest_local not in local_pool:
        excluded = (f". {_name(cheapest_local)} has the cheapest profile ({local_scores[cheapest_local]:.2f}) but is "
                    f"below the TF-IDF floor on NDCG@{k} ({criteria[cheapest_local][gate_metric]:.3f} vs "
                    f"{floor[gate_metric]:.3f}{_vs_floor(stats, cheapest_local, k)})")
    if local_scores and local_pool:
        local = max(local_pool, key=lambda m: local_scores[m])
        gate = "" if q else "; quality gate (must beat TF-IDF) not yet evaluated"
        lines["local_deployment"] = (f"{_name(local)}: best cost composite among eligible models ({local_scores[local]:.2f}"
                                     f"; latency, memory, size, complexity; 1 = cheapest), runs offline on CPU{gate}"
                                     f"{excluded}")
    else:
        lines["local_deployment"] = "none: no candidate both runs offline and beats the TF-IDF floor" + excluded
    win = winners.get("balanced")
    if not q:
        lines["overall"] = (f"{NOT_YET_EVALUATED} (needs relevance labels: the Overall Score uses label-based "
                            "quality, never mean similarity)")
    elif win is None:
        lines["overall"] = "none: no embedding model beat the TF-IDF floor; keep TF-IDF (ADR 0004)"
    else:
        same = sum(1 for w in winners.values() if w == win)
        balanced = scenarios["balanced"]["overall"]
        lines["overall"] = (f"{_name(win)}: Overall Score {balanced[win]:.3f}; the winner in "
                            f"{same} of {len(winners)} weight scenarios and {stability.get(win, 0):.0f}% of "
                            f"random weightings (among eligible models)")
        if balanced[FLOOR] > balanced[win]:
            lines["overall"] += (f". But the TF-IDF floor scores higher overall ({balanced[FLOOR]:.3f}): its cost "
                                 f"advantage outweighs the quality gap{_vs_floor(stats, win, k)}")
    return lines


def recommend(r: dict) -> dict:
    """The selection rule fixed in ADR 0012 before any human label existed, applied to a finished report.

    Only human_reviewed labels with evaluation validity OK can select a model. The Balanced winner is selected
    unless the TF-IDF floor scores higher overall AND the winner's quality gain over the floor is not
    statistically distinguishable: then the evidence says keep TF-IDF, which is a valid outcome.
    """
    labels = r["labels"]
    source, win = labels.get("used_source"), r["scenarios"]["balanced"]["winner"]
    keep = "Keep the TF-IDF baseline (ADR 0004)"
    if not labels["provided"]:
        return {"decision": "no_selection", "model": None,
                "line": f"Insufficient evidence: no relevance labels. {keep} until the human-label run."}
    level = labels["validity"]["level"]
    if source != "human_reviewed":
        return {"decision": "no_selection", "model": None,
                "line": f"Not a selection: {source} labels can't select a model (evaluation validity {level}). "
                        f"Provisional leader on these labels: {_name(win) if win else 'none'}. {keep} until the "
                        "human-label run."}
    if level != "OK":
        return {"decision": "no_selection", "model": None,
                "line": f"Not a selection: evaluation validity {level} ({'; '.join(labels['validity']['reasons'])}). {keep}."}
    if win is None:
        return {"decision": "keep_floor", "model": FLOOR,
                "line": f"{keep}: no embedding model beat the floor on the primary quality metric."}
    overall = r["scenarios"]["balanced"]["overall"]
    gain = (r["confidence"].get("vs_floor") or {}).get(win) or {}
    if overall[FLOOR] > overall[win] and not gain.get("distinguishable"):
        return {"decision": "keep_floor", "model": FLOOR,
                "line": f"{keep}: {_name(win)} leads the candidates, but its quality gain over the floor is not "
                        f"statistically distinguishable and the floor scores higher overall ({overall[FLOOR]:.3f} vs "
                        f"{overall[win]:.3f})."}
    return {"decision": "select", "model": win,
            "line": f"Select {_name(win)} as the primary embedding model (Balanced winner, Overall Score "
                    f"{overall[win]:.3f}; record it in model_versions with this experiment_id)."}


# ---------------------------------------------------------------- report

def _fmt(value, digits=3) -> str:
    return NOT_YET_EVALUATED if value is None else f"{value:.{digits}f}"


def _main_table(r: dict, k: int) -> list[str]:
    head = "| Model | MRR | NDCG@K | Recall@K | Precision@K | Latency | Memory | Size | Dimension | Complexity | Scalability | Overall Score |"
    lines = [head, "|" + "---|" * 12]
    overall = r["scenarios"]["balanced"]["overall"]
    q = r["quality"].get(r["config"]["evaluation"]["primary_protocol"], {})
    for key in RUNNABLE:
        c, cost = r["criteria"][key], r["costs"]["models"][key]
        quality_cells = []
        for metric in ("mrr", f"ndcg@{k}", f"recall@{k}", f"precision@{k}"):
            quality_cells.append(_fmt(q[key]["mean"][metric]) if key in q else NOT_YET_EVALUATED)
        dim = (f"{cost['dimension']} sparse (~{cost['mean_nonzeros_per_job_vector']:.0f} non-zero)"
               if MODEL_CONFIGS[key]["kind"] == "tfidf" else str(cost["dimension"]))
        if overall is None:
            score = NOT_YET_EVALUATED
        else:
            score = f"{overall[key]:.3f}"
            if key == FLOOR:
                score += " (floor)"
            elif key not in r["eligible"]:
                score += " (below floor)"
        lines.append(f"| {_name(key)} | " + " | ".join(quality_cells) +
                     f" | {c['latency']:.1f} ms | {c['memory']:.0f} MB | {cost['size_mb']:.1f} MB | {dim} | "
                     f"{c['complexity'] - 1} pt | {c['scalability']:.1f} min/100k jobs (extrapolated) | {score} |")
    custom, hybrid = NOT_RUN_MODELS["custom"], NOT_RUN_MODELS["hybrid"]
    if "e5_domain" not in RUNNABLE:  # slot E ran when a domain-adapted model exists (src/embeddings/domain_adapt.py)
        lines.append(f"| E. {custom['display']} | {custom['status']} |" + " — |" * 10)
    lines.append(f"| F. {hybrid['display']} | {hybrid['status']} |" + " — |" * 10)
    return lines


def render_markdown(r: dict) -> str:
    ev = r["config"]["evaluation"]
    k = ev["primary_k"]
    labels = r["labels"]
    stamp = labels.get("stamp")
    out = [f"# Embedding model evaluation: {r['experiment_id']}", ""]
    if stamp:
        out += [f"> **{stamp}**", ""]
    out += [f"Generated by `src/embeddings/model_evaluation_engine.py` on {r['finished_at']} "
            f"({r['runtime_seconds']} s). Weights: `{r['config']['weights_file']}` ({r['config']['weights_version']}, "
            f"sha256 {r['config']['weights_sha256'][:12]}). Dataset {r['dataset_version']}.", ""]

    out += ["## Labels", ""]
    if not labels["provided"]:
        out += ["No relevance labels were provided, so every quality metric reads **Not yet evaluated** and no "
                "Overall Score is computed. The cost side below is measured.",
                "Labels arrive from the labelling page (`/labelling`, human_reviewed); rerun with "
                "`--labels data/benchmark/human_labels.csv`.", ""]
    else:
        out += [f"- File: `{labels['path']}` (sha256 {labels['sha256'][:12]}): {labels['pairs']:,} pairs, "
                f"{labels['students']} students",
                f"- Sources: {labels['source_breakdown']}. Evaluated: **{labels['used_source']}** only.",
                f"- Splits: {labels['split_breakdown']}. Evaluated: **{labels['used_split']}** only "
                f"({labels['used_pairs']:,} pairs, {labels['used_students']} students). Train rows evaluated: 0.",
                f"- Grades: {labels['label_distribution']} (relevant = grade >= {ev['relevance_threshold']})",
                f"- **{labels['validity']['line']}**"]
        for reason in labels["validity"]["reasons"]:
            out.append(f"  - {reason}")
        out += [f"- Circularity check: provenance `{labels['validity']['generated_by']}`; mean per-student Kendall tau "
                f"between each model's scores and the labels: {labels['validity']['mean_per_student_kendall_tau']} "
                f"(>= 0.9 would mean the labels look model-generated)", ""]

    out += [f"## Model comparison (K = {k}, {ev['primary_protocol']} protocol, Balanced weights)", ""]
    if stamp:
        out += [f"**{stamp}**", ""]
    out += _main_table(r, k)
    out += ["", "Latency: CPU, one student-job pair (embed both + cosine), median of 30 warm trials. Memory: peak "
            "resident memory added by the model in an isolated process. Size: fp32 parameters (TF-IDF: vocabulary + "
            "IDF). Complexity: points from the engineering table. Scalability: CPU minutes to embed 100,000 job "
            "texts, extrapolated from measured throughput. TF-IDF is the floor every embedding model must beat; it is "
            "not a selection candidate.", ""]

    q = r["quality"].get(ev["primary_protocol"], {})
    if q:
        other_k = [kk for kk in ev["ks"] if kk != k]
        for kk in other_k:
            out += [f"### Quality at K = {kk}", "", "| Model | MRR | NDCG@K | Recall@K | Precision@K | Hit Rate@K |", "|---|---|---|---|---|---|"]
            for key in RUNNABLE:
                m = q[key]["mean"]
                out.append(f"| {_name(key)} | {m['mrr']:.3f} | {m[f'ndcg@{kk}']:.3f} | {m[f'recall@{kk}']:.3f} | "
                           f"{m[f'precision@{kk}']:.3f} | {m[f'hit_rate@{kk}']:.3f} |")
            degenerate = [kk2 for kk2 in q[FLOOR]["degenerate_k"]]
            if degenerate:
                out.append(f"\nAt K in {degenerate}, K >= the number of judged jobs for some students, so Precision@K "
                           "and Recall@K can't separate models there; NDCG and MRR still can.")
            out.append("")
        out += ["### Quality details", "", f"95% confidence intervals: paired bootstrap over students "
                f"({ev['bootstrap_resamples']} resamples).", "",
                f"| Model | MRR (95% CI) | NDCG@{k} (95% CI) | Hit Rate@{k} | Kendall tau | Students evaluated | Excluded (no relevant job) | vs TF-IDF floor, NDCG@{k} |",
                "|---|---|---|---|---|---|---|---|"]
        for key in RUNNABLE:
            m, ci = q[key]["mean"], r["confidence"][key]
            vs = r["confidence"]["vs_floor"].get(key) if key != FLOOR else None
            vs_text = "—" if vs is None else (f"{vs['mean_difference']:+.3f} ({vs['ci95'][0]:+.3f} to {vs['ci95'][1]:+.3f})"
                                             + ("" if vs["distinguishable"] else ", not distinguishable"))
            out.append(f"| {_name(key)} | {m['mrr']:.3f} ({ci['mrr'][0]:.3f}-{ci['mrr'][1]:.3f}) | {m[f'ndcg@{k}']:.3f} "
                       f"({ci[f'ndcg@{k}'][0]:.3f}-{ci[f'ndcg@{k}'][1]:.3f}) | {m[f'hit_rate@{k}']:.3f} | "
                       f"{_fmt(m['kendall_tau'])} | {q[key]['students_evaluated']} | {q[key]['students_excluded_no_relevant']} | {vs_text} |")
        if "section_weighted" in r["quality"]:
            sw = r["quality"]["section_weighted"]
            order_sw = sorted(RUNNABLE, key=lambda m: -sw[m]["mean"][f"ndcg@{k}"])
            order_ft = sorted(RUNNABLE, key=lambda m: -q[m]["mean"][f"ndcg@{k}"])
            out += ["", f"Robustness check: the section-weighted protocol (skills, experience and projects scored "
                    f"separately, weights renormalised over the sections present: audit I-3/I-4 fixed) gives NDCG@{k} "
                    + ", ".join(f"{m} {sw[m]['mean'][f'ndcg@{k}']:.3f}" for m in RUNNABLE)
                    + f". Quality order: {' > '.join(order_sw)} (full text: {' > '.join(order_ft)}); "
                    + ("the same order." if order_sw == order_ft else "the order differs, so the scoring protocol matters.")]
        out.append("")

    out += ["## Cost (measured)", "", "| Model | Version | Single pair, CPU: median / p95 / sd (ms) | Batch of 64, CPU (ms) | Throughput CPU (texts/s) | Throughput MPS (texts/s) | Memory after load / peak (MB) | Load (s) |", "|---|---|---|---|---|---|---|---|"]
    for key in RUNNABLE:
        c = r["costs"]["models"][key]
        sp, b = c["cpu"]["single_pair_latency_ms"], c["cpu"]["batch_latency_ms"]
        mps = c["mps"]["throughput_texts_per_s"] if isinstance(c["mps"], dict) else c["mps"]
        out.append(f"| {_name(key)} | `{c['version']}` | {sp['median']} / {sp['p95']} / {sp['stdev']} | {b['median']} | "
                   f"{c['cpu']['throughput_texts_per_s']} | {mps} | {c['memory_mb']['after_load']} / {c['memory_mb']['peak']} | {c['load_seconds']} |")
    hw = r["costs"]["hardware"]
    out += ["", f"Hardware: {hw.get('machine')} ({hw.get('system')}), {hw.get('cpu_cores')} logical cores, "
            f"{hw.get('ram_gb')} GB RAM, torch {hw.get('torch', 'n/a')}, MPS available: {hw.get('mps_available')}. "
            f"Measured {r['costs']['measured_at']} with other services (Docker) running: absolute numbers are "
            "specific to this laptop; the relative order is what transfers. Each model ran in its own process with "
            "network access disabled.", ""]

    out += ["## Engineering", "", "| Model | Local feasibility | Dependencies (installed) | Preprocessing | Complexity | Truncated texts (students / jobs) | Storage per 100k vectors | Reproducibility |", "|---|---|---|---|---|---|---|---|"]
    for key in RUNNABLE:
        e = r["engineering"][key]
        t, s, rep = e["truncation"], e["scalability_extrapolated"], e["reproducibility"]
        limit = f" (limit {t['max_seq_length']} tokens)" if t["max_seq_length"] else " (no length limit)"
        out.append(f"| {_name(key)} | {e['local_feasibility']} | {e['dependencies']['distributions']} packages, "
                   f"{e['dependencies']['installed_mb']:.0f} MB | {e['preprocessing']} | {e['complexity']['points']}: "
                   f"{'; '.join(e['complexity']['reasons'])} | {t['students_truncated_pct']}% / {t['jobs_truncated_pct']}%{limit} | "
                   f"{s['storage_mb_per_100k_vectors']} MB (extrapolated) | pinned `{str(rep['revision'])[:12]}`; "
                   f"CPU runs identical: {rep['deterministic_on_cpu']}; CPU vs MPS max diff "
                   f"{'n/a (CPU-only)' if rep['cpu_vs_mps_max_abs_diff'] is None else format(rep['cpu_vs_mps_max_abs_diff'], '.1e')} |")
    out.append("")

    out += ["## Raw cosine similarity is not comparable across model families", "",
            "Each model has its own similarity range, so a raw cosine of 0.80 means different things for E5 and SBERT. "
            "The engine never uses mean similarity as quality: the Overall Score uses label-based quality only. "
            "Range of full-text cosine scores over every student x distinct job:", "",
            "| Model | Mean | SD | 5th percentile | 95th percentile |", "|---|---|---|---|---|"]
    for key in RUNNABLE:
        g = r["cosine_ranges"][key]
        out.append(f"| {_name(key)} | {g['mean']} | {g['sd']} | {g['p5']} | {g['p95']} |")
    out.append("")

    out += ["## Multi-criteria decision and sensitivity analysis", "",
            f"Normalisation: {r['config']['normalization']['method']} across the five measured models, per criterion "
            f"(1 = best); cost criteria on log10 ({', '.join(r['config']['normalization']['log_scale'])}). "
            f"Overall Score = sum of weight x normalised criterion. Gate: an embedding model must beat the TF-IDF floor "
            f"on NDCG@{k} to be eligible.", ""]
    for name, sc in r["scenarios"].items():
        w = sc["weights"]
        quality_share = sum(w[c] for c in QUALITY)
        out += [f"### {name.replace('_', ' ').title()}: {sc['description']}",
                f"Weights: quality {quality_share:.2f} (" + ", ".join(f"{c} {w[c]}" for c in QUALITY) + "), cost "
                f"{1 - quality_share:.2f} (" + ", ".join(f"{c} {w[c]}" for c in COST) + ").", ""]
        if sc["overall"] is None:
            out += [f"Re-ranked table: {NOT_YET_EVALUATED} (needs relevance labels).", ""]
            continue
        out += ["| Rank | Model | Overall Score | Eligible |", "|---|---|---|---|"]
        for rank, key in enumerate(sc["ranking"], start=1):
            status = "floor" if key == FLOOR else ("yes" if key in r["eligible"] else "no (below floor)")
            out.append(f"| {rank} | {_name(key)} | {sc['overall'][key]:.3f} | {status} |")
        out += ["", f"Winner: **{_name(sc['winner']) if sc['winner'] else 'none (no eligible model)'}**", ""]
    if r["winner_changes"] is None:
        out += [f"Does the winner change across scenarios? {NOT_YET_EVALUATED}.", ""]
    else:
        out += [f"Does the winner change across scenarios? **{'Yes' if r['winner_changes'] else 'No'}**.", ""]
        vs = r["confidence"].get("vs_floor", {})
        shaky = [m for m in CANDIDATES if vs.get(m) and not vs[m]["distinguishable"]]
        if shaky:
            out += [f"How firm is the gate? It compares point estimates. For {', '.join(shaky)} the NDCG@{k} "
                    "difference from the TF-IDF floor is not statistically distinguishable (see Quality details), so "
                    "the eligible set itself could change with more labels.", ""]
        pc = r.get("protocol_sensitivity")
        if pc:
            same = pc["winner"] == r["scenarios"]["balanced"]["winner"]
            out += [f"Does the winner change with the scoring protocol? Under {pc['protocol']} (Balanced weights) the "
                    f"winner is **{_name(pc['winner']) if pc['winner'] else 'none'}** (eligible: "
                    f"{', '.join(pc['eligible']) or 'none'}): **{'No' if same else 'Yes'}**.", ""]
    if r["random_weight_stability"]:
        sens = r["config"]["sensitivity"]
        out += [f"Random weightings ({sens['random_weightings']} draws from a Dirichlet distribution centred on the "
                f"Balanced weights, concentration {sens['concentration']}, seed {sens['seed']}): share of draws each "
                "eligible model wins: " + ", ".join(f"{_name(m)} {p}%" for m, p in r["random_weight_stability"].items()) + ".", ""]

    out += ["## Best models", ""]
    for key, label in (("quality", "BEST MODEL FOR QUALITY"), ("speed", "BEST MODEL FOR SPEED"),
                       ("memory", "BEST MODEL FOR MEMORY"), ("local_deployment", "BEST MODEL FOR LOCAL DEPLOYMENT"),
                       ("overall", "BEST OVERALL MODEL (Balanced)")):
        out.append(f"- **{label}:** {r['best_models'][key]}")
    out += ["", f"**Decision (rule fixed in ADR 0012): {r['recommendation']['line']}**", ""]

    out += ["## Not run", ""] + ([] if "e5_domain" in RUNNABLE else [
            f"- **E. {NOT_RUN_MODELS['custom']['display']}: {NOT_RUN_MODELS['custom']['status']}.** {NOT_RUN_MODELS['custom']['reason']}"]) + [
            f"- **F. {NOT_RUN_MODELS['hybrid']['display']}:** {NOT_RUN_MODELS['hybrid']['reason']} This is a separate "
            "architecture, not a gap in this comparison.", ""]
    out += ["## Limitations", "",
            "- The job corpus is synthetic and templated: 376 distinct role texts. Results describe this data, not real job boards.",
            "- Quality is measured on the validation split only; test is held back for Phase 14, so these numbers are "
            "the basis for selection, not a final unbiased estimate."]
    source = labels.get("used_source")
    if source == "human_reviewed":
        out.append("- Human labels are pooled (TREC-style): the five models chose which jobs were judged. Only judged "
                   "jobs are ranked (condensed lists), which is robust to pool depth, but a model outside the pool "
                   "(e.g. Phase 7's hybrid) should be checked for unjudged jobs at the top of its ranking.")
    if source == "weakly_labelled":
        out.append("- Weak labels only encode \"job in the student's stated field\" from hand-written rules "
                   "(config/weak_label_rules.yaml); they ignore the student's actual skills, and they can't separate a "
                   "strong match from a moderate one. They test the framework and give a first signal, not a selection.")
    out.append("- Costs are from one developer laptop with other services running.")
    if stamp:
        out += ["", f"> **{stamp}**"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- outputs

def write_outputs(report: dict) -> dict[str, str]:
    folder = os.path.join(RESULTS, "embedding_evaluation")
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, report["experiment_id"])
    data = {k: v for k, v in report.items() if k != "markdown"}
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write(report["markdown"])
    source = report["labels"].get("used_source")
    latest = os.path.join(RESULTS, "embedding_model_evaluation" + (f"_{source}" if source else ""))
    shutil.copyfile(base + ".json", latest + ".json")
    shutil.copyfile(base + ".md", latest + ".md")
    return {"json": base + ".json", "markdown": base + ".md", "latest": latest + ".md"}


def record_experiment(dsn: str, report: dict) -> None:
    """Store the run in experiment_runs / experiment_metrics (spec §25)."""
    import psycopg
    from psycopg.types.json import Jsonb

    ev = report["config"]["evaluation"]
    labels = report["labels"]
    split_info = {s: {"students": report["splits"]["students"][s],
                      "label_pairs": labels.get("split_breakdown", {}).get(s, {}).get("pairs", 0),
                      "evaluated": s == ev["selection_split"] and labels["provided"]} for s in SPLITS}
    q = report["quality"].get(ev["primary_protocol"], {})
    rows = []
    for key in RUNNABLE:
        version = report["models"][key]["version"]
        for kk in ev["ks"]:
            for metric in ("mrr", f"ndcg@{kk}", f"recall@{kk}", f"precision@{kk}", f"hit_rate@{kk}"):
                if metric == "mrr" and kk != ev["ks"][0]:
                    continue
                value = q[key]["mean"].get(metric) if key in q else None
                rows.append((key, version, metric, ev["selection_split"], value,
                             "measured" if value is not None else "not_yet_evaluated"))
        c = report["criteria"][key]
        for metric in ("latency", "memory", "size", "dimension"):
            rows.append((key, version, f"{metric}", "n/a", float(c[metric]), "measured"))
        rows.append((key, version, "scalability", "n/a", float(c["scalability"]), "extrapolated"))
        overall = report["scenarios"]["balanced"]["overall"]
        rows.append((key, version, "overall_score_balanced", "n/a", overall[key] if overall else None,
                     "measured" if overall else "not_yet_evaluated"))
    rows.append(("custom", "none", "overall_score_balanced", "n/a", None, "not_feasible"))
    with psycopg.connect(dsn, connect_timeout=5) as conn, conn.transaction():
        conn.execute(
            "INSERT INTO experiment_runs (experiment_id, kind, started_at, finished_at, dataset_version, dataset_size, "
            "train_split, validation_split, test_split, random_seed, feature_version, embedding_model, embedding_version, "
            "prediction_model, hyperparameters, metrics, runtime_seconds, memory, hardware, labels, notes, limitations, report) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (report["experiment_id"], report["kind"], report["started_at"], report["finished_at"],
             report["dataset_version"], Jsonb(report["dataset_size"]), Jsonb(split_info["train"]),
             Jsonb(split_info["validation"]), Jsonb(split_info["test"]), ev["seed"], report["config"]["text_view"],
             "comparison: " + ", ".join(RUNNABLE), "; ".join(f"{k}={v['version']}" for k, v in report["models"].items()),
             "none (cosine-similarity ranking)", Jsonb(report["config"]),
             Jsonb({"quality": report["quality"], "criteria": report["criteria"]}), report["runtime_seconds"],
             Jsonb({k: report["costs"]["models"][k]["memory_mb"] for k in RUNNABLE}), Jsonb(report["costs"]["hardware"]),
             Jsonb({k: v for k, v in labels.items() if k != "manifest"} | {"validity_line": report["validity"]}),
             "Phase 6 embedding selection run. TF-IDF is the floor; custom model not feasible; hybrid is Phase 7/14.",
             "Synthetic templated job corpus; validation split only; laptop costs; see the report's Limitations.",
             Jsonb(json.loads(json.dumps({k: v for k, v in report.items() if k != "markdown"}, default=str)))))
        conn.cursor().executemany(
            "INSERT INTO experiment_metrics (experiment_id, model_name, model_version, metric, split, value, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)", [(report["experiment_id"], *r) for r in rows])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 6 Model Evaluation Engine")
    ap.add_argument("--labels", default=None, help="labels CSV (student_id,job_id,relevance_label,label_source,split)")
    ap.add_argument("--label-source", default=None, help="evaluate this label_source (default: the best one present)")
    ap.add_argument("--device", default=None, help="mps or cpu for scoring (default: mps when available)")
    ap.add_argument("--reuse-cost", action="store_true")
    ap.add_argument("--record-db", action="store_true")
    ap.add_argument("--record-existing", default=None, metavar="REPORT_JSON",
                    help="store an already written report in Postgres (EXPERIMENTS_DATABASE_URL) and exit")
    args = ap.parse_args()
    if args.record_existing:
        dsn = os.environ.get("EXPERIMENTS_DATABASE_URL")
        if not dsn:
            raise SystemExit("--record-existing needs EXPERIMENTS_DATABASE_URL")
        with open(args.record_existing, encoding="utf-8") as fh:
            existing = json.load(fh)
        record_experiment(dsn, existing)
        raise SystemExit(f"recorded {existing['experiment_id']} in Postgres")
    device = args.device
    if device is None:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if not args.labels:
        print("No labels path given: quality metrics will read 'Not yet evaluated' (cost is still measured).")
    report = run(args.labels, device, args.reuse_cost, args.label_source)
    paths = write_outputs(report)
    if args.record_db:
        dsn = os.environ.get("EXPERIMENTS_DATABASE_URL")
        if not dsn:
            raise SystemExit("--record-db needs EXPERIMENTS_DATABASE_URL")
        record_experiment(dsn, report)
        print(f"recorded {report['experiment_id']} in Postgres")
    print(report["markdown"])
    print(f"written: {paths}")
