"""Phase 14 Part A (spec §34): the five baselines, one benchmark, one metric suite (docs/adr/0021).

    python -m src.evaluation.baseline_comparison            (in the backend image: E5 weights and the trained model)

Every baseline scores the SAME human-labelled pairs of the SAME split (config/baseline_comparison.yaml), and every
score goes through the same metric code (src/evaluation/retrieval_metrics.py): each student's judged jobs are ranked
by the score, ties broken by job_id. Nothing is re-implemented that an earlier phase already measured:

  B1 keyword     the Profile/Job agents' own skill extraction + normalize_skills, overlap of normalised names (new:
                 no earlier phase had a keyword baseline)
  B2 TF-IDF      Phase 6's scorer (score_matrix) with the TF-IDF floor
  B3 structured  Phase 7's weighted_baseline_v1 on the Phase 8 feature matrix (as the Phase 8 engine scores it)
  B4 semantic    Phase 6's scorer with the selected embedding model (E5), alone
  B5 hybrid      the production Stage 4 model from config/matching_model.yaml on the same feature matrix

The test split was held back by Phases 6 and 8 for this comparison, so it answers the research questions. The
validation split is reported only to show that this module reproduces the Phase 6 and Phase 8 numbers.

Outputs: results/baseline_comparison/<experiment_id>.{md,json} and results/baseline_comparison.{md,json} (latest).
"""

import argparse
import hashlib
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.relevance_labels import circularity_check, load_labels
from src.evaluation.retrieval_metrics import bootstrap_ci, evaluate, paired_difference

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "baseline_comparison.yaml"
STUDENTS_CSV = ROOT / "data" / "processed" / "students_unified.csv"
JOBS_CSV = ROOT / "data" / "processed" / "jobs.csv"
FEATURES_DIR = ROOT / "data" / "benchmark" / "matching_features"
MODELS_DIR = ROOT / "models" / "matching"
RESULTS = ROOT / "results"
ORDER = ("b1_keyword", "b2_tfidf", "b3_structured", "b4_semantic", "b5_hybrid")
METRICS = ("precision@5", "recall@5", "precision@10", "recall@10", "mrr", "ndcg@5", "ndcg@10")
# Phase 6/8 reports whose validation numbers this module must reproduce (reproduction check).
PHASE6_REPORT = RESULTS / "embedding_model_evaluation_human_reviewed.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path = CONFIG) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if set(cfg["baselines"]) != set(ORDER):
        raise ValueError(f"{path}: baselines must be exactly {list(ORDER)}")
    if cfg["answer_split"] == "train" or "train" in cfg["splits"]:
        raise ValueError("train rows are never evaluated (the hybrid's model was trained on them)")
    cfg["sha256"] = _sha(path)
    return cfg


# ---------------------------------------------------------------- B1: normalised keyword matching

def _contains(a: str, b: str) -> bool:
    """a occurs in b as whole words (keys of one letter only match exactly: "c" must not hit "c#" or "r&d")."""
    return len(a) > 1 and re.search(rf"(?<!\w){re.escape(a)}(?!\w)", b) is not None


def keyword_skills(students: pd.DataFrame, jobs: pd.DataFrame) -> tuple[dict, dict]:
    """Normalised skill keys per student and per job, through the agents' own extraction + normalisation code."""
    from app.jobs.skills_split import split_skill_phrases
    from app.profile.sections import record_skills_text
    from app.profile.skills import normalize_skills
    from src.parsing.resume_text_parser import extract_skills
    from src.parsing.section_parser import _clean

    def keys(raw: list[str]) -> frozenset[str]:
        return frozenset(s.canonical.lower().strip() for s in normalize_skills(raw) if s.canonical.strip())

    s_keys = {r["student_id"]: keys(extract_skills(_clean(record_skills_text(r))))
              for r in students.to_dict("records")}
    j_keys = {r["job_id"]: keys(split_skill_phrases(r["skills"] or "")) for r in jobs.to_dict("records")}
    return s_keys, j_keys


def keyword_score(student: frozenset[str], job: frozenset[str]) -> float | None:
    """Share of the job's normalised skills the student names (exact, or whole-word containment either way).
    None when the job lists no skill: the pair can't be scored and ranks last."""
    if not job:
        return None
    matched = sum(1 for j in job if j in student or any(_contains(j, s) or _contains(s, j) for s in student))
    return matched / len(job)


def score_keyword(labels: pd.DataFrame, students: pd.DataFrame, jobs: pd.DataFrame) -> tuple[dict, dict]:
    s_keys, j_keys = keyword_skills(students, jobs)
    scores = {}
    for sid, jid in zip(labels["student_id"], labels["job_id"]):
        value = keyword_score(s_keys[sid], j_keys[jid])
        if value is not None:
            scores[(sid, jid)] = value
    coverage = {"students_with_skills": sum(1 for s in labels["student_id"].unique() if s_keys[s]),
                "students": int(labels["student_id"].nunique()),
                "jobs_with_skills": sum(1 for j in labels["job_id"].unique() if j_keys[j]),
                "jobs": int(labels["job_id"].nunique()),
                "pairs_with_any_match": sum(1 for v in scores.values() if v > 0)}
    return scores, coverage


# ---------------------------------------------------------------- B2 / B4: the Phase 6 scorer

def score_embedding(key: str, labels: pd.DataFrame, students: pd.DataFrame, jobs: pd.DataFrame) -> tuple[dict, str]:
    """Phase 6's own score_matrix (full_text protocol) for the labelled students x labelled jobs. Each cosine depends
    only on its pair (TF-IDF is fitted on the whole processed corpus inside the encoder), so restricting the matrix to
    the labelled entities gives exactly the scores the Phase 6 engine computed."""
    from src.embeddings.embedding_model_benchmark import as_pair_dict, load_encoder, score_matrix

    encoder = load_encoder(key, "cpu")
    sub_s = students[students["student_id"].isin(set(labels["student_id"]))]
    sub_j = jobs[jobs["job_id"].isin(set(labels["job_id"]))]
    sids, jids, matrix = score_matrix(encoder, sub_s, sub_j, "full_text")
    pairs = set(zip(labels["student_id"], labels["job_id"]))
    return {p: v for p, v in as_pair_dict(sids, jids, matrix).items() if p in pairs}, encoder.version


# ---------------------------------------------------------------- B3 / B5: the Phase 8 feature matrix

def production_model() -> dict:
    """The Stage 4 model the backend runs (config/matching_model.yaml), and the feature matrix it was trained on."""
    from app.matching.predictor import active_choice

    choice = active_choice()
    manifest = json.loads((MODELS_DIR / choice.artifacts_experiment / "manifest.json").read_text(encoding="utf-8"))
    features_path = FEATURES_DIR / manifest["feature_matrix"]["file"]
    if _sha(features_path) != manifest["feature_matrix"]["sha256"]:
        raise ValueError(f"{features_path.name} changed since {choice.artifacts_experiment} was trained")
    return {"choice": choice, "manifest": manifest, "features_path": features_path}


def score_feature_models(labels: pd.DataFrame, prod: dict) -> tuple[dict, dict, dict]:
    from app.matching.baseline import predict
    from app.matching.config import get_matching_config
    from app.matching.features import FEATURE_NAMES
    from src.scoring.matching_models import feature_matrix, load, read_features

    features = read_features(prod["features_path"])
    wanted = set(zip(labels["student_id"], labels["job_id"]))
    rows = features[[p in wanted for p in zip(features["student_id"], features["job_id"])]].reset_index(drop=True)
    if len(rows) != len(wanted):
        raise ValueError(f"{len(wanted) - len(rows)} labelled pairs have no features: rebuild the feature matrix")
    if (rows["split"] == "train").any():
        raise ValueError("a labelled pair sits in the train split")
    X, ids = feature_matrix(rows), list(zip(rows["student_id"], rows["job_id"]))

    mcfg = get_matching_config()
    b3 = {}
    for pair, vec in zip(ids, X.itertuples(index=False)):
        p = predict({f: (None if pd.isna(v) else float(v)) for f, v in zip(FEATURE_NAMES, vec)}, mcfg)
        if p["suitability"] is not None:
            b3[pair] = p["suitability"]

    choice, manifest = prod["choice"], prod["manifest"]
    art = manifest["models"][choice.active]
    model = load(MODELS_DIR / choice.artifacts_experiment / art["file"], choice.active, art["format"], art["sha256"])
    b5 = dict(zip(ids, model.score(X).round(8).tolist()))
    versions = {"b3_structured": mcfg.matching_model_version,
                "b5_hybrid": f"{choice.active}+{art['sha256'][:12]} (artifacts {choice.artifacts_experiment}, "
                             f"selected by {choice.selection_experiment})"}
    info = {"feature_matrix": prod["features_path"].name, "feature_version": manifest["feature_version"],
            "embedding_version": manifest["embedding_version"], "pairs_scored": len(rows)}
    return {"b3_structured": b3, "b5_hybrid": b5}, versions, info


# ---------------------------------------------------------------- the run

def evaluate_split(scores: dict, labels: pd.DataFrame, ev: dict) -> dict:
    out = {}
    for key in ORDER:
        r = evaluate(scores[key], labels, tuple(ev["ks"]), ev["relevance_threshold"])
        ci = {m: bootstrap_ci([v[m] for v in r["per_student"].values()], ev["bootstrap_resamples"], ev["seed"])
              for m in METRICS}
        out[key] = {"mean": r["mean"], "ci95": ci, "per_student": r["per_student"],
                    "students_evaluated": r["students_evaluated"],
                    "students_excluded_no_relevant": r["students_excluded_no_relevant"],
                    "pairs": r["pairs"], "unscored_pairs": r["unscored_pairs"], "degenerate_k": r["degenerate_k"]}
    return out


def compare(quality: dict, a: str, b: str, metric: str, ev: dict) -> dict:
    per = lambda k: {s: v[metric] for s, v in quality[k]["per_student"].items()}  # noqa: E731
    d = paired_difference(per(a), per(b), ev["bootstrap_resamples"], ev["seed"]) or {}
    if not d:
        verdict = "not comparable (too few students)"
    elif d["distinguishable"]:
        verdict = f"{a} outperforms {b}" if d["mean_difference"] > 0 else f"{b} outperforms {a}"
    elif d["mean_difference"] > 0:
        verdict = f"{a} higher, not statistically distinguishable"
    elif d["mean_difference"] < 0:
        verdict = f"{a} lower, not statistically distinguishable"
    else:
        verdict = "identical"
    return {"a": a, "b": b, "metric": metric, **d, "verdict": verdict}


def reproduction_check(quality_val: dict, versions: dict) -> list[dict]:
    """Validation numbers vs the stored Phase 6 / Phase 8 human-label reports. A mismatch means this module does not
    score a baseline the way the phase that selected it did, so its test numbers could not be compared."""
    rows = []
    if PHASE6_REPORT.exists():
        p6 = json.loads(PHASE6_REPORT.read_text(encoding="utf-8"))
        q6 = p6["quality"]["full_text"]
        for key, model in (("b2_tfidf", "tfidf"), ("b4_semantic", "e5")):
            if model in q6:
                same_version = p6["models"][model]["version"] == versions[key]
                for m in ("ndcg@5", "mrr"):
                    rows.append({"baseline": key, "metric": m, "source": p6["experiment_id"],
                                 "reported": q6[model]["mean"][m], "here": quality_val[key]["mean"][m],
                                 "same_model_version": same_version})
    sel = versions.get("_selection_report")
    if sel and Path(sel).exists():
        p8 = json.loads(Path(sel).read_text(encoding="utf-8"))
        for key, model in (("b3_structured", "weighted_baseline_v1"), ("b5_hybrid", versions["_active"])):
            reported = p8["quality"].get(model, {}).get("ranking", {}).get("mean", {})
            for m in ("ndcg@10", "mrr"):
                if m in reported:
                    rows.append({"baseline": key, "metric": m, "source": p8["experiment_id"], "reported": reported[m],
                                 "here": quality_val[key]["mean"][m], "same_model_version": True})
    for r in rows:
        r["match"] = bool(np.isclose(r["reported"], r["here"], atol=5e-4))
    return rows


def run(config_path: Path = CONFIG) -> dict:
    t0, started = time.time(), datetime.now(timezone.utc)
    cfg = load_config(config_path)
    ev = cfg["evaluation"]
    labelset = load_labels(str(ROOT / cfg["labels"]), str(STUDENTS_CSV), str(JOBS_CSV))
    source = cfg["label_source"]
    all_eval = labelset.select(source, cfg["splits"])
    students = pd.read_csv(STUDENTS_CSV, dtype=str, keep_default_na=False)
    jobs = pd.read_csv(JOBS_CSV, dtype=str, keep_default_na=False)

    print("  B1 keyword", flush=True)
    scores, versions, timings = {}, {}, {}
    t = time.time()
    scores["b1_keyword"], keyword_coverage = score_keyword(all_eval, students, jobs)
    versions["b1_keyword"] = "normalize_skills + " + __import__("app.profile.skills", fromlist=["x"]).alias_table_version()
    timings["b1_keyword"] = time.time() - t
    for key, model in (("b2_tfidf", "tfidf"), ("b4_semantic", "e5")):
        print(f"  {key} ({model})", flush=True)
        t = time.time()
        scores[key], versions[key] = score_embedding(model, all_eval, students, jobs)
        timings[key] = time.time() - t
    print("  B3 structured + B5 hybrid (feature matrix)", flush=True)
    t = time.time()
    prod = production_model()
    feature_scores, feature_versions, feature_info = score_feature_models(all_eval, prod)
    scores.update(feature_scores)
    versions.update(feature_versions)
    timings["b3_b5_feature_models"] = time.time() - t

    quality, comparisons, validity = {}, {}, {}
    for split in cfg["splits"]:
        labels = labelset.select(source, [split])
        quality[split] = evaluate_split(scores, labels, ev)
        comparisons[split] = {rq: [compare(quality[split], a, b, ev["primary_metric"], ev)
                                   | {"secondary": {m: compare(quality[split], a, b, m, ev)
                                                    for m in ev.get("secondary_metrics", [])}}
                                   for a, b in pairs]
                              for rq, pairs in cfg["comparisons"].items()}
        pairs = set(zip(labels["student_id"], labels["job_id"]))
        validity[split] = circularity_check(labelset, source, list(ORDER),
                                            {k: {p: v for p, v in scores[k].items() if p in pairs} for k in ORDER})

    selection_report = RESULTS / "matching_model_evaluation" / f"{prod['choice'].selection_experiment}.json"
    repro = reproduction_check(quality.get("validation", {}),
                               {**versions, "_selection_report": str(selection_report), "_active": prod["choice"].active}
                               ) if "validation" in quality else []

    finished = datetime.now(timezone.utc)
    report = {
        "experiment_id": f"base-{started:%Y%m%dT%H%M%SZ}-human", "kind": "baseline_comparison",
        "started_at": started.isoformat(timespec="seconds"), "finished_at": finished.isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 1),
        "dataset_version": f"students:{_sha(STUDENTS_CSV)[:12]} jobs:{_sha(JOBS_CSV)[:12]}",
        "config": {"file": str(config_path.relative_to(ROOT)), "version": cfg["version"], "sha256": cfg["sha256"],
                   "evaluation": ev, "answer_split": cfg["answer_split"], "splits": cfg["splits"]},
        "labels": {"path": cfg["labels"], "sha256": labelset.sha256, "source": source,
                   "split_breakdown": labelset.split_breakdown(),
                   "grades_per_split": {s: {int(g): int((labelset.select(source, [s])["relevance_label"] == g).sum())
                                            for g in (0, 1, 2, 3)} for s in cfg["splits"]},
                   "pool_models": (labelset.manifest or {}).get("pool", {}), "labellers": (labelset.manifest or {}).get("labellers")},
        "baselines": {k: {"display": cfg["baselines"][k]["display"], "method": cfg["baselines"][k]["method"],
                          "version": versions[k]} for k in ORDER},
        "keyword_coverage": keyword_coverage, "feature_matrix": feature_info,
        "scoring_seconds": {k: round(v, 1) for k, v in timings.items()},
        "quality": {s: {k: {kk: vv for kk, vv in r.items() if kk != "per_student"} for k, r in q.items()}
                    for s, q in quality.items()},
        "per_student": {s: {k: r["per_student"] for k, r in q.items()} for s, q in quality.items()},
        "comparisons": comparisons, "validity": validity, "reproduction": repro,
    }
    report["markdown"] = render_markdown(report)
    return report


# ---------------------------------------------------------------- report

def _ci(r: dict, m: str) -> str:
    ci = r["ci95"].get(m)
    return f"{r['mean'][m]:.3f} ({ci[0]:.3f}-{ci[1]:.3f})" if ci else f"{r['mean'][m]:.3f}"


def _table(report: dict, split: str) -> list[str]:
    q = report["quality"][split]
    head = ("| Baseline | P@5 | R@5 | P@10 | R@10 | MRR (95% CI) | NDCG@5 (95% CI) | NDCG@10 (95% CI) | Kendall tau | "
            "Students | Unscored pairs |")
    lines = [head, "|" + "---|" * 11]
    for k in ORDER:
        r, m = q[k], q[k]["mean"]
        tau = "—" if m.get("kendall_tau") is None else f"{m['kendall_tau']:.3f}"
        lines.append(f"| {report['baselines'][k]['display']} | {m['precision@5']:.3f} | {m['recall@5']:.3f} | "
                     f"{m['precision@10']:.3f} | {m['recall@10']:.3f} | {_ci(r, 'mrr')} | {_ci(r, 'ndcg@5')} | "
                     f"{_ci(r, 'ndcg@10')} | {tau} | {r['students_evaluated']} (+{r['students_excluded_no_relevant']} "
                     f"without a relevant job) | {r['unscored_pairs']} |")
    return lines


def _comparison_lines(report: dict, split: str) -> list[str]:
    names = {k: report["baselines"][k]["display"].split(":")[0] for k in ORDER}
    secondary = report["config"]["evaluation"].get("secondary_metrics", [])
    lines = ["| Question | Comparison (a − b) | Δ " + report["config"]["evaluation"]["primary_metric"].upper()
             + " | Paired 95% CI | Verdict | " + " | ".join(f"Secondary: Δ {m.upper()} (95% CI)" for m in secondary) + " |",
             "|---|---|---|---|---|" + "---|" * len(secondary)]
    for rq, rows in report["comparisons"][split].items():
        for c in rows:
            if "mean_difference" not in c:
                lines.append(f"| {rq} | {names[c['a']]} − {names[c['b']]} | — | — | {c['verdict']} |")
                continue
            verdict = c["verdict"].replace(c["a"], names[c["a"]]).replace(c["b"], names[c["b"]])
            extra = "".join(f" {x['mean_difference']:+.3f} ({x['ci95'][0]:+.3f} to {x['ci95'][1]:+.3f})"
                            f"{'*' if x['distinguishable'] else ''} |" for x in c.get("secondary", {}).values())
            lines.append(f"| {rq} | {names[c['a']]} − {names[c['b']]} | {c['mean_difference']:+.3f} | "
                         f"{c['ci95'][0]:+.3f} to {c['ci95'][1]:+.3f} | {verdict} |{extra}")
    return lines


def render_markdown(r: dict) -> str:
    ev, answer = r["config"]["evaluation"], r["config"]["answer_split"]
    lab = r["labels"]
    out = [f"# Baseline comparison (spec §34): {r['experiment_id']}", "",
           f"Generated by `src/evaluation/baseline_comparison.py` on {r['finished_at']} ({r['runtime_seconds']} s). "
           f"Config `{r['config']['file']}` ({r['config']['version']}, sha256 {r['config']['sha256'][:12]}), fixed before "
           f"this run. Dataset {r['dataset_version']}.", "",
           "## Benchmark", "",
           f"- Labels: `{lab['path']}` (sha256 {lab['sha256'][:12]}), **{lab['source']}**, {lab['labellers']} labeller, "
           f"rubric-v1; splits {lab['split_breakdown']}.",
           f"- Grades per split (0/1/2/3): " + "; ".join(f"{s} {g}" for s, g in lab["grades_per_split"].items()) + ".",
           f"- Every baseline ranks each student's judged jobs (12 per student) on the identical pairs; relevant = grade "
           f">= {ev['relevance_threshold']}; ties by job_id; CIs: paired bootstrap over students ({ev['bootstrap_resamples']} "
           "resamples).",
           f"- **Answer split: `{answer}`**, held back by Phases 6 and 8 and used here for the first time. `validation` is "
           "where Phases 6 and 8 selected E5 and the matching model, so it favours them by construction; it is shown only "
           "to check that this module reproduces those phases' numbers.", ""]
    for split in r["config"]["splits"]:
        role = "answers the research questions" if split == answer else "reproduction check only"
        out += [f"## `{split}` split ({role})", ""] + _table(r, split) + [""]
        out += [f"**Pre-registered comparisons** (primary metric {ev['primary_metric'].upper()}; a verdict of "
                "\"outperforms\" needs a paired 95% CI that excludes 0):", ""] + _comparison_lines(r, split) + [
                "", "Secondary metrics were added after the first run and are never decisive (* = CI excludes 0; with "
                f"{len(ev.get('secondary_metrics', []))} metrics x {sum(len(v) for v in r['comparisons'][split].values())} "
                "comparisons, an occasional marginal * is expected by chance).", ""]
        v = r["validity"][split]
        out += [f"- {v['line']}. Mean per-student Kendall tau with the labels: {v['mean_per_student_kendall_tau']} "
                "(>= 0.9 would mean the labels look model-generated).", ""]
    if r["reproduction"]:
        out += ["## Reproduction check (validation split)", "",
                "| Baseline | Metric | Reported by | Reported | This run | Same model version | Match |",
                "|---|---|---|---|---|---|---|"]
        for x in r["reproduction"]:
            out.append(f"| {x['baseline']} | {x['metric']} | `{x['source']}` | {x['reported']:.3f} | {x['here']:.3f} | "
                       f"{'yes' if x['same_model_version'] else 'no'} | {'yes' if x['match'] else '**no**'} |")
        out.append("")
    out += ["## How each baseline scores a pair", ""]
    for k in ORDER:
        b = r["baselines"][k]
        out.append(f"- **{b['display']}** (`{b['version']}`): {b['method']}")
    kc = r["keyword_coverage"]
    out += ["", f"Keyword coverage on the evaluated pairs: {kc['students_with_skills']}/{kc['students']} students and "
            f"{kc['jobs_with_skills']}/{kc['jobs']} jobs have at least one normalised skill; {kc['pairs_with_any_match']} "
            "pairs share at least one skill. Pairs a baseline can't score rank last (counted as \"unscored\").", "",
            "## Limitations", "",
            "- 15 test students (180 pairs) and one labeller: most differences between neighbouring baselines are not "
            "statistically distinguishable, and this report says so per comparison.",
            "- The labelling pool was built from the rankings of TF-IDF, SBERT, E5, BGE and MiniLM (TREC-style). B1, B3 "
            "and B5 did not nominate jobs. Condensed lists (only judged jobs are ranked) keep this fair to them, but none "
            "of the baselines is tested on jobs nobody nominated.",
            "- Ranking judged jobs evaluates scoring (Stages 3-4), not Stage 1 eligibility or Stage 2 top-50 retrieval: a "
            "judged job the production pipeline would filter out still gets ranked here.",
            "- B3 is not purely structured: 6 of its 20 features are E5 similarity percentiles (weight 0.38 of 1.00).",
            "- The job corpus is synthetic and templated (376 distinct role texts). Results describe this data.", ""]
    return "\n".join(out)


def write_outputs(report: dict) -> dict[str, str]:
    folder = RESULTS / "baseline_comparison"
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / report["experiment_id"]
    data = {k: v for k, v in report.items() if k != "markdown"}
    base.with_suffix(".json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    base.with_suffix(".md").write_text(report["markdown"], encoding="utf-8")
    shutil.copyfile(base.with_suffix(".json"), RESULTS / "baseline_comparison.json")
    shutil.copyfile(base.with_suffix(".md"), RESULTS / "baseline_comparison.md")
    return {"json": str(base.with_suffix(".json")), "markdown": str(base.with_suffix(".md"))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 14 Part A: the five baselines on the human benchmark")
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args()
    rep = run(args.config)
    paths = write_outputs(rep)
    print(rep["markdown"])
    print(f"written: {paths}")
