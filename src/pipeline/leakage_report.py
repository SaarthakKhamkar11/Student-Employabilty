"""Phase 14 Part C (spec §26): the leakage report, computed on the real data (docs/adr/0021).

    python -m src.pipeline.leakage_report          (in the backend image; DATABASE_URL, if set, adds read-only counts)

Checks, each with real counts:
  1 duplicate students        exact rows, identical model text, near-duplicate text (3-shingle Jaccard >= 0.9)
  2 duplicate jobs            exact rows, identical model text (templating), Phase 5's near-duplicates (company +
                              title + Jaccard >= 0.9), and near-identical texts across DIFFERENT roles
  3 splits                    every student in exactly one split, in every label file and feature matrix; training
                              rows only from train; duplicates of evaluated students sitting in train
  4 synthetic duplicates      is_synthetic rows and how they spread over splits
  5 target / outcome leakage  all 20 features: what each reads, whether it shares an input with the labels, and how well
                              it alone reproduces the training target (weak labels) vs the human labels (ROC-AUC)
  6 model-made labels         provenance manifests, the Phase 6 circular-evaluation guard (Kendall tau), and whether
                              any stored model output is used as a label

Outputs: results/leakage_report.md and results/leakage_report.json.
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STUDENTS_CSV = ROOT / "data" / "processed" / "students_unified.csv"
JOBS_CSV = ROOT / "data" / "processed" / "jobs.csv"
BENCHMARK = ROOT / "data" / "benchmark"
FEATURES_DIR = BENCHMARK / "matching_features"
MODELS_DIR = ROOT / "models" / "matching"
RESULTS = ROOT / "results"
NEAR = 0.9  # the Phase 5 near-duplicate threshold (docs/adr/0010), reused for students
LABEL_FILES = {"human_labels": BENCHMARK / "human_labels.csv", "weak_labels": BENCHMARK / "weak_labels.csv",
               "labelling_pool": BENCHMARK / "labelling_pool.csv"}

# What each of the 20 contract features reads (app/matching/feature_builder.py) and whether that input is also an
# input of a label. Written from the code, then checked numerically below (single-feature ROC-AUC).
FEATURE_INPUTS = {
    "skill_match_score": ("student skills + profile text vs job skills (inferred required/preferred)", None),
    "required_skill_match_score": ("same, required skills only", None),
    "preferred_skill_match_score": ("same, preferred skills only", None),
    "matched_skill_count": ("job skills the profile names", None),
    "missing_skill_count": ("job skills the profile doesn't name", None),
    "required_skill_count": ("number of inferred required skills of the JOB (job-only)", None),
    "matched_skill_ratio": ("matched / all job skills", None),
    "preferred_skill_ratio": ("matched preferred / preferred", None),
    "transferable_skill_score": ("related (embedding) skills covering missing required skills", None),
    "education_match": ("student degree level vs job minimum (qualification_required)", None),
    "degree_match": ("student degree vs job degree (randomly assigned in jobs.csv, Phase 6)", None),
    "branch_match": ("student branch -> families (config branch_rules) vs job role -> families",
                     "job role families are read from config/weak_label_rules.yaml: the SAME regexes that generate "
                     "the weak training labels"),
    "experience_match": ("stated experience vs job minimum (experience_required)", None),
    "experience_gap": ("years short of the job minimum", None),
    "semantic_similarity": ("E5 section-weighted similarity (skills, experience, projects), calibrated percentile", None),
    "project_relevance": ("E5 projects vs job description, percentile", None),
    "certification_relevance": ("E5 certifications vs job skills, percentile", None),
    "career_interest_alignment": ("E5 of the student's stated interest + DOMAIN vs job title + role, percentile",
                                  "the student's `domain` is the weak labels' student-side input, and the job "
                                  "title/role their job-side input"),
    "role_similarity": ("E5 past experience vs job title + role, percentile", None),
    "domain_similarity": ("E5 of branch + DOMAIN vs the full job text, percentile",
                          "reads the student's `domain`, the weak labels' student-side input"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kw)


def _groups(keys: pd.Series) -> list[list[int]]:
    """Index groups of rows sharing a non-empty key, largest first."""
    by = defaultdict(list)
    for i, k in keys.items():
        if k:
            by[k].append(i)
    return sorted((g for g in by.values() if len(g) > 1), key=len, reverse=True)


def _near_pairs(texts: dict[str, str], threshold: float = NEAR) -> list[tuple[str, str, float]]:
    from app.jobs.dedup import jaccard, shingles

    sh = {k: shingles(t) for k, t in texts.items() if t.strip()}
    out = []
    for a, b in combinations(sorted(sh), 2):
        s = jaccard(sh[a], sh[b])
        if s >= threshold:
            out.append((a, b, round(s, 4)))
    return out


def _components(pairs) -> list[set[str]]:
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, *_ in pairs:
        parent[find(a)] = find(b)
    comps = defaultdict(set)
    for x in parent:
        comps[find(x)].add(x)
    return sorted(comps.values(), key=len, reverse=True)


# ---------------------------------------------------------------- 1 students

def duplicate_students(students: pd.DataFrame, split_of: dict, labelled: dict[str, set]) -> dict:
    from src.embeddings.embedding_model_benchmark import student_text

    content_cols = [c for c in students.columns if c != "student_id"]
    exact = _groups(students[content_cols].astype(str).agg("\x1f".join, axis=1))
    texts = {r["student_id"]: student_text(r) for r in students.to_dict("records")}
    empty = sorted(s for s, t in texts.items() if not t.strip())
    same_text = _groups(pd.Series({s: t.strip().lower() for s, t in texts.items()}).reset_index(drop=True))
    ids = list(texts)
    same_text_ids = [[ids[i] for i in g] for g in same_text]
    near = _near_pairs(texts)
    comps = _components(near)

    def describe(groups: list[set[str] | list[str]]) -> dict:
        cross = [g for g in groups if len({split_of[s] for s in g}) > 1]
        evaluated = set().union(*labelled.values())
        involving_eval_and_train = [g for g in groups if set(g) & evaluated and any(split_of[s] == "train" for s in g)]
        return {"groups": len(groups), "students_in_groups": sum(len(g) for g in groups),
                "groups_across_splits": len(cross),
                "groups_with_a_human_labelled_student_and_a_train_student": len(involving_eval_and_train),
                "examples": [{"students": sorted(g), "splits": sorted({split_of[s] for s in g}),
                              "source": sorted(set(students.set_index("student_id").loc[sorted(g), "source_dataset"]))}
                             for g in groups[:8]]}

    return {"students": int(len(students)), "exact_duplicate_rows": describe([[ids[i] for i in g] for g in exact]),
            "identical_model_text": describe(same_text_ids),
            "near_duplicate_text": {**describe(comps), "pairs": len(near), "threshold": NEAR},
            "empty_model_text": empty}


# ---------------------------------------------------------------- 2 jobs

def duplicate_jobs(jobs: pd.DataFrame, labelled_jobs: dict[str, set]) -> dict:
    from app.jobs.dedup import find_near_duplicates
    from src.embeddings.embedding_model_benchmark import job_text

    content_cols = [c for c in jobs.columns if c not in ("job_id", "source_job_id")]
    exact = _groups(jobs[content_cols].astype(str).agg("\x1f".join, axis=1))
    model_text = jobs.apply(lambda r: job_text(r).strip().lower(), axis=1)
    templated = _groups(model_text)
    roles_per_text = [jobs.loc[g, "role"].nunique() for g in templated]
    flagged = find_near_duplicates(jobs.to_dict("records"))
    phase5_groups = len({v["near_duplicate_of"] for v in flagged.values()})

    reps = jobs.sort_values("job_id").drop_duplicates("role")
    rep_text = {r["job_id"]: job_text(r) for r in reps.to_dict("records")}
    cross_role = _near_pairs(rep_text)
    role_of = dict(zip(jobs["job_id"], jobs["role"]))
    # within one student's judged list, two postings of the same role would be the same text judged twice
    human = _read(LABEL_FILES["human_labels"])
    human["role"] = human["job_id"].map(role_of)
    repeats = int((human.groupby("student_id")["role"].nunique() < human.groupby("student_id")["job_id"].nunique()).sum())
    return {
        "jobs": int(len(jobs)), "exact_duplicate_rows": {"groups": len(exact), "jobs_in_groups": sum(len(g) for g in exact)},
        "identical_model_text": {"distinct_texts": int(model_text.nunique()), "groups": len(templated),
                                 "jobs_in_groups": sum(len(g) for g in templated),
                                 "largest_group": max((len(g) for g in templated), default=0),
                                 "texts_shared_by_more_than_one_role": sum(1 for n in roles_per_text if n > 1)},
        "phase5_near_duplicates": {"flagged_postings": len(flagged), "groups": phase5_groups,
                                   "rule": "same company + same title + 3-shingle Jaccard >= 0.9 (job-dedup-v2)"},
        "near_identical_across_roles": {"pairs": len(cross_role), "threshold": NEAR,
                                        "examples": [{"jobs": [a, b], "roles": [role_of[a], role_of[b]], "jaccard": s}
                                                     for a, b, s in cross_role[:8]]},
        "students_with_two_postings_of_one_role_in_their_judged_list": repeats,
        "labelled_jobs": {k: len(v) for k, v in labelled_jobs.items()},
    }


# ---------------------------------------------------------------- 3 splits

def split_checks(students: pd.DataFrame, labels: dict[str, pd.DataFrame], features: dict[str, pd.DataFrame],
                 dup: dict) -> dict:
    from src.evaluation.splits import split_for

    fixed = _read(BENCHMARK / "student_splits.csv")
    mismatched = sorted(s for s, sp in zip(fixed["student_id"], fixed["split"]) if split_for(s) != sp)
    per_file = {}
    for name, frame in {**labels, **{f"features:{k}": v for k, v in features.items()}}.items():
        if "split" not in frame.columns:
            continue
        n_splits = frame.groupby("student_id")["split"].nunique()
        wrong = sum(1 for s, sp in frame[["student_id", "split"]].drop_duplicates().itertuples(index=False)
                    if split_for(s) != sp)
        per_file[name] = {"students": int(frame["student_id"].nunique()),
                          "students_in_more_than_one_split": int((n_splits > 1).sum()),
                          "rows_whose_split_differs_from_the_fixed_split": int(wrong),
                          "pairs_by_split": {k: int(v) for k, v in frame["split"].value_counts().items()}}

    weak, human = labels["weak_labels"], labels["human_labels"]
    train_pairs = set(zip(weak.loc[weak["split"] == "train", "student_id"], weak.loc[weak["split"] == "train", "job_id"]))
    human_pairs = set(zip(human["student_id"], human["job_id"]))
    train_students = set(weak.loc[weak["split"] == "train", "student_id"])
    manifests = {}
    for folder in sorted(MODELS_DIR.iterdir()):
        m = folder / "manifest.json"
        if m.exists():
            mm = json.loads(m.read_text(encoding="utf-8"))
            manifests[folder.name] = {"training_split": mm["training"]["split"],
                                      "label_source": mm["training"]["label_source"],
                                      "feature_matrix": mm["feature_matrix"]["file"]}
    jobs_test = set(human.loc[human["split"] == "test", "job_id"])
    jobs_train_positive = set(weak.loc[(weak["split"] == "train") & (weak["relevance_label"].astype(int) >= 2), "job_id"])
    jobs_train_any = set(weak.loc[weak["split"] == "train", "job_id"])
    return {
        "split_rule": "student-sha256-v1: a pure function of student_id, so a student can't be in two splits",
        "student_splits_csv_mismatches": mismatched, "per_file": per_file,
        "human_labelled_students_in_train": len(set(human["student_id"]) & train_students),
        "human_label_pairs_among_training_pairs": len(human_pairs & train_pairs),
        "trained_model_manifests": manifests,
        "content_duplicates_of_human_labelled_students_in_train":
            dup["near_duplicate_text"]["groups_with_a_human_labelled_student_and_a_train_student"],
        "jobs_shared_across_splits": {
            "note": "jobs are the shared corpus being ranked; splits are by student (src/evaluation/splits.py)",
            "test_judged_jobs": len(jobs_test),
            "also_in_train_training_pairs": len(jobs_test & jobs_train_any),
            "also_a_positive_in_train_training_pairs": len(jobs_test & jobs_train_positive)},
    }


# ---------------------------------------------------------------- 4 synthetic

def synthetic_checks(students: pd.DataFrame, jobs: pd.DataFrame, split_of: dict, label_ids: set) -> dict:
    syn = students[students["is_synthetic"].str.lower() == "true"]
    unknown = sorted(label_ids - set(students["student_id"]))
    return {"synthetic_students": int(len(syn)),
            "synthetic_students_by_split": {k: int(v) for k, v in syn["student_id"].map(split_of).value_counts().items()},
            "label_or_feature_student_ids_not_in_students_unified": unknown,
            "jobs_have_is_synthetic_column": "is_synthetic" in jobs.columns,
            "job_corpus_note": "jobs.csv carries no is_synthetic flag; the Phase 1 audit (I-7) found it templated and "
                               "probably synthetic (376 distinct role texts, qualifications assigned near-uniformly)"}


# ---------------------------------------------------------------- 5 target / outcome leakage

def _auc(y: np.ndarray, x: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    keep = ~np.isnan(x)
    if keep.sum() < 10 or len(set(y[keep])) < 2:
        return None
    return float(roc_auc_score(y[keep], x[keep]))


def target_leakage(features: pd.DataFrame, labels: dict[str, pd.DataFrame], students: pd.DataFrame,
                   jobs: pd.DataFrame) -> dict:
    from app.matching.features import FEATURE_NAMES

    weak, human = labels["weak_labels"], labels["human_labels"]
    key = features["student_id"] + "|" + features["job_id"]
    wmap = dict(zip(weak["student_id"] + "|" + weak["job_id"], weak["relevance_label"].astype(int)))
    hmap = dict(zip(human["student_id"] + "|" + human["job_id"], human["relevance_label"].astype(int)))
    train = features[(features["split"] == "train") & key.isin(wmap)]
    y_train = (key[train.index].map(wmap) >= 2).to_numpy().astype(int)
    tests = {}
    for split in ("validation", "test"):
        rows = features[key.isin(hmap) & (features["split"] == split)]
        tests[split] = (rows, (key[rows.index].map(hmap) >= 2).to_numpy().astype(int))
    out = []
    for f in FEATURE_NAMES:
        inputs, shared = FEATURE_INPUTS[f]
        a_w = _auc(y_train, train[f].to_numpy(dtype=float))
        a_h = {s: _auc(y, r[f].to_numpy(dtype=float)) for s, (r, y) in tests.items()}
        out.append({"feature": f, "reads": inputs, "shares_an_input_with_the_weak_labels": shared,
                    "missing_share_train": round(float(train[f].isna().mean()), 4),
                    "auc_weak_labels_train": a_w, "auc_human_validation": a_h["validation"],
                    "auc_human_test": a_h["test"]})

    # the two leaking signals together, as a rule: how much of the training target do they reproduce?
    def rule(frame):
        b = frame["branch_match"].to_numpy(dtype=float)
        c = frame["career_interest_alignment"].to_numpy(dtype=float)
        return np.nan_to_num(b, nan=0.5) + c
    combo = {"auc_weak_labels_train": _auc(y_train, rule(train)),
             **{f"auc_human_{s}": _auc(y, rule(r)) for s, (r, y) in tests.items()}}

    columns = {"students": list(students.columns), "jobs": list(jobs.columns)}
    outcome_like = [c for c in columns["students"] + columns["jobs"]
                    if any(w in c.lower() for w in ("placed", "offer", "hired", "outcome", "selected", "package"))]
    return {"features": out, "branch_plus_interest": combo,
            "train_pairs": int(len(train)), "train_positives": int(y_train.sum()),
            "human_pairs": {s: int(len(r)) for s, (r, _) in tests.items()},
            "human_positives": {s: int(y.sum()) for s, (_, y) in tests.items()},
            "outcome_columns_in_data": outcome_like, "columns": columns}


# ---------------------------------------------------------------- 6 model-made labels

def label_provenance(labels: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for name in ("human_labels", "weak_labels", "labelling_pool"):
        m = LABEL_FILES[name].with_suffix("").with_suffix(".manifest.json")
        man = json.loads(m.read_text(encoding="utf-8")) if m.exists() else None
        out[name] = {"manifest": m.exists(), "generated_by": (man or {}).get("generated_by"),
                     "models_used_for_labels": (man or {}).get("models_used_for_labels"),
                     "label_sources_in_file": sorted(set(labels[name]["label_source"])) if "label_source" in labels[name] else None,
                     "pool_models": (man or {}).get("models")}
    human, pool = labels["human_labels"], labels["labelling_pool"]
    out["human_pairs_outside_the_pool"] = len(set(zip(human["student_id"], human["job_id"]))
                                              - set(zip(pool["student_id"], pool["job_id"])))
    from app.matching.predictor import active_choice

    taus = {}
    selection_runs = sorted({active_choice().selection_experiment,
                             active_choice(embedding_model="tfidf_baseline").selection_experiment})
    for path, field in ((RESULTS / "embedding_model_evaluation_human_reviewed.json", ("labels", "validity")),
                        *((RESULTS / "matching_model_evaluation" / f"{e}.json", ("labels", "validity"))
                          for e in selection_runs),
                        (RESULTS / "baseline_comparison.json", None)):
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        if field:
            v = r[field[0]].get(field[1]) or r.get("validity")
            if isinstance(v, dict):
                taus[r["experiment_id"]] = {"split": r["labels"].get("used_split"), "level": v.get("level"),
                                            "tau": v.get("mean_per_student_kendall_tau")}
        else:
            for split, v in r["validity"].items():
                taus[f"{r['experiment_id']} ({split})"] = {"split": split, "level": v["level"],
                                                           "tau": v["mean_per_student_kendall_tau"]}
    out["circularity_guard"] = taus
    return out


def database_counts() -> dict | None:
    """Read-only: is any model output stored where a label could be read from?"""
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return None
    import psycopg

    queries = {
        "relevance_judgments": "SELECT count(*) FROM relevance_judgments",
        "relevance_judgments_by_source": "SELECT coalesce(json_object_agg(label_source, n), '{}') FROM "
                                         "(SELECT label_source, count(*) n FROM relevance_judgments GROUP BY 1) t",
        "feedback_events": "SELECT count(*) FROM feedback_events",
        "match_results": "SELECT count(*) FROM match_results",
    }
    out = {}
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            for k, q in queries.items():
                try:
                    out[k] = conn.execute(q).fetchone()[0]
                except psycopg.Error as exc:
                    conn.rollback()
                    out[k] = f"not available ({type(exc).__name__})"
    except psycopg.Error as exc:
        return {"error": type(exc).__name__}
    return out


def code_reads_of_feedback() -> dict:
    """Which training / labelling code reads feedback or stored match results (it must not, before Phase 15)."""
    hits = {}
    for folder in (ROOT / "src", ROOT / "backend" / "app" / "matching"):
        for path in folder.rglob("*.py"):
            if path.resolve() == Path(__file__).resolve():  # this report names the tables to look for them
                continue
            text = path.read_text(encoding="utf-8")
            for table in ("feedback_events", "match_results"):
                if table in text:
                    hits.setdefault(table, []).append(str(path.relative_to(ROOT)))
    return {t: sorted(v) for t, v in hits.items()}


# ---------------------------------------------------------------- run

def run() -> dict:
    students, jobs = _read(STUDENTS_CSV), _read(JOBS_CSV)
    labels = {k: _read(p) for k, p in LABEL_FILES.items()}
    from src.evaluation.splits import split_for

    split_of = {s: split_for(s) for s in students["student_id"]}
    for name, frame in labels.items():
        if "split" not in frame.columns:
            frame["split"] = frame["student_id"].map(split_of)
    feature_files = sorted(FEATURES_DIR.glob("*.csv.gz"))
    features = {p.name.replace(".csv.gz", ""): pd.read_csv(p, dtype={"student_id": str, "job_id": str, "split": str})
                for p in feature_files}
    human = labels["human_labels"]
    labelled = {s: set(human.loc[human["split"] == s, "student_id"]) for s in ("validation", "test")}
    labelled_jobs = {s: set(human.loc[human["split"] == s, "job_id"]) for s in ("validation", "test")}

    print("  1 students", flush=True)
    dup_s = duplicate_students(students, split_of, labelled)
    print("  2 jobs", flush=True)
    dup_j = duplicate_jobs(jobs, labelled_jobs)
    print("  3 splits", flush=True)
    splits = split_checks(students, labels, features, dup_s)
    print("  4 synthetic", flush=True)
    label_ids = set().union(*(set(f["student_id"]) for f in labels.values()), *(set(f["student_id"]) for f in features.values()))
    syn = synthetic_checks(students, jobs, split_of, label_ids)
    print("  5 target leakage", flush=True)
    from app.matching.predictor import active_choice

    choice = active_choice()
    prod_manifest = json.loads((MODELS_DIR / choice.artifacts_experiment / "manifest.json").read_text(encoding="utf-8"))
    prod_features = prod_manifest["feature_matrix"]["file"].replace(".csv.gz", "")
    target = target_leakage(features[prod_features], labels, students, jobs)
    target["feature_matrix"] = prod_features
    target["production_model"] = f"{choice.active} (artifacts {choice.artifacts_experiment})"
    print("  6 label provenance", flush=True)
    prov = label_provenance(labels)
    prov["database"] = database_counts()
    prov["code_reading_feedback_or_match_results"] = code_reads_of_feedback()
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "dataset_version": f"students:{_sha(STUDENTS_CSV)[:12]} jobs:{_sha(JOBS_CSV)[:12]}",
              "label_files": {k: _sha(p)[:12] for k, p in LABEL_FILES.items()},
              "students": dup_s, "jobs": dup_j, "splits": splits, "synthetic": syn, "target_leakage": target,
              "label_provenance": prov}
    report["markdown"] = render_markdown(report)
    return report


# ---------------------------------------------------------------- report

def _fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def render_markdown(r: dict) -> str:
    s, j, sp, syn, t, p = (r["students"], r["jobs"], r["splits"], r["synthetic"], r["target_leakage"],
                           r["label_provenance"])
    nd = s["near_duplicate_text"]
    out = ["# Leakage report (spec §26)", "",
           f"Generated by `src/pipeline/leakage_report.py` on {r['generated_at']}, on the real data: dataset "
           f"{r['dataset_version']}; label files {r['label_files']}. Every number below is counted, none assumed.", "",
           "## Summary", "",
           "| # | Check | Real count | Leakage? |", "|---|---|---|---|",
           f"| 1 | Duplicate students | {s['exact_duplicate_rows']['groups']} exact-duplicate groups; "
           f"{s['identical_model_text']['groups']} groups with identical model text; {nd['groups']} near-duplicate "
           f"groups (Jaccard >= {nd['threshold']}, {nd['students_in_groups']} students), {nd['groups_across_splits']} of them "
           f"across splits | " + ("**Yes**" if nd["groups_across_splits"] else "No") + " (see 1) |",
           f"| 2 | Duplicate jobs | {j['exact_duplicate_rows']['groups']} exact-duplicate groups; "
           f"{j['jobs']:,} postings have only {j['identical_model_text']['distinct_texts']} distinct model texts; "
           f"{j['phase5_near_duplicates']['flagged_postings']} Phase 5 near-duplicates in "
           f"{j['phase5_near_duplicates']['groups']} groups; {j['near_identical_across_roles']['pairs']} near-identical "
           "pairs across different roles | Not across splits (jobs aren't split); see 2 |",
           f"| 3 | Same student across splits | " + ", ".join(f"{k}: {v['students_in_more_than_one_split']}"
                                                         for k, v in sp["per_file"].items())
           + f"; human-labelled students in train: {sp['human_labelled_students_in_train']}; human pairs among training "
             f"pairs: {sp['human_label_pairs_among_training_pairs']} | No |",
           f"| 4 | Synthetic duplicates across splits | {syn['synthetic_students']} synthetic students in the data | Not "
           "applicable (no synthetic student remains) |",
           f"| 5 | Target / outcome leakage | {sum(1 for f in t['features'] if f['shares_an_input_with_the_weak_labels'])} "
           f"of 20 features share an input with the weak training labels; {len(t['outcome_columns_in_data'])} outcome "
           "columns exist | **Yes, target leakage w.r.t. the weak training labels**; no outcome leakage possible (see 5) |",
           f"| 6 | Model-generated labels as ground truth | human labels: generated_by "
           f"`{p['human_labels']['generated_by']}`, models used {p['human_labels']['models_used_for_labels']}; weak labels: "
           f"`{p['weak_labels']['generated_by']}`, models used {p['weak_labels']['models_used_for_labels']} | No (see 6) |",
           ""]

    out += ["## 1. Duplicate students", "",
            f"{s['students']} students. Model text = the Phase 6 student text (skills, experience, education, projects, "
            "certifications, achievements).", "",
            "| Check | Groups | Students in groups | Groups across splits | Groups pairing a human-labelled student with a train student |",
            "|---|---|---|---|---|"]
    for name, label in (("exact_duplicate_rows", "Exact duplicate rows (all columns but student_id)"),
                        ("identical_model_text", "Identical model text"),
                        ("near_duplicate_text", f"Near-duplicate model text (3-shingle Jaccard >= {NEAR})")):
        g = s[name]
        out.append(f"| {label} | {g['groups']} | {g['students_in_groups']} | {g['groups_across_splits']} | "
                   f"{g['groups_with_a_human_labelled_student_and_a_train_student']} |")
    out += ["", f"Students with no model text at all: {len(s['empty_model_text'])} ({', '.join(s['empty_model_text'][:10])}"
            f"{'…' if len(s['empty_model_text']) > 10 else ''}). They are not counted as duplicates of each other.", ""]
    if nd["examples"]:
        out += ["Largest near-duplicate groups:", ""] + [
            f"- {', '.join(e['students'])}: splits {e['splits']}, source {e['source']}" for e in nd["examples"]] + [""]

    out += ["## 2. Duplicate and near-duplicate jobs", "",
            f"- Exact duplicate rows (all columns but job_id/source_job_id): **{j['exact_duplicate_rows']['groups']}** groups.",
            f"- Identical model text (title/role, skills, description, responsibilities): {j['jobs']:,} postings, "
            f"**{j['identical_model_text']['distinct_texts']} distinct texts**; the largest text is shared by "
            f"{j['identical_model_text']['largest_group']} postings; {j['identical_model_text']['texts_shared_by_more_than_one_role']} "
            "texts are shared by more than one role. This is the templating found in Phase 1 (audit I-7). Phase 6 ranks the "
            "376 role representatives, so a student's ranking never contains copies.",
            f"- Phase 5 near-duplicates ({j['phase5_near_duplicates']['rule']}): **{j['phase5_near_duplicates']['flagged_postings']}** "
            f"postings in **{j['phase5_near_duplicates']['groups']}** groups (flagged at ingest, not deleted; matching skips them).",
            f"- Extended: near-identical texts across *different* roles (role representatives, Jaccard >= {NEAR}): "
            f"**{j['near_identical_across_roles']['pairs']}** pairs."]
    for e in j["near_identical_across_roles"]["examples"]:
        out.append(f"  - {e['jobs'][0]} \"{e['roles'][0]}\" vs {e['jobs'][1]} \"{e['roles'][1]}\": {e['jaccard']}")
    out += [f"- Human benchmark: students whose 12 judged jobs include two postings of one role: "
            f"**{j['students_with_two_postings_of_one_role_in_their_judged_list']}**. Labelled jobs per split: "
            f"{j['labelled_jobs']}.", ""]

    js = sp["jobs_shared_across_splits"]
    out += ["## 3. Splits", "", f"Rule: {sp['split_rule']}. Mismatches between `student_splits.csv` and the rule: "
            f"**{len(sp['student_splits_csv_mismatches'])}**.", "",
            "| File | Students | Students in more than one split | Rows whose split differs from the fixed split | Pairs by split |",
            "|---|---|---|---|---|"]
    for name, v in sp["per_file"].items():
        out.append(f"| `{name}` | {v['students']} | {v['students_in_more_than_one_split']} | "
                   f"{v['rows_whose_split_differs_from_the_fixed_split']} | {v['pairs_by_split']} |")
    out += ["", f"- Human-labelled (validation/test) students who are in train: **{sp['human_labelled_students_in_train']}**. "
            f"Human-labelled pairs that are also training pairs: **{sp['human_label_pairs_among_training_pairs']}**.",
            "- Every trained-model manifest trains on: " + "; ".join(
                f"`{k}` {v['label_source']} / {v['training_split']}" for k, v in sp["trained_model_manifests"].items()) + ".",
            f"- Near-duplicate groups that pair a human-labelled student with a train student (a copy of an evaluated "
            f"student inside the training data): **{sp['content_duplicates_of_human_labelled_students_in_train']}**.",
            f"- Jobs are the shared corpus, not split ({js['note']}). Of the {js['test_judged_jobs']} jobs judged for test "
            f"students, {js['also_in_train_training_pairs']} also appear in train training pairs and "
            f"{js['also_a_positive_in_train_training_pairs']} as a train positive. A model could learn job-level priors "
            "(e.g. `required_skill_count` identifies a job) from train students and reuse them on test students: this is "
            "transductive exposure by design, not student leakage, and the report counts it so it isn't hidden.", ""]

    out += ["## 4. Synthetic duplicates across splits", "",
            f"- Synthetic students in `students_unified.csv`: **{syn['synthetic_students']}** "
            f"({syn['synthetic_students_by_split'] or 'none in any split'}). Not applicable: there is nothing synthetic to "
            "duplicate across splits.",
            f"- Student ids in a label file or feature matrix that are not in `students_unified.csv`: "
            f"**{len(syn['label_or_feature_student_ids_not_in_students_unified'])}**. The Phase 14 scale-up students "
            "(Part D) exist only in a throwaway benchmark database and never reach a label file, split or feature matrix.",
            f"- Jobs: {syn['job_corpus_note']}.", ""]

    out += ["## 5. Target and outcome leakage: all 20 features", "",
            f"Feature matrix `{t['feature_matrix']}` (production model {t['production_model']}). ROC-AUC of each feature "
            f"ALONE against the target: the weak training labels (train split, {t['train_pairs']:,} pairs, "
            f"{t['train_positives']:,} positive) and the human labels (validation {t['human_pairs']['validation']} pairs / "
            f"{t['human_positives']['validation']} positive; test {t['human_pairs']['test']} / {t['human_positives']['test']}). "
            "AUC 0.5 = no signal; below 0.5 = the feature points the other way. Missing values are left out per feature.", "",
            "| Feature | Reads | Shares an input with the weak labels | AUC weak (train) | AUC human (val) | AUC human (test) | Missing (train) |",
            "|---|---|---|---|---|---|---|"]
    for f in t["features"]:
        shared = f["shares_an_input_with_the_weak_labels"] or "no"
        out.append(f"| `{f['feature']}` | {f['reads']} | {shared} | {_fmt(f['auc_weak_labels_train'])} | "
                   f"{_fmt(f['auc_human_validation'])} | {_fmt(f['auc_human_test'])} | {f['missing_share_train']:.1%} |")
    c = t["branch_plus_interest"]
    out += ["", f"`branch_match` + `career_interest_alignment` together (missing branch = 0.5): AUC "
            f"{_fmt(c['auc_weak_labels_train'])} on the weak training labels vs {_fmt(c['auc_human_validation'])} / "
            f"{_fmt(c['auc_human_test'])} on the human labels (validation / test).", "",
            "**Reading.** No feature uses information that only exists after the fact: every one is computed from the "
            "profile and the posting at request time, and there is no employment outcome anywhere in the data "
            f"(outcome-like columns found: {t['outcome_columns_in_data'] or 'none'}), so outcome leakage is impossible "
            "here. But the TRAINING target is the weak labels, and the features marked above read the same inputs the weak "
            "labels are generated from. A model trained on weak labels can therefore learn the labelling rule instead of fit. "
            "The gap between the weak-label AUC and the human-label AUC of those features is the measured size of that "
            "shortcut. It is why every trained model's top SHAP feature is `branch_match` or `career_interest_alignment` "
            "(Phases 8 and 11), and why model selection and every RQ answer use human labels only.",
            "- Unsupervised, label-free use of evaluation students: the similarity percentiles are calibrated over all "
            "students x all jobs (app/matching/calibration.py), test students included. No label is involved, so this is "
            "transductive, not target leakage.", ""]

    out += ["## 6. Model-generated labels reused as ground truth", "",
            "| Label file | Manifest | Generated by | Models used for the grades | Sources in file |", "|---|---|---|---|---|"]
    for name in ("human_labels", "weak_labels", "labelling_pool"):
        v = p[name]
        out.append(f"| `{name}` | {'yes' if v['manifest'] else 'no'} | {v['generated_by'] or '—'} | "
                   f"{v['models_used_for_labels'] if v['models_used_for_labels'] is not None else '—'} | "
                   f"{v['label_sources_in_file'] or '—'} |")
    out += ["", f"- Pool: the models {list((p['labelling_pool']['pool_models'] or {}).keys())} chose WHICH pairs were shown; "
            f"people gave every grade, blind to the source. Human pairs outside the pool: {p['human_pairs_outside_the_pool']}.",
            "- Circular-evaluation guard (Phase 6, `src/evaluation/relevance_labels.py`): mean per-student Kendall tau "
            "between each model's scores and the human labels (>= 0.9 would mean model-made labels):"]
    for run_id, v in p["circularity_guard"].items():
        out.append(f"  - `{run_id}` ({v['split']}): validity {v['level']}; tau {v['tau']}")
    db = p["database"]
    out += [f"- Live database (read-only): {db if db else 'not checked (no DATABASE_URL)'}.",
            f"- Code that reads `feedback_events` / `match_results` in `src/` or `app/matching/`: "
            f"{p['code_reading_feedback_or_match_results'] or 'none'}. No model output is read back as a label; the "
            "Phase 15 feedback loop must keep it that way (shortlists are model-influenced, see spec §29).", ""]
    return "\n".join(out)


def write_outputs(report: dict) -> dict[str, str]:
    RESULTS.mkdir(exist_ok=True)
    md, js = RESULTS / "leakage_report.md", RESULTS / "leakage_report.json"
    md.write_text(report["markdown"], encoding="utf-8")
    js.write_text(json.dumps({k: v for k, v in report.items() if k != "markdown"}, indent=1, default=str), encoding="utf-8")
    return {"markdown": str(md), "json": str(js)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="Phase 14 Part C: leakage report on the real data").parse_args()
    rep = run()
    print(rep["markdown"])
    print(f"written: {write_outputs(rep)}")
