"""Choose the student-job pairs the team grades on the labelling page (human benchmark, option 1).

Students: 30 by default, drawn only from the validation and test splits (src/evaluation/splits.py),
so the human labels are evaluation-only and no student can leak into training.
  * engineering and IT students only: the system serves an engineering college, so labels are
    not spent on the Kaggle non-engineering resumes (law, HR, arts, ...)
  * 2 placement-cell students for every Kaggle student, spread over domains by systematic
    sampling (sorted by domain, every n-th student from a seeded start)
  * at least 15 words of profile text, so there is something to judge
Jobs: one representative per role (the 376 distinct jobs, see load_benchmark_data).

Per student, 12 jobs (TREC-style pooling, the standard way to build a judged test collection):
  * 10 nominated round-robin by the five compared models (TF-IDF floor, SBERT, E5, BGE, MiniLM):
    each model's 1st choice, then each model's 2nd, and so on, in a seeded per-student model
    order, skipping jobs already in the pool. Every model gets the same number of slots.
  * 2 drawn uniformly at random from the remaining jobs, as a check on pool bias.
  * shown in a seeded random order. The page never shows which model nominated a job, or any score.

The models only choose WHICH pairs are shown; people assign every grade. That is why this is not
circular evaluation (spec §8): the grades are independent of the models. The per-model ranks of
each pooled job are kept in labelling_pool_sources.csv for the pool-bias analysis. Don't open it
while labelling.

Usage:  python -m src.evaluation.labelling_pool
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.embeddings.embedding_model_benchmark import (MODEL_CONFIGS, load_benchmark_data, load_encoder,
                                                      score_matrix, student_text)
from src.evaluation.splits import SPLIT_VERSION, split_for

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCHMARK_DIR = os.path.join(ROOT, "data", "benchmark")
POOL_MODELS = ("tfidf", "sbert", "e5", "bge", "minilm")
NON_ENGINEERING_BRANCH = "Non-Engineering / Management"
SEED = 42
MIN_WORDS = 15


def select_students(students: pd.DataFrame, n_students: int = 30, seed: int = SEED) -> pd.DataFrame:
    s = students.copy()
    s["split"] = s["student_id"].map(split_for)
    s["words"] = [len(student_text(r).split()) for r in s.to_dict("records")]
    eligible = s[s["split"].isin(["validation", "test"]) & (s["branch"] != NON_ENGINEERING_BRANCH)
                 & (s["words"] >= MIN_WORDS)]
    per_split = n_students // 2
    quotas = {"placement_cell": per_split * 2 // 3, "kaggle_resume_dataset": per_split - per_split * 2 // 3}
    rng = np.random.default_rng(seed)
    chosen = []
    for split in ("validation", "test"):
        for source, quota in quotas.items():
            group = eligible[(eligible["split"] == split) & (eligible["source_dataset"] == source)]
            group = group.sort_values(["domain", "student_id"]).reset_index(drop=True)
            if len(group) < quota:
                raise ValueError(f"only {len(group)} eligible {source} students in {split}; need {quota}")
            step = len(group) / quota
            start = rng.uniform(0, step)
            chosen.append(group.iloc[[int(start + i * step) for i in range(quota)]])
    return pd.concat(chosen).reset_index(drop=True)


def model_rankings(students: pd.DataFrame, jobs: pd.DataFrame, models=POOL_MODELS, device: str = "cpu") -> dict:
    """{model: {student_id: [job_id, ...] best first}} under the full-text protocol."""
    rankings, versions = {}, {}
    for key in models:
        encoder = load_encoder(key, device if MODEL_CONFIGS[key]["kind"] != "tfidf" else "cpu")
        sids, jids, m = score_matrix(encoder, students, jobs, "full_text")
        versions[key] = encoder.version
        rankings[key] = {}
        for r, sid in enumerate(sids):
            scores = np.where(np.isnan(m[r]), -np.inf, m[r])
            order = sorted(range(len(jids)), key=lambda c: (-scores[c], jids[c]))  # ties by job_id
            rankings[key][sid] = [jids[c] for c in order]
        del encoder
    return {"rankings": rankings, "versions": versions}


def _student_rng(seed: int, student_id: str) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}:{student_id}".encode()).digest()[:8], "big"))


def build_pool(selected: pd.DataFrame, jobs: pd.DataFrame, rankings: dict, n_pooled: int = 10, n_random: int = 2,
               seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_jobs = list(jobs["job_id"])
    pool_rows, source_rows = [], []
    for sid, split in zip(selected["student_id"], selected["split"]):
        rng = _student_rng(seed, sid)
        order = list(rankings)
        rng.shuffle(order)
        pooled, depth = [], 0
        while len(pooled) < n_pooled:
            for model in order:
                job = rankings[model][sid][depth]
                if job not in pooled and len(pooled) < n_pooled:
                    pooled.append(job)
            depth += 1
        rest = [j for j in all_jobs if j not in pooled]
        randoms = [rest[i] for i in rng.choice(len(rest), n_random, replace=False)]
        shown = pooled + randoms
        display = rng.permutation(len(shown))
        for position, idx in enumerate(display, start=1):
            job = shown[idx]
            pool_rows.append({"student_id": sid, "job_id": job, "split": split, "display_order": position})
            source_rows.append({"student_id": sid, "job_id": job, "random_fill": job in randoms,
                                **{f"rank_{m}": rankings[m][sid].index(job) + 1 for m in rankings}})
    return pd.DataFrame(pool_rows), pd.DataFrame(source_rows)


def write_pool(out_dir: str = BENCHMARK_DIR, n_students: int = 30, n_pooled: int = 10, n_random: int = 2,
               device: str = "cpu", seed: int = SEED) -> dict:
    students, jobs, _ = load_benchmark_data()
    selected = select_students(students, n_students, seed)
    ranked = model_rankings(selected, jobs, POOL_MODELS, device)
    pool, sources = build_pool(selected, jobs, ranked["rankings"], n_pooled, n_random, seed)
    version = "pool-" + hashlib.sha256(pool.to_csv(index=False).encode()).hexdigest()[:12]
    os.makedirs(out_dir, exist_ok=True)
    pool.insert(0, "pool_version", version)
    pool.to_csv(os.path.join(out_dir, "labelling_pool.csv"), index=False)
    sources.to_csv(os.path.join(out_dir, "labelling_pool_sources.csv"), index=False)
    manifest = {
        "pool_version": version, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed, "split_version": SPLIT_VERSION, "students": int(selected["student_id"].nunique()),
        "pairs": int(len(pool)), "jobs_per_student": n_pooled + n_random,
        "students_by_split": selected["split"].value_counts().to_dict(),
        "students_by_source": selected["source_dataset"].value_counts().to_dict(),
        "students_by_domain": selected["domain"].value_counts().to_dict(),
        "method": f"round-robin pooling of each model's top-ranked jobs ({n_pooled} per student, seeded "
                  f"model order) + {n_random} uniform random jobs; seeded display order",
        "models": {k: ranked["versions"][k] for k in POOL_MODELS}, "protocol": "full_text", "device": device,
        "job_corpus": f"{len(jobs)} role representatives (lowest job_id per role)",
        "eligibility": f"validation/test split; branch != {NON_ENGINEERING_BRANCH!r}; >= {MIN_WORDS} words of profile text",
        "random_fill_pairs": int(sources["random_fill"].sum()),
    }
    with open(os.path.join(out_dir, "labelling_pool.manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--students", type=int, default=30)
    ap.add_argument("--device", default="cpu", help="cpu (default) or mps; scores agree to ~1e-7 (see the benchmark)")
    args = ap.parse_args()
    print(json.dumps(write_pool(n_students=args.students, device=args.device), indent=2))
