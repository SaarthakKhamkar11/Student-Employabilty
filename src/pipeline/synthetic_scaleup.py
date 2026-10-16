"""Scale-up data for the Phase 14 performance benchmark ONLY (spec §28, §35; docs/adr/0021).

Phase 1 left no synthetic-generation script (PROJECT_AUDIT.md: the repo held preprocess.py, the embedding benchmark
and two CSVs), only the `is_synthetic` / `synthetic_fields` columns of STUDENT_SCHEMA. This module is that generator,
written in Phase 14 for one purpose: measuring latency, throughput and memory at more students than exist.

Students: the 286 real rows, then synthetic rows up to the requested size. Each synthetic student copies the
structured fields (degree, branch, graduation year, CGPA, gender, career interests, domain, country, source) of one
real "template" student, and draws each free-text field (skills, projects, internships, certifications,
achievements) from a real student of the SAME source dataset and branch, so a synthetic profile stays internally
consistent (a civil engineer gets civil skills and projects). Every synthetic row has is_synthetic=True and lists
every recombined field in synthetic_fields. Seeded: the same size and seed give the same rows, and a larger size
extends a smaller one (the first 1,000 students of the 5,000 set are the 1,000 set).

Jobs: a seeded sample of REAL postings from jobs.csv (no synthetic jobs are generated).

These rows are generated in memory and loaded only into a throwaway benchmark database. They are never written to
data/, never labelled, never split, and never used for training or evaluation: they say nothing about matching
quality, and the benchmark report flags every size that contains them.
"""

import numpy as np
import pandas as pd

SEED = 14
TEXT_FIELDS = ("skills_raw", "technical_projects", "internship_raw", "certifications", "achievements")
TEMPLATE_FIELDS = ("degree", "branch", "graduation_year", "gender", "avg_cgpa_sem5", "career_interests", "domain",
                   "country", "source_dataset")
SYNTHETIC_FIELDS = "; ".join(TEXT_FIELDS) + " (each recombined from a real student of the same source and branch); " \
                   + ", ".join(TEMPLATE_FIELDS) + " (copied from one real template student)"


def scale_students(real: pd.DataFrame, total: int, seed: int = SEED) -> pd.DataFrame:
    """The real students followed by total - len(real) synthetic ones (all columns as strings, like the CSV)."""
    real = real.astype(str).reset_index(drop=True)
    if total <= len(real):
        return real.iloc[:total].copy()
    rng = np.random.default_rng(seed)
    cohorts = {k: g.index.to_numpy() for k, g in real.groupby(["source_dataset", "branch"])}
    by_source = {k: g.index.to_numpy() for k, g in real.groupby("source_dataset")}
    rows = []
    for i in range(total - len(real)):
        template = real.iloc[int(rng.integers(len(real)))]
        pool = cohorts.get((template["source_dataset"], template["branch"]))
        if pool is None or len(pool) < 2:
            pool = by_source[template["source_dataset"]]
        row = {c: template[c] for c in TEMPLATE_FIELDS}
        for field in TEXT_FIELDS:
            row[field] = real.at[int(pool[rng.integers(len(pool))]), field]
        row.update(student_id=f"S{10001 + i:05d}", is_synthetic="True", synthetic_fields=SYNTHETIC_FIELDS)
        rows.append(row)
    synthetic = pd.DataFrame(rows, columns=real.columns)
    return pd.concat([real, synthetic], ignore_index=True)


def sample_jobs(jobs: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """n real postings, seeded, in job_id order (all of them when n >= the corpus)."""
    jobs = jobs.astype(str).sort_values("job_id").reset_index(drop=True)
    if n >= len(jobs):
        return jobs
    picked = np.sort(np.random.default_rng(seed).choice(len(jobs), size=n, replace=False))
    return jobs.iloc[picked].reset_index(drop=True)
