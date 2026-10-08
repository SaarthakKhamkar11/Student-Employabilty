"""Ingest job postings (1,048,575 rows -> stratified sample by country).

Provenance caveat: the source looks programmatically generated and
Excel-truncated (PROJECT_AUDIT.md, problem I-7).
"""

import pandas as pd

from config.schema import JOB_SCHEMA

JOB_SAMPLE_N = 5000
SEED = 42

JOB_COL_MAP = {
    "Job Id": "source_job_id", "Experience": "experience_required",
    "Qualifications": "qualification_required", "Salary Range": "salary_range",
    "location": "location", "Country": "country", "Work Type": "work_type",
    "Job Title": "job_title", "Role": "role", "Job Portal": "job_portal",
    "Job Description": "job_description", "skills": "skills",
    "Responsibilities": "responsibilities", "Company": "company_name",
    "Preference": "gender_preference",
}


def ingest_jobs(path, sample_n=JOB_SAMPLE_N, seed=SEED):
    df = pd.read_csv(path, usecols=list(JOB_COL_MAP.keys())).rename(columns=JOB_COL_MAP)
    df["gender_preference"] = df["gender_preference"].str.lower()
    df["source_job_id"] = df["source_job_id"].astype(str)
    df = df.drop_duplicates().reset_index(drop=True)   # true full-row dupes only
    df.insert(0, "job_id", [f"J{i+1:07d}" for i in range(len(df))])
    df = df[JOB_SCHEMA]

    if sample_n and sample_n < len(df):
        parts = [g.sample(n=min(max(1, round(sample_n * len(g) / len(df))), len(g)),
                           random_state=seed)
                 for _, g in df.groupby("country")]
        sample = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
        sample = sample.reset_index(drop=True).iloc[:sample_n]
    else:
        sample = df
    return sample, df
