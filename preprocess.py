#PHASE1

import csv
import os
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (edit these)
# ---------------------------------------------------------------------------
STUDENT_XLSX = "Student Data.xlsx"
RESUME_CSV = "UpdatedResumeDataSet.csv"
JOBS_CSV = "job_descriptions.csv"
OUT_DIR = "data/processed"
RESULTS_DIR = "results"
JOB_SAMPLE_N = 5000
SEED = 42

# ---------------------------------------------------------------------------
# 1. Canonical schema (single source of truth for column names)
# ---------------------------------------------------------------------------
JOB_SCHEMA = [
    "job_id", "source_job_id", "experience_required", "qualification_required",
    "salary_range", "location", "country", "work_type", "job_title", "role",
    "job_portal", "job_description", "skills", "responsibilities",
    "company_name", "gender_preference",
]
STUDENT_SCHEMA = [
    "student_id", "degree", "branch", "graduation_year", "gender",
    "avg_cgpa_sem5", "achievements", "career_interests", "domain",
    "internship_raw", "technical_projects", "skills_raw", "certifications",
    "country", "source_dataset", "is_synthetic", "synthetic_fields",
]

# ---------------------------------------------------------------------------
# 2. Resume free-text parser (heuristic — see notes at bottom)
# ---------------------------------------------------------------------------
SECTION_MARKERS = ["education details", "skill details", "skills",
                    "company details", "experience", "projects",
                    "certifications", "declaration"]
BULLET_NOISE = re.compile(r"[â¢\x80-\xff\uf0b7\u2022]+")
DEGREE_RE = re.compile(
    r"\b(B\.?\s?E|B\.?\s?Tech|M\.?\s?Tech|B\.?\s?Sc|M\.?\s?Sc|MBA|MCA|BCA|Diploma|Ph\.?D)\b",
    re.IGNORECASE,
)


def _clean(text):
    text = BULLET_NOISE.sub(" ", text)
    return re.sub(r"[\r\n]+", "\n", text).strip()


def _slice_section(text_lower, text_orig, marker):
    idx = text_lower.find(marker)
    if idx == -1:
        return ""
    start, end = idx + len(marker), len(text_orig)
    for other in SECTION_MARKERS:
        if other == marker:
            continue
        pos = text_lower.find(other, start)
        if pos != -1:
            end = min(end, pos)
    return text_orig[start:end].strip()


def extract_skills(text):
    tl = text.lower()
    chunk = _slice_section(tl, text, "skill details") or _slice_section(tl, text, "skills")
    if not chunk:
        return []
    out = []
    for p in re.split(r"[,\n]", chunk):
        p = re.sub(r"^[A-Za-z /]+:\s*", "", p).strip(" .-*")
        if p and len(p) < 60 and not p.lower().startswith("company"):
            out.append(p)
    return out[:40]


def parse_resume(raw_text):
    text = _clean(raw_text)
    tl = text.lower()
    education = re.sub(r"\s+", " ", _slice_section(tl, text, "education details")).strip()[:300]
    experience = re.sub(r"\s+", " ", (_slice_section(tl, text, "company details")
                                       or _slice_section(tl, text, "experience"))).strip()[:500]
    return {"skills": extract_skills(text), "education_raw": education, "experience_raw": experience}


def guess_degree(education_raw):
    m = DEGREE_RE.search(education_raw or "")
    return m.group(1).upper().replace(" ", "") if m else ""


# ---------------------------------------------------------------------------
# 3. Ingestion: placement-cell (real, 120 rows)
# ---------------------------------------------------------------------------
STUDENT_COL_MAP = {
    "Degree": "degree", "Branch": "branch", "Graduation Year": "graduation_year",
    "Gender": "gender", "Average CGPA until Sem 5": "avg_cgpa_sem5",
    "Achievements / Awards (If Any)": "achievements", "Career Interests": "career_interests",
    "Domain": "domain",
    "Internship - Company's Name and Role and Duration": "internship_raw",
    "Technical Projects / Open Source Contribution (ANY 1)": "technical_projects",
    "Skills": "skills_raw", "Certifications (ANY 3)": "certifications",
}


def ingest_placement_cell(path, start_id=1):
    df = pd.read_excel(path).rename(columns=STUDENT_COL_MAP)
    df["gender"] = df["gender"].astype(str).str.lower()
    df["branch"] = df["branch"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["country"] = "India"           # form has no location field, all India-based (confirmed)
    df["source_dataset"] = "placement_cell"
    df["is_synthetic"] = False
    df["synthetic_fields"] = ""
    df = df.reset_index(drop=True)
    df.insert(0, "student_id", [f"S{start_id + i:05d}" for i in range(len(df))])
    return df[STUDENT_SCHEMA]


# ---------------------------------------------------------------------------
# 4. Ingestion: Kaggle multi-branch resumes (real, dedup 962 -> 166)
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    "Data Science": ("Computer Science and Engineering (Data Science)", "Data Science"),
    "Java Developer": ("Information Technology", "Software Development"),
    "Python Developer": ("Information Technology", "Software Development"),
    "DotNet Developer": ("Information Technology", "Software Development"),
    "Web Designing": ("Information Technology", "Web Development"),
    "Testing": ("Information Technology", "QA/Testing"),
    "Automation Testing": ("Information Technology", "QA/Testing"),
    "DevOps Engineer": ("Information Technology", "Cloud/DevOps"),
    "Hadoop": ("Information Technology", "Big Data"),
    "ETL Developer": ("Information Technology", "Data Engineering"),
    "Database": ("Information Technology", "Database Administration"),
    "Blockchain": ("Information Technology", "Blockchain"),
    "SAP Developer": ("Information Technology", "Enterprise Systems"),
    "Network Security Engineer": ("Information Technology", "Networking/Security"),
    "Mechanical Engineer": ("Mechanical Engineering", "Core Mechanical"),
    "Civil Engineer": ("Civil Engineering", "Core Civil"),
    "Electrical Engineering": ("Electrical Engineering", "Core Electrical"),
    "Business Analyst": ("Non-Engineering / Management", "Business Analysis"),
    "Operations Manager": ("Non-Engineering / Management", "Operations"),
    "PMO": ("Non-Engineering / Management", "Project Management"),
    "HR": ("Non-Engineering / Management", "Human Resources"),
    "Sales": ("Non-Engineering / Management", "Sales"),
    "Advocate": ("Non-Engineering / Management", "Legal"),
    "Arts": ("Non-Engineering / Management", "Arts"),
    "Health and fitness": ("Non-Engineering / Management", "Health & Fitness"),
}


def ingest_kaggle_resumes(path, start_id=200):
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        branch, domain = CATEGORY_MAP.get(r["Category"], (r["Category"], r["Category"]))
        parsed = parse_resume(r["Resume"])
        rows.append({
            "student_id": f"S{start_id + i:05d}",
            "degree": guess_degree(parsed["education_raw"]),
            "branch": branch,
            "graduation_year": "", "gender": "", "avg_cgpa_sem5": "",  # not in source
            "achievements": "",                                        # not in source
            "career_interests": r["Category"],
            "domain": domain,
            "internship_raw": parsed["experience_raw"],
            "technical_projects": "",                                  # not cleanly separable
            "skills_raw": "; ".join(parsed["skills"]),
            "certifications": "", "country": "",                       # not in source
            "source_dataset": "kaggle_resume_dataset",
            "is_synthetic": False,        # resume itself is real, unedited
            "synthetic_fields": "",       # nothing fabricated; blanks left blank
        })
    return pd.DataFrame(rows)[STUDENT_SCHEMA]


# ---------------------------------------------------------------------------
# 5. Ingestion: real job postings (1,048,575 rows -> stratified sample)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 6. Data-quality report (every number here is measured, not estimated)
# ---------------------------------------------------------------------------
def write_quality_report(jobs_df, students_df, out_path):
    def missing(df, cols):
        return {c: int(df[c].astype(str).str.strip().eq("").sum()) for c in cols}

    report = f"""# Data Quality Report
## Jobs (sampled from real job_descriptions.csv)
- Row count: {len(jobs_df)}
- Duplicate job_id: {jobs_df['job_id'].duplicated().sum()}
- Missing fields: {missing(jobs_df, JOB_SCHEMA)}
- gender_preference distribution: {jobs_df['gender_preference'].value_counts().to_dict()}

## Students (real, unified)
- Row count: {len(students_df)}
- Duplicate student_id: {students_df['student_id'].duplicated().sum()}
- Rows by source_dataset: {students_df['source_dataset'].value_counts().to_dict()}
- Rows by branch: {students_df['branch'].value_counts().to_dict()}
- Missing fields (blank, never fabricated): {missing(students_df, STUDENT_SCHEMA)}
"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    placement = ingest_placement_cell(STUDENT_XLSX)                       # 120 real rows
    kaggle = ingest_kaggle_resumes(RESUME_CSV)                            # 166 real rows
    students = pd.concat([placement, kaggle], ignore_index=True)

    jobs_sample, jobs_full = ingest_jobs(JOBS_CSV)                        # 5,000-row sample

    os.makedirs(OUT_DIR, exist_ok=True)
    students.to_csv(f"{OUT_DIR}/students_unified.csv", index=False)
    jobs_sample.to_csv(f"{OUT_DIR}/jobs.csv", index=False)
    print(f"students_unified.csv: {len(students)} rows | jobs.csv: {len(jobs_sample)} rows")

    write_quality_report(jobs_sample, students, f"{RESULTS_DIR}/data_quality_report.md")

# Known limitations (documented, not hidden):
# - resume_text_parser is heuristic regex section-splitting; fails to find
#   a skills section on ~21% of Kaggle resumes (inconsistent headers).
# - Kaggle resumes have no degree/gender/CGPA/graduation_year/country ->
#   left blank rather than guessed.
# - job_descriptions.csv Job Id collides due to float-precision loss in the
#   source file -> job_id is reassigned sequentially here; original kept
#   in source_job_id.