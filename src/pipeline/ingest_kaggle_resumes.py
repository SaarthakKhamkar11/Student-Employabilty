"""Ingest the Kaggle multi-branch resume dataset (real, dedup 962 -> 166)."""

import pandas as pd

from config.schema import STUDENT_SCHEMA
from src.parsing.resume_text_parser import guess_degree, parse_resume

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
