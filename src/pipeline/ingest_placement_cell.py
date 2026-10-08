"""Ingest the placement-cell Excel export (real, 120 rows)."""

import pandas as pd

from config.schema import STUDENT_SCHEMA

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
