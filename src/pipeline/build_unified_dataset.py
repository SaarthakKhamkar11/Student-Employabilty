"""Build data/processed/{students_unified,jobs}.csv from the raw sources.

Run from the repo root:  python preprocess.py
                     or: python -m src.pipeline.build_unified_dataset
Raw files go in data/raw/ (gitignored), or point RAW_DATA_DIR elsewhere.
"""

import os
import sys

import pandas as pd

from src.pipeline.ingest_kaggle_resumes import ingest_kaggle_resumes
from src.pipeline.ingest_placement_cell import ingest_placement_cell
from src.pipeline.ingest_real_jobs import ingest_jobs
from src.pipeline.validate_and_report import write_quality_report

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.environ.get("RAW_DATA_DIR", os.path.join(ROOT, "data", "raw"))
STUDENT_XLSX = os.path.join(RAW_DIR, "Student Data.xlsx")
RESUME_CSV = os.path.join(RAW_DIR, "UpdatedResumeDataSet.csv")
JOBS_CSV = os.path.join(RAW_DIR, "job_descriptions.csv")
OUT_DIR = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(ROOT, "results")


def main():
    missing_inputs = [p for p in (STUDENT_XLSX, RESUME_CSV, JOBS_CSV) if not os.path.exists(p)]
    if missing_inputs:
        sys.exit("Raw input file(s) not found:\n  " + "\n  ".join(missing_inputs)
                 + "\nPut them in data/raw/ or set RAW_DATA_DIR (see PROGRESS.md).")

    placement = ingest_placement_cell(STUDENT_XLSX)                       # 120 real rows
    kaggle = ingest_kaggle_resumes(RESUME_CSV)                            # 166 real rows
    students = pd.concat([placement, kaggle], ignore_index=True)

    jobs_sample, jobs_full = ingest_jobs(JOBS_CSV)                        # 5,000-row sample

    os.makedirs(OUT_DIR, exist_ok=True)
    students.to_csv(f"{OUT_DIR}/students_unified.csv", index=False)
    jobs_sample.to_csv(f"{OUT_DIR}/jobs.csv", index=False)
    print(f"students_unified.csv: {len(students)} rows | jobs.csv: {len(jobs_sample)} rows")

    write_quality_report(jobs_sample, students, f"{RESULTS_DIR}/data_quality_report.md")


if __name__ == "__main__":
    main()

# Known limitations (documented, not hidden):
# - resume_text_parser is heuristic regex section-splitting; fails to find
#   a skills section on ~21% of Kaggle resumes (inconsistent headers).
# - Kaggle resumes have no degree/gender/CGPA/graduation_year/country ->
#   left blank rather than guessed.
# - job_descriptions.csv Job Id collides due to float-precision loss in the
#   source file -> job_id is reassigned sequentially here; original kept
#   in source_job_id.
