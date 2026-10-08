"""Data-quality report (every number here is measured, not estimated)."""

import os

from config.schema import JOB_SCHEMA, STUDENT_SCHEMA

def write_quality_report(jobs_df, students_df, out_path):
    def missing(df, cols):
        # NaN (e.g. empty Excel cells) must count as missing, not as the string "nan"
        return {c: int((df[c].isna() | df[c].astype(str).str.strip().eq("")).sum()) for c in cols}

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
