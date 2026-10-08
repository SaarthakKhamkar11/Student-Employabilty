"""Canonical schema: the single source of truth for column names.

Used by the data pipeline (src/pipeline/) and mirrored field-for-field by the
backend Pydantic models (backend/app/schemas/student.py and job.py).
"""

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
