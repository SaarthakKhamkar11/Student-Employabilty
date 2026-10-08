"""Student record, mirroring config/schema.py STUDENT_SCHEMA field for field and in order.

Rows of data/processed/students_unified.csv load directly into this model, and
blank cells become None. A field that is blank in the source stays None: it is
never filled with a guess. results/data_quality_report.md has the measured
blank counts per field.
"""

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StudentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    student_id: str
    degree: str | None = None  # not detected in 65 Kaggle resumes
    branch: str
    graduation_year: int | None = None  # absent from the Kaggle source
    gender: str | None = Field(
        None, description="Protected attribute: never a matching feature (spec §11); fairness audit only."
    )
    avg_cgpa_sem5: float | None = Field(None, ge=0, le=10)  # absent from the Kaggle source
    achievements: str | None = None
    career_interests: str
    domain: str
    internship_raw: str | None = None
    technical_projects: str | None = None
    skills_raw: str | None = None
    certifications: str | None = None
    country: str | None = None
    source_dataset: str
    is_synthetic: bool
    synthetic_fields: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value
