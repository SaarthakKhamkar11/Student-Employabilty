"""Job record, mirroring config/schema.py JOB_SCHEMA field for field and in order.

Rows of data/processed/jobs.csv load directly into this model. No field is
blank in the current data, so all fields are required. Phase 5 (Job
Intelligence Agent) decides which fields uploaded JDs may omit.
"""

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    job_id: str
    source_job_id: str  # float-precision collisions in the source (PROJECT_AUDIT.md I-7)
    experience_required: str
    qualification_required: str
    salary_range: str
    location: str
    country: str
    work_type: str
    job_title: str
    role: str
    job_portal: str
    job_description: str
    skills: str
    responsibilities: str
    company_name: str
    gender_preference: str = Field(
        description="Protected attribute: never a matching feature (spec §11); fairness audit only."
    )

    @field_validator("*", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value
