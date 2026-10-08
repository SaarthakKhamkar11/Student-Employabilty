"""Job record, mirroring config/schema.py JOB_SCHEMA field for field and in order.

Rows of data/processed/jobs.csv load directly into this model. No field is
blank in the current data, so all fields are required. Phase 5 (Job
Intelligence Agent) decides which fields uploaded JDs may omit.
"""

import math
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.student import Missing, PIIReport, SkillsMissing


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


# --------------------------------------------------------------------------
# Structured posting built by the Job Intelligence Agent (Phase 5)
# --------------------------------------------------------------------------
RequirementBasis = Literal[
    "explicit_required_section", "explicit_preferred_section", "explicit_required_language",
    "explicit_preferred_language", "named_in_role_text", "not_named_in_role_text", "first_listed_fallback",
]


class JobSkill(BaseModel):
    """A normalised job skill (same normaliser as students) plus its INFERRED requirement level."""

    model_config = ConfigDict(extra="forbid")

    raw: str
    canonical: str
    category: Literal["technical", "soft", "unclassified"]
    method: Literal["exact", "alias", "unmapped"]
    evidence: str | None = None
    requirement: Literal["required", "preferred"]
    requirement_basis: RequirementBasis  # which heuristic rule decided (docs/adr/0008)


class SkillSplitInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inferred: Literal[True] = True  # never ground truth
    method: str
    note: str


class JobVisibility(BaseModel):
    """Visibility filter inputs only (docs/adr/0009). Nothing here may enter a scoring feature."""

    model_config = ConfigDict(extra="forbid")

    gender_preference: str | Missing


class DuplicateCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector_version: str
    near_duplicate_of: str | None  # canonical job_id if this posting is a near-duplicate
    similarity: float | None = None


class JobPosting(BaseModel):
    """A job's structured posting. Raw fields keep JOB_SCHEMA names; derived fields are new and marked."""

    model_config = ConfigDict(extra="forbid")

    # Fields the matching stage may use. gender_preference is deliberately absent (spec §11, ADR 0009).
    SCORING_FIELDS: ClassVar[tuple[str, ...]] = (
        "job_title", "role", "job_description", "responsibilities", "skills", "required_skills",
        "preferred_skills", "min_degree", "min_degree_level", "min_experience_years", "max_experience_years",
        "work_type", "location", "country",
    )

    job_id: str
    source: Literal["stored_record", "job_description_text"]
    # mapped through from JOB_SCHEMA
    job_title: str | Missing
    role: str | Missing
    company_name: str | Missing
    location: str | Missing
    country: str | Missing
    work_type: str | Missing
    salary_range: str | Missing
    job_portal: str | Missing
    job_description: str | Missing
    qualification_required: str | Missing
    experience_required: str | Missing
    # derived (Phase 5)
    responsibilities: list[str] | Missing
    min_degree: str | Missing
    min_degree_level: int | Missing  # 1 diploma, 2 bachelor, 3 master, 4 doctorate
    min_experience_years: float | Missing
    max_experience_years: float | Missing
    skills: list[JobSkill] | SkillsMissing
    required_skills: list[str] | SkillsMissing
    preferred_skills: list[str] | SkillsMissing
    skill_split: SkillSplitInfo | Literal["Not yet evaluated"]
    visibility: JobVisibility
    duplicate_check: DuplicateCheck | Literal["Not yet evaluated"]
    pii: PIIReport | Literal["Not yet evaluated"]
    missing_fields: list[str]
    extractor_version: str
