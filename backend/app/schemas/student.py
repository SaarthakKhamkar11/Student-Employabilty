"""Student record, mirroring config/schema.py STUDENT_SCHEMA field for field and in order.

Rows of data/processed/students_unified.csv load directly into this model, and
blank cells become None. A field that is blank in the source stays None: it is
never filled with a guess. results/data_quality_report.md has the measured
blank counts per field.
"""

import math
from typing import Any, Literal

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


# --------------------------------------------------------------------------
# Structured profile built by the Profile Agent (Phase 4)
# --------------------------------------------------------------------------
Missing = Literal["Not detected", "Not yet evaluated"]
SkillsMissing = Literal["Skill not detected", "Not yet evaluated"]


class NormalizedSkill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str  # as written in the source
    canonical: str  # from the alias table; for method="unmapped", the cleaned text itself
    category: Literal["technical", "soft", "unclassified"]
    method: Literal["exact", "alias", "unmapped"]
    evidence: str | None = None  # stated experience, e.g. "24 months"


class PIIReport(BaseModel):
    """What the PII detector flagged: types and counts only, never the values."""

    model_config = ConfigDict(extra="forbid")

    detector_version: str
    detected: bool
    counts: dict[str, int]


class StudentProfile(BaseModel):
    """A student's structured profile. Every field holds evidence or an explicit sentinel, never a guess.

    PII is excluded (spec §38), and so is gender (a protected attribute, spec §11).
    """

    model_config = ConfigDict(extra="forbid")

    student_id: str
    source: Literal["stored_record", "resume", "stored_record+resume"]
    degree: str | Missing
    branch: str | Missing
    graduation_year: int | Missing
    avg_cgpa_sem5: float | Missing
    education: str | Missing
    skills: list[NormalizedSkill] | SkillsMissing
    projects: list[str] | Missing
    experience: list[str] | Missing
    certifications: list[str] | Missing
    achievements: str | Missing
    career_interests: str | Missing
    domain: str | Missing
    pii: PIIReport | Literal["Not yet evaluated"]
    missing_fields: list[str]
    extractor_version: str
