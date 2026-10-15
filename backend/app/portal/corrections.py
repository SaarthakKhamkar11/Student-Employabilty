"""Human confirmation of parsed data (Phase 13): the student reviews the Phase 4 profile, the company the Phase 5
posting, before either is treated as final.

A correction goes through the same registered tools the agents use, so it is normalised, PII-checked, validated,
re-embedded with the configured model and saved exactly like parser output:

    student: normalize_skills -> validate_profile -> generate_embedding -> save_profile
    company: normalize_skills -> validate_job_posting -> generate_embedding -> save_job

What changed is recorded: corrected skills carry their origin (student: evidence "confirmed by the student";
job: requirement_basis "confirmed_by_company"), and every confirmation writes an audit_logs row.
"""

from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

from app.agents.job_intelligence_agent import job_section_texts
from app.agents.profile_agent import section_texts
from app.schemas.agent_state import NOT_DETECTED, SKILL_NOT_DETECTED
from app.tools.registry import ToolError, ToolRegistry

STUDENT_EVIDENCE = "confirmed by the student"
DEGREE_LEVELS = {"Diploma": 1, "Bachelor's": 2, "Master's": 3, "Doctorate": 4}


def _clean_list(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    out, seen = [], set()
    for v in values:
        v = " ".join(str(v).split())
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


class ProfileCorrection(BaseModel):
    """Only the fields sent are changed. A blank value means "not stated" (stored as the 'Not detected' sentinel)."""

    degree: str | None = Field(None, max_length=100)
    branch: str | None = Field(None, max_length=150)
    graduation_year: int | None = Field(None, ge=1990, le=2035)
    career_interests: str | None = Field(None, max_length=500)
    domain: str | None = Field(None, max_length=150)
    skills: list[str] | None = Field(None, max_length=150)  # the full confirmed skill list (raw names)
    projects: list[str] | None = Field(None, max_length=50)
    experience: list[str] | None = Field(None, max_length=50)
    certifications: list[str] | None = Field(None, max_length=50)

    _lists = field_validator("skills", "projects", "experience", "certifications")(_clean_list)


class JobCorrection(BaseModel):
    job_title: str | None = Field(None, max_length=200)
    role: str | None = Field(None, max_length=200)
    location: str | None = Field(None, max_length=200)
    country: str | None = Field(None, max_length=100)
    work_type: str | None = Field(None, max_length=50)
    company_name: str | None = Field(None, max_length=200)
    salary_range: str | None = Field(None, max_length=150)
    job_description: str | None = Field(None, max_length=20_000)
    responsibilities: list[str] | None = Field(None, max_length=60)
    required_skills: list[str] | None = Field(None, max_length=80)
    preferred_skills: list[str] | None = Field(None, max_length=80)
    min_degree: Literal["Diploma", "Bachelor's", "Master's", "Doctorate", ""] | None = None
    min_experience_years: float | None = Field(None, ge=0, le=40)
    max_experience_years: float | None = Field(None, ge=0, le=50)

    _lists = field_validator("responsibilities", "required_skills", "preferred_skills")(_clean_list)


async def _call(registry: ToolRegistry, tool: str, args: dict[str, Any]):
    try:
        return await registry.call(tool, args)
    except ToolError as exc:
        raise HTTPException(422, f"{tool}: {exc}") from None


def _text(v: str | None) -> str:
    v = " ".join((v or "").split())
    return v or NOT_DETECTED


async def correct_profile(registry: ToolRegistry, profile: dict[str, Any], c: ProfileCorrection) -> tuple[dict, list[str], dict]:
    """Apply a correction to a stored StudentProfile dict. Returns (profile, changed field names, saved info)."""
    p = dict(profile)
    changed: list[str] = []
    for field in ("degree", "branch", "career_interests", "domain"):
        value = getattr(c, field)
        if value is not None and _text(value) != p.get(field):
            p[field], changed = _text(value), changed + [field]
    if c.graduation_year is not None and c.graduation_year != p.get("graduation_year"):
        p["graduation_year"], changed = c.graduation_year, changed + ["graduation_year"]
    for field in ("projects", "experience", "certifications"):
        value = getattr(c, field)
        if value is not None and (value or NOT_DETECTED) != p.get(field):
            p[field], changed = (value or NOT_DETECTED), changed + [field]
    if c.skills is not None:
        current = p.get("skills") if isinstance(p.get("skills"), list) else []
        by_canonical = {s["canonical"].lower(): s for s in current}
        by_raw = {s["raw"].lower(): s for s in current}
        new_raw = [s for s in c.skills if s.lower() not in by_canonical and s.lower() not in by_raw]
        normalised = []
        if new_raw:
            out = await _call(registry, "normalize_skills", {"skills": new_raw})
            normalised = [dict(s.model_dump(), evidence=STUDENT_EVIDENCE) for s in out.skills]
        kept = [s for s in current if s["canonical"].lower() in {x.lower() for x in c.skills}
                or s["raw"].lower() in {x.lower() for x in c.skills}]
        merged, seen = [], set()
        for s in kept + normalised:
            if s["canonical"].lower() not in seen:
                seen.add(s["canonical"].lower())
                merged.append(s)
        if [s["canonical"] for s in merged] != [s["canonical"] for s in current]:
            p["skills"], changed = (merged or SKILL_NOT_DETECTED), changed + ["skills"]
    if not changed:
        return p, [], {}
    check = await _call(registry, "validate_profile", {"profile": p})
    if not check.valid:
        raise HTTPException(422, "The corrected profile was rejected: " + "; ".join(check.problems))
    p["missing_fields"] = check.missing_fields
    embedding = await _call(registry, "generate_embedding", {"section_texts": section_texts(p), "role": "query"})
    saved = await _call(registry, "save_profile", {"profile": p, "embedding": embedding.model_dump()})
    return p, changed, {"embedding_model": embedding.model_name, "embedding_version": embedding.model_version,
                        "embeddings_stored": saved.embeddings_stored}


def _job_skill(s: dict, requirement: str) -> dict:
    return {"raw": s["raw"], "canonical": s["canonical"], "category": s["category"], "method": s["method"],
            "evidence": s.get("evidence"), "requirement": requirement, "requirement_basis": "confirmed_by_company"}


async def correct_posting(registry: ToolRegistry, posting: dict[str, Any], c: JobCorrection) -> tuple[dict, list[str], dict]:
    p = dict(posting)
    changed: list[str] = []
    for field in ("job_title", "role", "location", "country", "work_type", "company_name", "salary_range",
                  "job_description"):
        value = getattr(c, field)
        if value is not None and _text(value) != p.get(field):
            p[field], changed = _text(value), changed + [field]
    if c.responsibilities is not None and (c.responsibilities or NOT_DETECTED) != p.get("responsibilities"):
        p["responsibilities"], changed = (c.responsibilities or NOT_DETECTED), changed + ["responsibilities"]
    if c.min_degree is not None:
        degree = c.min_degree or NOT_DETECTED
        if degree != p.get("min_degree"):
            p["min_degree"] = degree
            p["min_degree_level"] = DEGREE_LEVELS.get(c.min_degree, NOT_DETECTED)
            changed.append("min_degree")
    for field in ("min_experience_years", "max_experience_years"):
        value = getattr(c, field)
        if value is not None and value != p.get(field):
            p[field], changed = value, changed + [field]
    if c.required_skills is not None or c.preferred_skills is not None:
        current = p.get("skills") if isinstance(p.get("skills"), list) else []
        req = c.required_skills if c.required_skills is not None else (
            p["required_skills"] if isinstance(p.get("required_skills"), list) else [])
        pref = c.preferred_skills if c.preferred_skills is not None else (
            p["preferred_skills"] if isinstance(p.get("preferred_skills"), list) else [])
        pref = [s for s in pref if s.lower() not in {r.lower() for r in req}]
        index = {s["canonical"].lower(): s for s in current}
        index.update({s["raw"].lower(): s for s in current})
        unknown = [s for s in req + pref if s.lower() not in index]
        if unknown:
            out = await _call(registry, "normalize_skills", {"skills": unknown})
            for raw, s in zip(unknown, out.skills):
                index[raw.lower()] = s.model_dump()
        skills, seen = [], set()
        for names, requirement in ((req, "required"), (pref, "preferred")):
            for name in names:
                s = index[name.lower()]
                if s["canonical"].lower() in seen:
                    continue
                seen.add(s["canonical"].lower())
                keep = s.get("requirement") == requirement and s.get("requirement_basis")
                skills.append(dict(s) if keep else _job_skill(s, requirement))
        before = [(s["canonical"], s["requirement"]) for s in current]
        if [(s["canonical"], s["requirement"]) for s in skills] != before:
            p["skills"] = skills or SKILL_NOT_DETECTED
            p["required_skills"] = [s["canonical"] for s in skills if s["requirement"] == "required"] or SKILL_NOT_DETECTED
            p["preferred_skills"] = [s["canonical"] for s in skills if s["requirement"] == "preferred"] or SKILL_NOT_DETECTED
            changed.append("skills")
    if not changed:
        return p, [], {}
    check = await _call(registry, "validate_job_posting", {"posting": p})
    if not check.valid:
        raise HTTPException(422, "The corrected job was rejected: " + "; ".join(check.problems))
    if hasattr(check, "missing_fields"):
        p["missing_fields"] = check.missing_fields
    embedding = await _call(registry, "generate_embedding", {"section_texts": job_section_texts(p), "role": "passage"})
    saved = await _call(registry, "save_job", {"posting": p, "embedding": embedding.model_dump(), "is_new": False})
    return p, changed, {"embedding_model": embedding.model_name, "embedding_version": embedding.model_version,
                        "embeddings_stored": saved.embeddings_stored}
