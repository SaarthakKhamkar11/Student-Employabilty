"""Job Intelligence Agent tools (spec §5.3 pipeline, §20 tool list).

Shared with the Profile Agent, and deliberately not duplicated:
- detect_pii: the same detector and masking;
- normalize_skills: the same alias table, so student and job skills are
  normalised identically (needed for Phase 7 features);
- generate_embedding: the same configured embedding model, in one shared
  vector space.

Reused from src/parsing: parse_job_sections (built on the extended
_slice_section) and guess_degree().
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from app.core.config import Settings
from app.core.logging import get_logger, log_event
from app.db.job_repository import JobRepository
from app.db.profile_repository import EmbeddingRow
from app.jobs import dedup
from app.jobs.requirements import (
    HEURISTIC_VERSION, INFERRED_NOTE, classify_requirements, parse_degree, parse_experience,
)
from app.jobs.skills_split import split_skill_phrases
from app.jobs.visibility import is_visible_to
from app.profile import pii as pii_detector
from app.profile.text_extraction import ResumeRejected, extract_text
from app.schemas.agent_state import NOT_DETECTED, NOT_YET_EVALUATED, SKILL_NOT_DETECTED
from app.schemas.job import DuplicateCheck, JobPosting, JobSkill, JobVisibility, SkillSplitInfo
from app.schemas.student import NormalizedSkill, PIIReport
from app.tools.profile_tools import GenerateEmbeddingOut
from app.tools.registry import ToolInputRejected, ToolRegistry
from src.parsing.section_parser import parse_job_sections

logger = get_logger("tools.job")
EXTRACTOR_VERSION = "job-v1"
RECORD_TEXT_FIELDS = ("job_description", "responsibilities", "skills")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractJobTextIn(_Model):
    text: str | None = None
    content_b64: str | None = None
    filename: str | None = None


class ExtractJobTextOut(_Model):
    text: str
    char_count: int


class LoadJobIn(_Model):
    job_id: str
    required: bool


class LoadJobOut(_Model):
    found: bool
    record: dict[str, Any] | None


class ParseJobIn(_Model):
    text: str | None = None  # free-text JD (already PII-masked)
    record: dict[str, Any] | None = None  # stored JOB_SCHEMA row (masked free-text fields)


class JobSections(_Model):
    title: str | None
    description: str | None
    responsibilities: list[str]
    skills_text: str | None
    required_text: str | None
    preferred_text: str | None
    qualification_text: str | None
    experience_text: str | None
    headers_found: list[str]


class ParseJobOut(_Model):
    sections: JobSections


class ExtractJobSkillsIn(_Model):
    sections: JobSections


class OriginSkill(_Model):
    phrase: str
    origin: Literal["required", "preferred"] | None  # the JD section it came from, if explicit


class ExtractJobSkillsOut(_Model):
    skills: list[OriginSkill]


class ClassifyIn(_Model):
    skills: list[dict[str, Any]]  # NormalizedSkill fields + origin
    description: str
    responsibilities: str


class ClassifyOut(_Model):
    skills: list[JobSkill]
    split: SkillSplitInfo


class RequirementsIn(_Model):
    qualification_text: str | None
    experience_text: str | None


class RequirementsOut(_Model):
    min_degree: str | None
    min_degree_level: int | None
    min_experience_years: float | None
    max_experience_years: float | None


class DuplicatesIn(_Model):
    job: dict[str, Any]  # job_id (if stored), company_name, job_title, job_description, responsibilities, skills


class AssembleJobIn(_Model):
    job_id: str
    source: Literal["stored_record", "job_description_text"]
    record: dict[str, Any] | None = None
    sections: JobSections
    skills: list[JobSkill] | Literal["Skill not detected", "Not yet evaluated"]
    split: SkillSplitInfo | Literal["Not yet evaluated"]
    requirements: RequirementsOut | Literal["Not yet evaluated"]
    duplicate_check: DuplicateCheck | Literal["Not yet evaluated"]
    pii: PIIReport | Literal["Not yet evaluated"]


class ValidateJobIn(_Model):
    posting: dict[str, Any]


class ValidateJobOut(_Model):
    valid: bool
    missing_fields: list[str]
    problems: list[str]


class SaveJobIn(_Model):
    posting: dict[str, Any]
    embedding: GenerateEmbeddingOut | None = None
    is_new: bool


class SaveJobOut(_Model):
    saved: bool
    job_id: str
    rows: dict[str, int]
    embeddings_stored: int


class VisibilityIn(_Model):
    gender_preference: str | None
    student_gender: str | None


class VisibilityOut(_Model):
    visible: bool
    filter_enabled: bool


def missing_fields(posting: JobPosting) -> list[str]:
    return [k for k, v in posting.model_dump(exclude={"visibility"}).items()
            if isinstance(v, str) and v in (NOT_DETECTED, NOT_YET_EVALUATED, SKILL_NOT_DETECTED)]


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def register_job_tools(registry: ToolRegistry, *, repo: JobRepository | None, settings: Settings) -> None:
    def _repo() -> JobRepository:
        if repo is None:
            raise RuntimeError("no database configured (DATABASE_URL)")
        return repo

    @registry.register("extract_job_text", input_schema=ExtractJobTextIn, output_schema=ExtractJobTextOut)
    def tool_extract_job_text(text=None, content_b64=None, filename=None):
        """Job description text from raw text or a .txt/.pdf/.docx file; rejects unusable input."""
        try:
            extracted, _ = extract_text(text, content_b64, filename)
        except ResumeRejected as exc:  # same checks as resumes; the message names the right document
            raise ToolInputRejected("extract_job_text",
                                    "job description rejected: " + str(exc).replace("resume", "job description")) from exc
        return {"text": extracted, "char_count": len(extracted)}

    @registry.register("load_job_record", input_schema=LoadJobIn, output_schema=LoadJobOut)
    def tool_load_job_record(job_id, required):
        """Fetch a stored job (jobs table). Unknown and required is a deterministic rejection."""
        record = _repo().get_job(job_id)
        if record is None and required:
            raise ToolInputRejected("load_job_record", f"unknown job_id {job_id}")
        return {"found": record is not None, "record": record}

    @registry.register("parse_job_description", input_schema=ParseJobIn, output_schema=ParseJobOut)
    def tool_parse_job_description(text=None, record=None):
        """Requirement extraction: JD headers via src/parsing parse_job_sections, or a stored row's columns."""
        if record is not None:
            duties = [s.strip() for s in (record.get("responsibilities") or "").replace("\n", " ").split(". ")
                      if s.strip()]
            return {"sections": JobSections(
                title=_text(record.get("job_title")), description=_text(record.get("job_description")),
                responsibilities=[d if d.endswith(".") else d + "." for d in duties],
                skills_text=_text(record.get("skills")), required_text=None, preferred_text=None,
                qualification_text=_text(record.get("qualification_required")),
                experience_text=_text(record.get("experience_required")), headers_found=[])}
        found = parse_job_sections(text or "")
        headers = [k for k in found if k != "title"]
        body = found.get("description") or (text if not headers else None)  # no headers: the whole text is the body
        duties = [d.strip(" -*•\t") for d in (found.get("responsibilities") or "").split("\n") if d.strip(" -*•\t")]
        return {"sections": JobSections(
            title=found.get("title") or None, description=_text(body), responsibilities=duties,
            skills_text=_text(found.get("skills")), required_text=_text(found.get("required")),
            preferred_text=_text(found.get("preferred")),
            qualification_text=_text(found.get("education")) or _text(found.get("required")),
            experience_text=_text(found.get("experience")) or _text(found.get("required")), headers_found=headers)}

    @registry.register("extract_job_skills", input_schema=ExtractJobSkillsIn, output_schema=ExtractJobSkillsOut)
    def tool_extract_job_skills(sections):
        """Skill phrases from the skills field/section and, for free text, the required/preferred sections."""
        s = JobSections.model_validate(sections)
        out, seen = [], set()
        for text, origin in ((s.required_text, "required"), (s.preferred_text, "preferred"), (s.skills_text, None)):
            for phrase in split_skill_phrases(text or ""):
                if phrase.lower() not in seen:
                    seen.add(phrase.lower())
                    out.append({"phrase": phrase, "origin": origin})
        return {"skills": out}

    @registry.register("classify_skill_requirements", input_schema=ClassifyIn, output_schema=ClassifyOut)
    def tool_classify_skill_requirements(skills, description, responsibilities):
        """INFERRED required/preferred split (docs/adr/0008); every skill records the rule that decided."""
        classified = classify_requirements(skills, description, responsibilities)
        return {"skills": classified,
                "split": SkillSplitInfo(method=HEURISTIC_VERSION, note=INFERRED_NOTE)}

    @registry.register("extract_requirements", input_schema=RequirementsIn, output_schema=RequirementsOut)
    def tool_extract_requirements(qualification_text, experience_text):
        """Minimum degree (via guess_degree) and years of experience as numbers."""
        degree, level = parse_degree(qualification_text or "")
        low, high = parse_experience(experience_text or "")
        return {"min_degree": degree, "min_degree_level": level, "min_experience_years": low,
                "max_experience_years": high}

    @registry.register("check_duplicates", input_schema=DuplicatesIn, output_schema=DuplicateCheck)
    def tool_check_duplicates(job):
        """Near-duplicate check against stored jobs before storing (docs/adr/0010)."""
        if not dedup.norm(job.get("job_title")):
            return {"detector_version": dedup.DETECTOR_VERSION, "near_duplicate_of": None, "similarity": None}
        # no company: compared with stored jobs that have no company either (v2)
        match = dedup.match_existing(job, _repo().candidates(job.get("company_name") or None, job["job_title"]))
        if match:
            log_event(logger, "near_duplicate_flagged", job_id=job.get("job_id"), **match)
        return {"detector_version": dedup.DETECTOR_VERSION, **(match or {"near_duplicate_of": None})}

    @registry.register("extract_job_posting", input_schema=AssembleJobIn, output_schema=JobPosting)
    def tool_extract_job_posting(job_id, source, sections, skills, split, requirements, duplicate_check, pii,
                                 record=None):
        """Assemble the structured JobPosting. Missing values become sentinels, never defaults."""
        record, s = record or {}, JobSections.model_validate(sections)

        def field(name: str, fallback: str | None = None, extracted: bool = False) -> Any:
            """A stored row's blank is "Not detected". For free text, only extracted fields can be "Not
            detected"; fields no extractor looks for (company, location ...) are "Not yet evaluated"."""
            value = _text(record.get(name)) or fallback
            return value if value else (NOT_DETECTED if record or extracted else NOT_YET_EVALUATED)

        def req(name: str) -> Any:
            if isinstance(requirements, str):
                return NOT_YET_EVALUATED
            value = requirements[name] if isinstance(requirements, dict) else getattr(requirements, name)
            return value if value is not None else NOT_DETECTED

        job_skills = skills if isinstance(skills, str) else [JobSkill.model_validate(x) for x in skills]

        def names(kind: str) -> Any:
            if isinstance(job_skills, str):
                return job_skills
            return [x.canonical for x in job_skills if x.requirement == kind] or SKILL_NOT_DETECTED

        posting = JobPosting(
            job_id=job_id, source=source,
            job_title=field("job_title", s.title, extracted=True), role=field("role"),
            company_name=field("company_name"),
            location=field("location"), country=field("country"), work_type=field("work_type"),
            salary_range=field("salary_range"), job_portal=field("job_portal"),
            job_description=s.description or NOT_DETECTED,
            qualification_required=field("qualification_required", s.qualification_text, extracted=True),
            experience_required=field("experience_required", s.experience_text, extracted=True),
            responsibilities=s.responsibilities or NOT_DETECTED,
            min_degree=req("min_degree"), min_degree_level=req("min_degree_level"),
            min_experience_years=req("min_experience_years"), max_experience_years=req("max_experience_years"),
            skills=job_skills, required_skills=names("required"), preferred_skills=names("preferred"),
            skill_split=split, visibility=JobVisibility(gender_preference=field("gender_preference")),
            duplicate_check=duplicate_check, pii=pii, missing_fields=[], extractor_version=EXTRACTOR_VERSION,
        )
        return posting.model_copy(update={"missing_fields": missing_fields(posting)})

    @registry.register("validate_job_posting", input_schema=ValidateJobIn, output_schema=ValidateJobOut)
    def tool_validate_job_posting(posting):
        """Schema check, sentinel list, range checks, INFERRED marker, and refusal if PII remains."""
        p = JobPosting.model_validate(posting)
        problems = []
        if isinstance(p.min_experience_years, float) and isinstance(p.max_experience_years, float) \
                and p.min_experience_years > p.max_experience_years:
            problems.append("min_experience_years > max_experience_years")
        if isinstance(p.skills, list) and (isinstance(p.skill_split, str) or not p.skill_split.inferred):
            problems.append("required/preferred split must be marked as inferred")
        text = " ".join(str(v) for v in p.model_dump(exclude={"pii", "missing_fields", "visibility"}).values())
        leaked = {f.type for f in pii_detector.detect(text, field="job") if f.type in ("email", "phone", "dob")}
        if leaked:
            problems.append(f"PII present in structured posting: {sorted(leaked)}")
        return {"valid": not problems, "missing_fields": missing_fields(p), "problems": problems}

    @registry.register("save_job", input_schema=SaveJobIn, output_schema=SaveJobOut)
    def tool_save_job(posting, is_new, embedding=None):
        """Write the posting, skills, responsibilities and embeddings (model name and version) in one transaction."""
        p = JobPosting.model_validate(posting)
        embedding = GenerateEmbeddingOut.model_validate(embedding) if embedding is not None else None
        rows = [EmbeddingRow(section=k, model_name=embedding.model_name, model_version=embedding.model_version,
                             dimension=embedding.dimension, values=v.values, indices=v.indices,
                             interim=embedding.interim)
                for k, v in (embedding.vectors.items() if embedding else [])]
        result = _repo().save_posting(p, rows, is_new)
        log_event(logger, "job_saved", job_id=p.job_id, rows=result.rows, embeddings_stored=result.embeddings_stored,
                  level=logging.INFO)
        return {"saved": True, "job_id": p.job_id, "rows": result.rows, "embeddings_stored": result.embeddings_stored}

    @registry.register("check_job_visibility", input_schema=VisibilityIn, output_schema=VisibilityOut)
    def tool_check_job_visibility(gender_preference, student_gender):
        """Visibility filter only (docs/adr/0009). The Matching Agent calls this BEFORE scoring, never as a feature."""
        enabled = settings.job_gender_visibility_filter
        return {"visible": is_visible_to(gender_preference, student_gender, enabled=enabled),
                "filter_enabled": enabled}
