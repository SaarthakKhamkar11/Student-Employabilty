"""Profile Agent tools (spec §5.2 pipeline, §20 tool list). Each is deterministic and schema-validated.

Parsing reuses the existing src/parsing code (wrapped by app.profile.sections
and extract_skills); nothing is reimplemented here. Tools raise
ToolInputRejected for input that can never succeed (an empty resume, an
unknown student), so the agent does not waste retries on it.
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from app.core.logging import get_logger, log_event, register_sensitive_values
from app.db.profile_repository import EmbeddingRow, ProfileRepository
from app.embeddings.provider import EmbeddingProvider, Vector
from app.profile import pii as pii_detector
from app.profile.sections import NOT_EXTRACTED_FOR_SOURCE, record_sections, resume_sections, resume_skill_items
from app.profile.skills import alias_table_version, normalize_skills
from app.profile.text_extraction import ResumeRejected, extract_text
from app.schemas.agent_state import NOT_DETECTED, NOT_YET_EVALUATED
from app.schemas.student import NormalizedSkill, PIIReport, SkillsMissing, StudentProfile
from app.tools.registry import ToolInputRejected, ToolRegistry
from src.parsing.resume_text_parser import extract_skills
from src.parsing.section_parser import _clean

logger = get_logger("tools.profile")

# v2 (2026-10-15): PDF word gaps by font size, heading-based resume sections and skills (app/profile/sections.py)
EXTRACTOR_VERSION = "profile-v2"
# Free-text fields of a stored record that the PII detector scans.
RECORD_TEXT_FIELDS = ("degree", "branch", "achievements", "career_interests", "domain", "internship_raw",
                      "technical_projects", "skills_raw", "certifications")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractTextIn(_Model):
    text: str | None = None
    content_b64: str | None = None
    filename: str | None = None


class ExtractTextOut(_Model):
    text: str
    source_format: Literal["text", "txt", "pdf", "docx"]
    char_count: int


class DetectPiiIn(_Model):
    fields: dict[str, str]


class DetectPiiOut(_Model):
    masked_fields: dict[str, str]
    report: PIIReport
    findings: list[pii_detector.PIIFinding]  # type and offsets only, no values


class LoadStudentIn(_Model):
    student_id: str
    required: bool


class LoadStudentOut(_Model):
    found: bool
    record: dict[str, Any] | None
    # A resume upload's raw text is never stored (PII), so its stored structured profile is the only source
    # for it: set for uploads, None otherwise.
    stored_profile: dict[str, Any] | None = None


class TextIn(_Model):
    text: str


class SectionsOut(_Model):
    sections: dict[str, Any]


class RecordIn(_Model):
    record: dict[str, Any]


class ExtractSkillsIn(_Model):
    text: str
    # "resume": an uploaded resume (heading-aware extractor); "record": a stored record's skills_raw
    source: Literal["resume", "record"] = "record"


class ExtractSkillsOut(_Model):
    skills: list[str]


class NormalizeSkillsIn(_Model):
    skills: list[str]


class NormalizeSkillsOut(_Model):
    skills: list[NormalizedSkill]
    alias_table_version: str
    unmapped: int


class ExtractProfileIn(_Model):
    student_id: str
    source: Literal["stored_record", "resume", "stored_record+resume"]
    sections: dict[str, Any]
    skills: list[NormalizedSkill] | SkillsMissing
    record: dict[str, Any] | None = None
    pii: PIIReport | Literal["Not yet evaluated"]


class GenerateEmbeddingIn(_Model):
    section_texts: dict[str, str]
    role: Literal["query", "passage"]  # student sections are queries, job sections passages (embeddings/provider.py)


class FallbackVectors(_Model):
    """The same sections under the fallback model (TF-IDF), stored next to the primary vectors (docs/adr/0012)."""

    model_name: str
    model_version: str
    dimension: int
    interim: bool
    vectors: dict[str, Vector]


class GenerateEmbeddingOut(_Model):
    model_name: str
    model_version: str
    dimension: int
    interim: bool
    vectors: dict[str, Vector]
    no_signal_sections: list[str]  # text with no in-vocabulary terms: not embedded
    fallback: FallbackVectors | None = None


def embedding_rows(embedding: "GenerateEmbeddingOut | None") -> list[EmbeddingRow]:
    """Stored rows for the primary model's vectors and, when present, the fallback model's."""
    if embedding is None:
        return []
    out = []
    for block in (embedding, embedding.fallback):
        if block is not None:
            out += [EmbeddingRow(section=s, model_name=block.model_name, model_version=block.model_version,
                                 dimension=block.dimension, values=v.values, indices=v.indices, interim=block.interim)
                    for s, v in block.vectors.items()]
    return out


class ValidateProfileIn(_Model):
    profile: dict[str, Any]


class ValidateProfileOut(_Model):
    valid: bool
    missing_fields: list[str]
    problems: list[str]


class SaveProfileIn(_Model):
    profile: dict[str, Any]
    embedding: GenerateEmbeddingOut | None = None


class SaveProfileOut(_Model):
    saved: bool
    student_id: str
    rows: dict[str, int]
    embeddings_stored: int


def missing_fields(profile: StudentProfile) -> list[str]:
    return [name for name, value in profile.model_dump().items()
            if isinstance(value, str) and value in ("Not detected", "Not yet evaluated", "Skill not detected")]


def register_profile_tools(registry: ToolRegistry, *, repo: ProfileRepository | None,
                           embedder: EmbeddingProvider, fallback_embedder: EmbeddingProvider | None = None) -> None:
    def _repo() -> ProfileRepository:
        if repo is None:
            raise RuntimeError("no database configured (DATABASE_URL)")
        return repo

    @registry.register("extract_text", input_schema=ExtractTextIn, output_schema=ExtractTextOut)
    def tool_extract_text(text=None, content_b64=None, filename=None):
        """Resume text from raw text or a .txt/.pdf/.docx file; rejects empty, binary or undecodable input."""
        try:
            extracted, fmt = extract_text(text, content_b64, filename)
        except ResumeRejected as exc:
            raise ToolInputRejected("extract_text", f"resume rejected: {exc}") from exc
        return {"text": extracted, "source_format": fmt, "char_count": len(extracted)}

    @registry.register("detect_pii", input_schema=DetectPiiIn, output_schema=DetectPiiOut)
    def tool_detect_pii(fields):
        """Flag (and mask, for downstream use) emails, phones, URLs, names, addresses and dates of birth."""
        findings, masked = [], {}
        for name, value in fields.items():
            found = pii_detector.detect(value, field=name)
            findings += found
            # Phase 12: the detected values themselves are redacted from every later log and trace record of
            # this request (held in memory only; app/core/logging.py)
            for f in found:
                register_sensitive_values([value[f.start:f.end]], words=f.type == "name")
            masked[name] = pii_detector.mask(value, found)
        counts: dict[str, int] = {}
        for f in findings:
            counts[f.type] = counts.get(f.type, 0) + 1
        report = PIIReport(detector_version=pii_detector.DETECTOR_VERSION, detected=bool(findings), counts=counts)
        # Counts only, never values. Logged as {"type", "count"} entries, because keys such as
        # "email" are (rightly) masked by the log redactor.
        log_event(logger, "pii_flags", detector_version=report.detector_version,
                  flags=[{"type": t, "count": n} for t, n in sorted(counts.items())], fields_scanned=sorted(fields))
        return {"masked_fields": masked, "report": report, "findings": findings}

    @registry.register("load_student_record", input_schema=LoadStudentIn, output_schema=LoadStudentOut)
    def tool_load_student_record(student_id, required):
        """Fetch a stored student (student_profiles). Unknown and required is a deterministic rejection."""
        record = _repo().get_student(student_id)
        if record is None and required:
            raise ToolInputRejected("load_student_record", f"unknown student_id {student_id}")
        stored = None
        if record is not None and record.get("source_dataset") == "resume_upload":
            stored = _repo().get_stored_profile(student_id)
            if stored is None and required:  # no resume now, none stored: nothing honest to build from
                raise ToolInputRejected("load_student_record", f"{student_id}: upload the resume again")
        return {"found": record is not None, "record": record, "stored_profile": stored}

    @registry.register("parse_resume", input_schema=TextIn, output_schema=SectionsOut)
    def tool_parse_resume(text):
        """Section extraction from resume text via the existing src/parsing code."""
        return {"sections": resume_sections(text)}

    @registry.register("parse_student_sections", input_schema=RecordIn, output_schema=SectionsOut)
    def tool_parse_student_sections(record):
        """Section extraction from a stored record's columns."""
        return {"sections": record_sections(record)}

    @registry.register("extract_skills", input_schema=ExtractSkillsIn, output_schema=ExtractSkillsOut)
    def tool_extract_skills(text, source="record"):
        """Uploaded resumes: the heading-aware resume_skill_items() (category prefixes, grouped skills in
        parentheses, wrapped lines), falling back to src/parsing's extract_skills() when it finds no skills section.
        Stored records: src/parsing's extract_skills() unchanged, on _clean(text) as parse_resume() does.
        """
        if source == "resume":
            items = resume_skill_items(text)
            if items:
                return {"skills": items}
        return {"skills": extract_skills(_clean(text))}

    @registry.register("normalize_skills", input_schema=NormalizeSkillsIn, output_schema=NormalizeSkillsOut)
    def tool_normalize_skills(skills):
        """Map raw skills to canonical names via app/profile/data/skill_aliases.csv; unknown ones stay as written."""
        normalized = normalize_skills(skills)
        return {"skills": normalized, "alias_table_version": alias_table_version(),
                "unmapped": sum(s.method == "unmapped" for s in normalized)}

    @registry.register("extract_profile", input_schema=ExtractProfileIn, output_schema=StudentProfile)
    def tool_extract_profile(student_id, source, sections, skills, record=None, pii=NOT_YET_EVALUATED):
        """Assemble the structured StudentProfile. Blank or unknown values become sentinels, never defaults."""
        record = record or {}
        not_extracted = NOT_EXTRACTED_FOR_SOURCE.get(record.get("source_dataset") or "", set())

        def from_record(field: str) -> Any:
            value = record.get(field)
            if value not in (None, ""):
                return value
            # no record, or a field never extracted for this source: not evaluated; otherwise genuinely blank
            return NOT_YET_EVALUATED if not record or field in not_extracted else NOT_DETECTED

        profile = StudentProfile(
            student_id=student_id, source=source,
            degree=sections["degree"], branch=from_record("branch"),
            graduation_year=from_record("graduation_year"), avg_cgpa_sem5=from_record("avg_cgpa_sem5"),
            education=sections["education"], skills=skills, projects=sections["projects"],
            experience=sections["experience"], certifications=sections["certifications"],
            achievements=from_record("achievements"), career_interests=from_record("career_interests"),
            domain=from_record("domain"), pii=pii, missing_fields=[], extractor_version=EXTRACTOR_VERSION,
        )
        return profile.model_copy(update={"missing_fields": missing_fields(profile)})

    @registry.register("generate_embedding", input_schema=GenerateEmbeddingIn, output_schema=GenerateEmbeddingOut)
    def tool_generate_embedding(section_texts, role):
        """Embed sections with the currently configured embedding model (EMBEDDING_MODEL), as student (query) or
        job (passage) text."""
        names = [n for n, t in section_texts.items() if t.strip()]
        vectors = embedder.embed([section_texts[n] for n in names], role)
        version, dimension = embedder.describe()
        kept = {n: v for n, v in zip(names, vectors) if v is not None}
        out = {"model_name": embedder.model_name, "model_version": version, "dimension": dimension,
               "interim": embedder.interim, "vectors": kept, "no_signal_sections": [n for n in names if n not in kept]}
        if fallback_embedder is not None and names:
            try:  # cheap (TF-IDF); a failure here never costs the primary vectors
                fb = fallback_embedder.embed([section_texts[n] for n in names], role)
                fb_version, fb_dimension = fallback_embedder.describe()
                out["fallback"] = {"model_name": fallback_embedder.model_name, "model_version": fb_version,
                                   "dimension": fb_dimension, "interim": fallback_embedder.interim,
                                   "vectors": {n: v for n, v in zip(names, fb) if v is not None}}
            except Exception as exc:  # noqa: BLE001
                log_event(logger, "fallback_embedding_failed", logging.WARNING, error=type(exc).__name__)
        return out

    @registry.register("validate_profile", input_schema=ValidateProfileIn, output_schema=ValidateProfileOut)
    def tool_validate_profile(profile):
        """Schema check, list sentinel fields, and refuse a profile that still contains PII."""
        profile_obj = StudentProfile.model_validate(profile)
        problems = []
        if isinstance(profile_obj.graduation_year, int) and not 1990 <= profile_obj.graduation_year <= 2035:
            problems.append("graduation_year outside 1990-2035")
        leaked = {f.type for v in _strings(profile_obj.model_dump(exclude={"pii", "missing_fields"}))
                  for f in pii_detector.detect(v, field="profile") if f.type in ("email", "phone", "url", "dob")}
        if leaked:
            problems.append(f"PII present in structured profile: {sorted(leaked)}")
        return {"valid": not problems, "missing_fields": missing_fields(profile_obj), "problems": problems}

    @registry.register("save_profile", input_schema=SaveProfileIn, output_schema=SaveProfileOut)
    def tool_save_profile(profile, embedding=None):
        """Write the profile, its child rows and its embeddings (with model name and version) in one transaction."""
        profile_obj = StudentProfile.model_validate(profile)
        # the registry hands validated inputs over as plain dicts
        embedding = GenerateEmbeddingOut.model_validate(embedding) if embedding is not None else None
        result = _repo().save_profile(profile_obj, embedding_rows(embedding))
        log_event(logger, "profile_saved", student_id=profile_obj.student_id, rows=result.rows,
                  embeddings_stored=result.embeddings_stored, level=logging.INFO)
        return {"saved": True, "student_id": profile_obj.student_id, "rows": result.rows,
                "embeddings_stored": result.embeddings_stored}


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)
