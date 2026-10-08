"""Section extraction: thin wrappers over the existing parser in src/parsing (no logic duplicated).

- Resume text: parse_resume() for education and experience, guess_degree() for
  the degree, and the existing _slice_section() for projects and certifications.
  Those two markers were already in SECTION_MARKERS but were never extracted.
- Stored records: sections come straight from the student_profiles columns.

A section the parser looked for and did not find is NOT_DETECTED. A section
no extractor looks for (for this source) is NOT_YET_EVALUATED.
"""

import re
from typing import Any

from app.schemas.agent_state import NOT_DETECTED, NOT_YET_EVALUATED
from src.parsing.resume_text_parser import guess_degree, parse_resume
from src.parsing.section_parser import _clean, _slice_section

# Fields src/pipeline never extracted for a source, so a blank one means "not evaluated", not "absent".
NOT_EXTRACTED_FOR_SOURCE = {
    "kaggle_resume_dataset": {"technical_projects", "certifications", "achievements", "graduation_year",
                              "avg_cgpa_sem5", "country"},
}


def _items(text: str, limit: int = 20) -> list[str]:
    """Split a section into list items by line or bullet (never by comma: titles can contain commas)."""
    parts = re.split(r"\n|\s[•·▪➢]\s|\s\*\s", text)
    return [p.strip(" -*•\t") for p in parts if len(p.strip(" -*•\t")) > 2][:limit]


def resume_sections(text: str) -> dict[str, Any]:
    parsed = parse_resume(text)
    clean = _clean(text)
    lower = clean.lower()
    projects = re.sub(r"[ \t]+", " ", _slice_section(lower, clean, "projects")).strip()[:1500]
    certifications = re.sub(r"[ \t]+", " ", _slice_section(lower, clean, "certifications")).strip()[:800]
    return {
        "education": parsed["education_raw"] or NOT_DETECTED,
        "degree": guess_degree(parsed["education_raw"]) or NOT_DETECTED,
        "experience": [parsed["experience_raw"]] if parsed["experience_raw"] else NOT_DETECTED,
        "projects": _items(projects) or NOT_DETECTED,
        "certifications": _items(certifications) or NOT_DETECTED,
    }


def record_sections(record: dict[str, Any]) -> dict[str, Any]:
    not_extracted = NOT_EXTRACTED_FOR_SOURCE.get(record.get("source_dataset") or "", set())

    def missing(field: str) -> str:
        return NOT_YET_EVALUATED if field in not_extracted else NOT_DETECTED

    def text(field: str) -> str:
        return (record.get(field) or "").strip()

    education = " ".join(filter(None, [text("degree"), text("branch")]))
    return {
        "education": education or missing("degree"),
        "degree": text("degree") or missing("degree"),
        "experience": [text("internship_raw")] if text("internship_raw") else missing("internship_raw"),
        "projects": _items(text("technical_projects")) or missing("technical_projects"),
        "certifications": _items(text("certifications")) or missing("certifications"),
    }


def record_skills_text(record: dict[str, Any]) -> str:
    """Make stored skills_raw readable by the existing extract_skills(), which slices a "Skills" section.

    Kaggle rows store skills '; '-joined (src/pipeline/ingest_kaggle_resumes.py), so the join is reversed.
    """
    return "Skills\n" + (record.get("skills_raw") or "").replace("; ", "\n")
