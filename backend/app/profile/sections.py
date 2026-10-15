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


# ------------------------------------------------------------------------------------------- uploaded resumes
# Resume headings, matched as a whole line (case-insensitive). A line that is one of these ends the section above
# it. Uploaded resumes use far more headings than the Kaggle export src/parsing was written for.
_RESUME_HEADINGS = {
    "education": ["education", "education details", "academic background", "academics", "qualifications",
                  "academic qualifications"],
    "skills": ["skills", "technical skills", "skill details", "key skills", "core skills", "core competencies",
               "technical expertise", "tools and technologies", "tools & technologies", "skills & tools",
               "skills and tools", "technologies"],
    "projects": ["projects", "academic projects", "personal projects", "key projects", "technical projects",
                 "project experience", "selected projects"],
    "experience": ["experience", "work experience", "professional experience", "internships", "internship",
                   "internship experience", "employment", "employment history", "work history",
                   "leadership experience", "leadership", "positions of responsibility", "volunteer experience",
                   "extracurricular activities", "extra-curricular activities", "activities"],
    "certifications": ["certifications", "certificates", "certification", "licenses & certifications",
                       "courses", "courses & certifications", "certifications & courses", "online courses"],
    "other": ["achievements", "awards", "honors", "honours", "awards & achievements", "publications", "summary",
              "profile", "objective", "career objective", "interests", "hobbies", "languages", "declaration",
              "references", "personal details", "contact"],
}
_HEADING_TO_SECTION = {h: s for s, hs in _RESUME_HEADINGS.items() for h in hs}
_BULLET = re.compile(r"^\s*[•·▪➢●○◦■□✓✔\-–*]\s+")
_PDF_BULLET_NOISE = re.compile(r"[]")
_SENTENCE_END = re.compile(r"[.;:!?)]$")
_DATED = re.compile(r"\b(?:19|20)\d{2}\b|\bpresent\b|\bcurrent\b", re.I)


def _heading(line: str) -> str | None:
    """The section a line introduces, if it is a heading line ("PROJECTS", "Work Experience:")."""
    key = re.sub(r"\s+", " ", line.strip().rstrip(":").strip()).lower()
    return _HEADING_TO_SECTION.get(key) if 0 < len(key) <= 40 else None


def split_resume_sections(text: str) -> dict[str, list[str]]:
    """Resume lines grouped under the heading above them. Lines before the first heading go to "header"."""
    out: dict[str, list[str]] = {"header": []}
    current = "header"
    for raw in _PDF_BULLET_NOISE.sub("•", text).replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        section = _heading(line)
        if section:
            current = section
            out.setdefault(current, [])
            continue
        out.setdefault(current, []).append(line)
    return out


def group_entries(lines: list[str], limit: int = 20, max_chars: int = 1200) -> list[str]:
    """One readable item per resume entry: its title line, then its bullets joined as sentences.

    A PDF wraps a long bullet over several lines, so a line continues the previous one unless it starts a new
    bullet, or the previous line ended a sentence and this one starts with a capital (the next entry's title).
    A title line followed by a non-bullet line (an organisation, a place) keeps both in the title.
    """
    entries: list[dict[str, Any]] = []
    for line in lines:
        is_bullet = bool(_BULLET.match(line))
        text = _BULLET.sub("", line).strip()
        if not text:
            continue
        cur = entries[-1] if entries else None
        if is_bullet:
            if cur is None:
                cur = {"title": "", "bullets": []}
                entries.append(cur)
            cur["bullets"].append(text)
        elif cur is None:
            entries.append({"title": text, "bullets": []})
        elif cur["bullets"]:
            last = cur["bullets"][-1]
            if _SENTENCE_END.search(last) and text[:1].isupper():
                entries.append({"title": text, "bullets": []})
            else:
                cur["bullets"][-1] = f"{last} {text}"
        elif cur["title"] and not text[:1].isupper():  # a wrapped title line
            cur["title"] = f"{cur['title']} {text}"
        elif cur["title"] and _DATED.search(cur["title"]) and not _DATED.search(text) and len(cur["title"]) < 120:
            # a dated role header, then its organisation or place: "Vice Chairperson June 2025 - present" /
            # "DJSCE Mumbai". Two undated lines in a row are two separate entries ("Chatbot", "Sales dashboard").
            cur["title"] = f"{cur['title']} · {text}"
        else:
            entries.append({"title": text, "bullets": []})
    items = []
    for e in entries:
        body = " ".join(b if _SENTENCE_END.search(b) else f"{b}." for b in e["bullets"])
        item = f"{e['title']}: {body}" if e["title"] and body else (e["title"] or body)
        item = re.sub(r"\s+", " ", item).strip()
        if len(item) > 2:
            items.append(item[:max_chars])
    return items[:limit]


def _split_top_level(text: str) -> list[str]:
    """Split on commas, semicolons and pipes outside parentheses: 'Python (Pandas, NumPy), SQL' -> 2 items."""
    parts, depth, buf = [], 0, []
    for ch in text:
        depth += ch == "("
        depth -= ch == ")" and depth > 0
        if ch in ",;|" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


_CATEGORY_PREFIX = re.compile(r"^([A-Za-z][A-Za-z0-9 &/+.\-]{1,50}):\s+")


def resume_skill_items(text: str, limit: int = 60) -> list[str]:
    """Skills from an uploaded resume's skills section, one item per skill.

    Handles the layouts src/parsing.extract_skills() misses: category prefixes with '&' ("Reporting &
    Visualisation: Power BI"), groups in parentheses ("Python (Pandas, NumPy)" -> Python, Pandas, NumPy; the
    parentheses are split later by normalize_skills), and dash-joined qualifiers ("Excel (Advanced) – dashboards").
    Wrapped lines are joined before splitting, so a skill broken across two PDF lines stays one skill.
    """
    lines = split_resume_sections(text).get("skills", [])
    if not lines:
        return []
    groups: list[str] = []
    for line in lines:
        line = _BULLET.sub("", line).strip()
        if groups and not _CATEGORY_PREFIX.match(line) and groups[-1].rstrip().endswith((",", "(", "&", "/", "-")):
            groups[-1] = f"{groups[-1]} {line}"
        elif groups and not _CATEGORY_PREFIX.match(line) and _CATEGORY_PREFIX.match(groups[-1]):
            groups[-1] = f"{groups[-1]}, {line}" if not groups[-1].rstrip().endswith(",") else f"{groups[-1]} {line}"
        else:
            groups.append(line)
    out: list[str] = []

    def add(piece: str) -> None:
        piece = piece.strip(" .")
        if 1 < len(piece) <= 60 and piece.lower() not in {o.lower() for o in out}:
            out.append(piece)

    for group in groups:
        group = _CATEGORY_PREFIX.sub("", group)
        for item in _split_top_level(group):
            # en/em dashes join a qualifier ("Excel – dashboards"); a plain hyphen is left alone, because stated
            # experience is written with it ("Python- Experience - 24 months") and normalize_skills reads it
            for piece in re.split(r"\s+[–—]\s+", item):
                # "Python (Pandas, NumPy)" -> Python, Pandas, NumPy; "Excel (Advanced)" -> Excel
                m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", piece)
                if m:
                    add(m.group(1))
                    inner = [x.strip() for x in m.group(2).split(",")]
                    if len(inner) > 1 or inner[0].lower() not in _QUALIFIERS:
                        for x in inner:
                            add(x)
                else:
                    add(piece)
    return out[:limit]


_QUALIFIERS = {"advanced", "basic", "basics", "intermediate", "beginner", "expert", "proficient", "familiar",
               "working knowledge", "learning"}


def resume_sections(text: str) -> dict[str, Any]:
    """Sections of an uploaded resume. Heading-based (split_resume_sections); src/parsing's Kaggle-shaped parser
    is the fallback for a resume without recognisable headings."""
    found = split_resume_sections(text)
    parsed = parse_resume(text)
    education_lines = found.get("education", [])
    education = re.sub(r"\s+", " ", " ".join(education_lines[:4])).strip()[:300] or parsed["education_raw"]
    experience = group_entries(found.get("experience", []))
    if not experience and parsed["experience_raw"]:
        experience = [parsed["experience_raw"]]
    projects = group_entries(found.get("projects", []))
    if not projects:
        clean = _clean(text)
        projects = _items(re.sub(r"[ \t]+", " ", _slice_section(clean.lower(), clean, "projects")).strip()[:1500])
    certifications = [re.sub(r"\s+", " ", _BULLET.sub("", c)).strip() for c in found.get("certifications", [])]
    certifications = [c for c in certifications if len(c) > 2][:20]
    if not certifications:
        clean = _clean(text)
        certifications = _items(re.sub(r"[ \t]+", " ", _slice_section(clean.lower(), clean, "certifications")).strip()[:800])
    return {
        "education": education or NOT_DETECTED,
        "degree": guess_degree(education) or guess_degree(text[:2000]) or NOT_DETECTED,
        "experience": experience or NOT_DETECTED,
        "projects": projects or NOT_DETECTED,
        "certifications": certifications or NOT_DETECTED,
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
