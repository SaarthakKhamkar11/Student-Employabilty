"""Resume free-text parser (heuristic): skills, education and experience.

Projects and certifications are only used as section boundaries; they are not
extracted yet (PROJECT_AUDIT.md, Section 0).
"""

import re

from src.parsing.section_parser import _clean, _slice_section

DEGREE_RE = re.compile(
    r"\b(B\.?\s?E|B\.?\s?Tech|M\.?\s?Tech|B\.?\s?Sc|M\.?\s?Sc|MBA|MCA|BCA|Diploma|Ph\.?D)\b",
    re.IGNORECASE,
)


def extract_skills(text):
    tl = text.lower()
    chunk = _slice_section(tl, text, "skill details") or _slice_section(tl, text, "skills")
    if not chunk:
        return []
    out = []
    for p in re.split(r"[,\n]", chunk):
        p = re.sub(r"^[A-Za-z /]+:\s*", "", p).strip(" .-*")
        if p and len(p) < 60 and not p.lower().startswith("company"):
            out.append(p)
    return out[:40]


def parse_resume(raw_text):
    text = _clean(raw_text)
    tl = text.lower()
    education = re.sub(r"\s+", " ", _slice_section(tl, text, "education details")).strip()[:300]
    experience = re.sub(r"\s+", " ", (_slice_section(tl, text, "company details")
                                       or _slice_section(tl, text, "experience"))).strip()[:500]
    return {"skills": extract_skills(text), "education_raw": education, "experience_raw": experience}


def guess_degree(education_raw):
    m = DEGREE_RE.search(education_raw or "")
    return m.group(1).upper().replace(" ", "") if m else ""
