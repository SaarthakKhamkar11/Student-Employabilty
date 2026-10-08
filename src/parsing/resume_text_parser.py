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
        # Bug fix (Phase 4, docs/adr/0006): strip(" .-*") turned ".NET" into "NET". A leading
        # dot is now kept when a letter follows it; trailing dots are still removed.
        p = re.sub(r"^[A-Za-z /]+:\s*", "", p).strip(" -*").rstrip(".")
        p = p if re.match(r"\.[A-Za-z]", p) else p.lstrip(". -*")
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
    for m in DEGREE_RE.finditer(education_raw or ""):
        # Bug fix (Phase 5, docs/adr/0006): IGNORECASE also matched the ordinary word "be"
        # ("should be proficient" -> "BE"). Bare "be"/"Be" is skipped; "BE", "B.E", "b.e" still match.
        if m.group(1).replace(" ", "").lower() == "be" and not m.group(1).isupper():
            continue
        return m.group(1).upper().replace(" ", "")
    return ""
