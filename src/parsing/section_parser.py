"""Heuristic regex section slicer for free-text resumes and (since Phase 5) job descriptions.

Known limitation: inconsistent resume headers mean no skills section is found
on ~20% of the deduplicated Kaggle resumes (PROJECT_AUDIT.md, problem I-8).
"""

import re

SECTION_MARKERS = ["education details", "skill details", "skills",
                    "company details", "experience", "projects",
                    "certifications", "declaration"]
BULLET_NOISE = re.compile(r"[â¢\x80-\xff\uf0b7\u2022]+")


def _clean(text):
    text = BULLET_NOISE.sub(" ", text)
    return re.sub(r"[\r\n]+", "\n", text).strip()


def _header_pattern(marker):
    # Phase 5: a header is the marker at the start of a line, then ":" (inline content allowed) or end of line.
    return re.compile(r"(?m)^[ \t]*" + re.escape(marker) + r"[ \t]*(?::|$)")


def _slice_section(text_lower, text_orig, marker, markers=SECTION_MARKERS, line_anchored=False):
    """Text after `marker` up to the next of `markers`.

    Defaults are the original resume behaviour (substring matching, resume markers).
    Phase 5 added `markers` and `line_anchored` for job descriptions, where words such
    as "experience" or "required" also appear inside sentences.
    """
    if line_anchored:
        found = _header_pattern(marker).search(text_lower)
        if not found:
            return ""
        start, end = found.end(), len(text_orig)
        for other in markers:
            if other != marker and (nxt := _header_pattern(other).search(text_lower, start)):
                end = min(end, nxt.start())
        return text_orig[start:end].strip()
    idx = text_lower.find(marker)
    if idx == -1:
        return ""
    start, end = idx + len(marker), len(text_orig)
    for other in markers:
        if other == marker:
            continue
        pos = text_lower.find(other, start)
        if pos != -1:
            end = min(end, pos)
    return text_orig[start:end].strip()


# Phase 5: job-description headers, grouped by the section they introduce.
JOB_SECTION_MARKERS = {
    "description": ["job description", "about the role", "role overview", "overview", "summary"],
    "responsibilities": ["responsibilities", "key responsibilities", "role and responsibilities", "duties",
                         "what you will do", "what you'll do"],
    "required": ["requirements", "required skills", "required qualifications", "minimum qualifications",
                 "qualifications", "must have", "must-have", "who you are"],
    "preferred": ["preferred skills", "preferred qualifications", "nice to have", "nice-to-have", "good to have",
                  "bonus points", "desirable"],
    "skills": ["skills", "key skills", "technical skills"],
    "experience": ["experience"],
    "education": ["education"],
    "other": ["benefits", "perks", "about us", "about the company", "how to apply"],
}
_ALL_JOB_MARKERS = [m for group in JOB_SECTION_MARKERS.values() for m in group]


def parse_job_sections(text):
    """Split a free-text job description into sections by line-anchored headers.

    Returns {"title": first short line or "", "<section>": text, ...} with only the sections found.
    Line endings are normalised, but the resume-specific bullet-noise cleaning of _clean() is not
    applied: it would also strip accented letters from job text.
    """
    text = re.sub(r"\r\n?", "\n", text).strip()
    lower = text.lower()
    first = next((line.strip() for line in text.split("\n") if line.strip()), "")
    is_header = any(_header_pattern(m).match(first.lower()) for m in _ALL_JOB_MARKERS)
    found = {"title": first if len(first) <= 80 and not is_header else ""}
    for section, group in JOB_SECTION_MARKERS.items():
        parts = [_slice_section(lower, text, m, markers=_ALL_JOB_MARKERS, line_anchored=True) for m in group]
        joined = "\n".join(p for p in parts if p)
        if joined:
            found[section] = joined
    return found
