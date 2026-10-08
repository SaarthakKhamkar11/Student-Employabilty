"""Heuristic regex section slicer for free-text resumes.

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


def _slice_section(text_lower, text_orig, marker):
    idx = text_lower.find(marker)
    if idx == -1:
        return ""
    start, end = idx + len(marker), len(text_orig)
    for other in SECTION_MARKERS:
        if other == marker:
            continue
        pos = text_lower.find(other, start)
        if pos != -1:
            end = min(end, pos)
    return text_orig[start:end].strip()
