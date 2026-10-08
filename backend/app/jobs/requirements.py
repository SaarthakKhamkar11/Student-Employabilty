"""Inferred requirements: required vs preferred skills, minimum degree, experience range.

The REQUIRED/PREFERRED split is inferred by a heuristic, never read from ground
truth (docs/adr/0008). Rules are applied per skill, in order, and the first
that fires is recorded as `requirement_basis`:

1. explicit_required_section / explicit_preferred_section: the skill came from
   a "Requirements"/"Must have" or a "Nice to have"/"Preferred" section of a free-text JD.
2. explicit_preferred_language / explicit_required_language: a sentence of the
   description or responsibilities names the skill next to cue words such as
   "preferred", "a plus", "nice to have" / "must", "required", "mandatory".
3. named_in_role_text -> required: at least 2/3 of the skill's content words appear,
   lightly stemmed, in the description or responsibilities. v2 also matches related
   word forms (see _related): planning/plan, coordination/coordinate,
   management/manage.
4. not_named_in_role_text -> preferred: otherwise.
5. first_listed_fallback: if nothing ended up required, the first skill labelled by
   rule 4 is marked required (a posting with zero required skills is implausible).
   Explicitly preferred skills are never promoted.

Degree: the existing guess_degree() (src/parsing) is tried first. Four job-side
qualifications it doesn't know (B.Com, M.Com, BBA, BA, MA) are handled here,
so the student-side parser and its data stay unchanged.
"""

import re
from typing import Any

from src.parsing.resume_text_parser import guess_degree

HEURISTIC_VERSION = "required-preferred-heuristic-v2"
INFERRED_NOTE = ("Required/preferred split is INFERRED by heuristic (docs/adr/0008), not ground truth. "
                 "The source data does not state which skills are required.")

# Requirement *phrasing* only. Bare "essential", "required" and "advantage" were tried and produced
# only false positives on the job data ("essential healthcare services", "required for web
# applications", "competitive advantage"), so they are excluded.
_REQUIRED_CUES = re.compile(r"\b(must[- ]have|must be|mandatory|(?:is|are) (?:a )?(?:must|required)|"
                            r"required (?:skills?|qualifications?|experience))\b", re.I)
_PREFERRED_CUES = re.compile(r"\b(nice[- ]to[- ]have|good[- ]to[- ]have|preferred|(?:is |are )?a plus|desirable|"
                             r"bonus points?)\b", re.I)
_GENERIC = {"and", "or", "of", "the", "in", "for", "with", "to", "a", "an", "on", "e.g.", "skills", "skill",
            "knowledge", "proficiency", "expertise", "ability", "understanding", "familiarity", "experience",
            "strong", "basic", "advanced", "etc", "eg", "including", "tools", "techniques", "principles"}

_EXTRA_DEGREES = re.compile(r"\b(B\.?\s?Com|M\.?\s?Com|BBA|MBA|B\.?\s?A|M\.?\s?A)\b")  # case-sensitive: "BA", not "ba"
DEGREE_LEVEL = {"DIPLOMA": 1, "BE": 2, "BTECH": 2, "BSC": 2, "BCA": 2, "BCOM": 2, "BBA": 2, "BA": 2,
                "ME": 3, "MTECH": 3, "MSC": 3, "MCA": 3, "MBA": 3, "MCOM": 3, "MA": 3, "PHD": 4}
_EXPERIENCE_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", re.I)
_EXPERIENCE_MIN = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.I)


def _stem(word: str) -> str:
    w = word.lower().strip(".,;:()'\"")
    for suffix, keep in (("ies", "y"), ("ing", ""), ("ed", ""), ("s", "")):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3 and not w.endswith("ss"):
            w = w[: -len(suffix)] + keep
            # v2: "planning" -> "plann" -> "plan" (a doubled final consonant left by the suffix)
            if suffix in ("ing", "ed") and len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "aeioulsz":
                w = w[:-1]
            return w
    return w


def _related(a: str, b: str) -> bool:
    """v2: same word, a prefix of the other (shorter >= 5: manage/management, transport/transportation),
    or a shared 8-letter start (coordination/coordinate). 8, not 7, keeps communication/community apart."""
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 5 and long_.startswith(short):
        return True
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return common >= 8


def _overlap(words: set[str], role_words: set[str]) -> int:
    return sum(any(_related(w, r) for r in role_words) for w in words)


def content_words(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[A-Za-z][A-Za-z+#.\-]*", text) if w.lower() not in _GENERIC} - {""}


def _sentences(*texts: str) -> list[str]:
    return [s for t in texts for s in re.split(r"(?<=[.!?])\s+|\n", t or "") if s.strip()]


def classify_requirements(skills: list[dict[str, Any]], description: str, responsibilities: str) -> list[dict[str, Any]]:
    """Add requirement and requirement_basis to each skill (dicts with raw, canonical, ..., origin)."""
    role_words = content_words(f"{description} {responsibilities}")
    sentences = [(content_words(s), s) for s in _sentences(description, responsibilities)]
    out = []
    for skill in skills:
        words = content_words(skill["raw"]) | content_words(skill["canonical"])
        origin = skill.get("origin")
        if origin == "required":
            requirement, basis = "required", "explicit_required_section"
        elif origin == "preferred":
            requirement, basis = "preferred", "explicit_preferred_section"
        else:
            cue_sentences = [s for ws, s in sentences if words and _overlap(words, ws)]
            if any(_PREFERRED_CUES.search(s) for s in cue_sentences):
                requirement, basis = "preferred", "explicit_preferred_language"
            elif any(_REQUIRED_CUES.search(s) for s in cue_sentences):
                requirement, basis = "required", "explicit_required_language"
            elif words and _overlap(words, role_words) / len(words) >= 2 / 3:
                requirement, basis = "required", "named_in_role_text"
            else:
                requirement, basis = "preferred", "not_named_in_role_text"
        out.append({**{k: v for k, v in skill.items() if k != "origin"},
                    "requirement": requirement, "requirement_basis": basis})
    if not any(s["requirement"] == "required" for s in out):
        # Promote the first skill whose label was itself only inferred. Explicitly preferred skills
        # stay preferred: explicit evidence always beats this fallback.
        for i, s in enumerate(out):
            if s["requirement_basis"] == "not_named_in_role_text":
                out[i] = {**s, "requirement": "required", "requirement_basis": "first_listed_fallback"}
                break
    return out


def parse_degree(text: str) -> tuple[str | None, int | None]:
    """(degree code, level 1 diploma .. 4 doctorate), reusing guess_degree() first."""
    degree = guess_degree(text or "")
    if not degree:
        m = _EXTRA_DEGREES.search(text or "")
        degree = m.group(1).upper().replace(" ", "") if m else ""
    if not degree:
        return None, None
    return degree, DEGREE_LEVEL.get(degree.replace(".", ""))


def parse_experience(text: str) -> tuple[float | None, float | None]:
    """(min years, max years) from "2 to 8 Years", "0-2 years" or "3+ years"; (None, None) if not stated."""
    m = _EXPERIENCE_RANGE.search(text or "")
    if m:
        low, high = float(m.group(1)), float(m.group(2))
        return (low, high) if low <= high else (high, low)
    m = _EXPERIENCE_MIN.search(text or "")
    return (float(m.group(1)), None) if m else (None, None)
