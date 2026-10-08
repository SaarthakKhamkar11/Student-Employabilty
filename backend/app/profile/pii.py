"""PII detection for resumes: auditable regular expressions plus a header heuristic for names.

It flags, and masks for downstream use, these types: email, phone, url
(personal profile and portfolio links), name, address and date of birth.
Detected values are never returned or logged, only their type, count and
character offsets. Masked text is what later stages, the LLM and the ML
pipeline get (spec §38).

Known blind spots (docs/adr/0005; measured rates in backend/tests/test_pii.py):
- Names are found only on the resume's first non-empty line or after a "Name:"
  label. A name anywhere else, a one-word name, or an all-lowercase name is missed.
- An address is found only after an "Address:" label, or on a line with a
  6-digit Indian PIN code plus an address word. Addresses without either are missed.
- Phone numbers need 10+ digits. Short or oddly separated numbers are missed,
  and long numeric IDs can be falsely flagged.
This is a baseline, not a guarantee. Never treat "no PII detected" as "contains no PII".
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

DETECTOR_VERSION = "pii-regex-v2"
PIIType = Literal["email", "phone", "url", "name", "address", "dob"]

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# A sentence-final "." is allowed after the number; ".<digit>" (a decimal) is not. (Phase 5 fix: the
# original lookahead rejected any trailing ".", so every phone number ending a sentence was missed.)
_PHONE = re.compile(r"(?<![\w.])(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?){2,4}\d{2,5}(?!\w)(?!\.\d)")
_URL = re.compile(r"\b(?:https?://|www\.)\S+|\b(?:linkedin\.com|github\.com|gitlab\.com|behance\.net)/\S*", re.I)
_DOB = re.compile(r"\b(?:d\.?o\.?b\.?|date\s+of\s+birth)\s*[:\-]?\s*[\w ,./-]{6,20}", re.I)
_LABELLED_NAME = re.compile(r"(?im)^\s*(?:full\s+)?name\s*[:\-]\s*(.+?)\s*$")
_LABELLED_ADDRESS = re.compile(r"(?im)^\s*(?:permanent\s+|current\s+|postal\s+)?address\s*[:\-]\s*(.+?)\s*$")
_PIN_LINE = re.compile(r"(?im)^.*\b[1-9]\d{5}\b.*$")
_ADDRESS_WORDS = re.compile(r"\b(road|rd\.?|street|st\.|nagar|lane|sector|flat|near|colony|apartment|apt|"
                            r"society|chs|building|floor|block|east|west|marg|chowk|village|dist(?:rict)?)\b", re.I)
_NAME_TOKEN = re.compile(r"^(?:[A-Z][a-z]+|[A-Z]\.?|[A-Z]{2,})$")
# Header words that start resumes but are not names (seen in the Kaggle data's first lines).
_NOT_NAME = {"resume", "curriculum", "vitae", "cv", "education", "details", "skills", "skill", "profile",
             "summary", "objective", "career", "personal", "technical", "areas", "interest", "experience",
             "engineer", "developer", "analyst", "scientist", "manager", "consultant", "student", "data",
             "software", "civil", "mechanical", "electrical", "testing", "about", "me", "contact", "information",
             "skillset", "operating", "systems", "system", "windows", "knowledge", "computer", "languages",
             "tools", "oracle", "dba", "sql", "java", "python", "web", "project", "projects", "work"}


class PIIFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    type: PIIType
    start: int
    end: int


def _name_on_first_line(text: str) -> tuple[int, int] | None:
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            candidate = re.split(r"\s*[|,•·]\s*", stripped)[0]  # "Asha K | asha@x.com" -> "Asha K"
            tokens = candidate.split()
            if (2 <= len(tokens) <= 4 and all(_NAME_TOKEN.match(t) for t in tokens)
                    and not {t.lower().strip(".") for t in tokens} & _NOT_NAME):
                start = offset + line.index(candidate)
                return start, start + len(candidate)
            return None
        offset += len(line)
    return None


def detect(text: str, field: str = "resume") -> list[PIIFinding]:
    spans: list[tuple[str, int, int]] = []
    for kind, pattern in (("email", _EMAIL), ("url", _URL), ("dob", _DOB)):
        spans += [(kind, m.start(), m.end()) for m in pattern.finditer(text)]
    for m in _PHONE.finditer(text):
        groups = re.findall(r"\d+", m.group())
        looks_like_years = all(len(g) == 4 and 1950 <= int(g) <= 2039 for g in groups)  # "2006-2007 2008-2009"
        if sum(len(g) for g in groups) >= 10 and not looks_like_years:
            spans.append(("phone", m.start(), m.end()))
    for m in _LABELLED_NAME.finditer(text):
        spans.append(("name", m.start(1), m.end(1)))
    if field == "resume" and (first := _name_on_first_line(text)):
        spans.append(("name", *first))
    for m in _LABELLED_ADDRESS.finditer(text):
        spans.append(("address", m.start(1), m.end(1)))
    for m in _PIN_LINE.finditer(text):
        if len(m.group()) <= 160 and _ADDRESS_WORDS.search(m.group()):
            spans.append(("address", m.start(), m.end()))
    # keep the earliest, longest span where findings overlap (e.g. an email inside a URL)
    spans.sort(key=lambda s: (s[1], -(s[2] - s[1])))
    kept: list[tuple[str, int, int]] = []
    for span in spans:
        if not kept or span[1] >= kept[-1][2]:
            kept.append(span)
    return [PIIFinding(field=field, type=k, start=s, end=e) for k, s, e in kept]


def mask(text: str, findings: list[PIIFinding]) -> str:
    for f in sorted(findings, key=lambda f: f.start, reverse=True):
        text = text[: f.start] + f"[{f.type.upper()}]" + text[f.end:]
    return text
