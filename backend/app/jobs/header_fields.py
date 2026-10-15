"""Labelled facts in a free-text job description: job title, role, company, location, work type and salary.

Deterministic: a value is taken only from a labelled line ("Location: Mumbai", "CTC - 6-8 LPA") or, for salary
and work type, from an unambiguous pattern anywhere in the text ("₹6,00,000 per annum", "Full-time"). Nothing is
guessed; a fact the text doesn't state stays absent and the posting reports it as "Not detected".
"""

import re

_LABELS = {
    "job_title": ["job title", "title", "position", "position title", "designation", "opening"],
    "role": ["role", "job role", "role name", "function", "department"],
    "company_name": ["company", "company name", "organization", "organisation", "employer", "hiring company"],
    "location": ["location", "job location", "work location", "based in", "place of work", "office location",
                 "city", "locations"],
    "work_type": ["job type", "employment type", "work type", "type", "engagement", "work mode", "employment"],
    "salary_range": ["salary", "salary range", "ctc", "compensation", "pay", "stipend", "package", "remuneration",
                     "pay range", "salary (ctc)", "annual ctc", "expected ctc", "budget"],
}
_LABEL_OF = {label: field for field, labels in _LABELS.items() for label in labels}
_LINE = re.compile(r"^\s*[-*•]?\s*([A-Za-z][A-Za-z ()/]{1,30}?)\s*[:\-–|]\s+(.+?)\s*$")

_CURRENCY = r"(?:₹|rs\.?|inr|\$|usd|€|eur|£|gbp)"
_AMOUNT = r"\d[\d,]*(?:\.\d+)?\s*(?:k|l|lakh|lakhs|lac|lacs|cr|crore|m|mn|million)?"
_SALARY = re.compile(
    rf"({_CURRENCY}\s*{_AMOUNT}(?:\s*(?:-|–|to)\s*{_CURRENCY}?\s*{_AMOUNT})?"
    rf"(?:\s*(?:lpa|per\s+(?:annum|year|month|hour)|p\.?a\.?|/\s*(?:yr|year|month|mo|hr|hour)|pm|a\s+year|a\s+month))?"
    rf"|{_AMOUNT}(?:\s*(?:-|–|to)\s*{_AMOUNT})?\s*(?:lpa|lakhs?\s+per\s+annum|ctc))",
    re.I)
_WORK_TYPES = [("internship", "Internship"), ("full-time", "Full-Time"), ("full time", "Full-Time"),
               ("part-time", "Part-Time"), ("part time", "Part-Time"), ("contract", "Contract"),
               ("temporary", "Temporary"), ("freelance", "Contract")]
_MODES = [("remote", "Remote"), ("hybrid", "Hybrid"), ("on-site", "On-site"), ("onsite", "On-site")]


def _work_type(text: str, labelled: bool = True) -> str | None:
    """From a labelled value every type counts; from running text only the unambiguous ones ("contract" also
    appears in "contract management")."""
    lower = text.lower()
    kinds = _WORK_TYPES if labelled else [k for k in _WORK_TYPES if k[1] in ("Internship", "Full-Time", "Part-Time")]
    kind = next((label for key, label in kinds if re.search(rf"\b{re.escape(key)}\b", lower)), None)
    mode = next((label for key, label in _MODES if re.search(rf"\b{re.escape(key)}\b", lower)), None)
    return " · ".join(x for x in (kind, mode) if x) or None


def extract_header_fields(text: str) -> dict[str, str]:
    """{field: value} for the facts the text states. Fields: job_title, role, company_name, location, work_type,
    salary_range."""
    out: dict[str, str] = {}
    for line in text.replace("\r", "").split("\n")[:80]:
        for part in re.split(r"\s+\|\s+", line):  # "Location: Pune | Work type: Full-time"
            m = _LINE.match(part)
            if not m:
                continue
            field = _LABEL_OF.get(m.group(1).strip().lower())
            value = m.group(2).strip().strip(".")
            if field and value and len(value) <= 150 and field not in out:
                out[field] = _work_type(value) or value if field == "work_type" else value
    if "salary_range" not in out and (m := _SALARY.search(text)):
        out["salary_range"] = m.group(1).strip()
    if "work_type" not in out and (kind := _work_type(text[:3000], labelled=False)):
        out["work_type"] = kind
    return out
