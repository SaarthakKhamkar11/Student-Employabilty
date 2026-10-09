"""What the matching stages read about a student or a job: views built from the stored structured profile
(Phase 4) and posting (Phase 5).

A StudentView never holds the student's gender. The visibility filter gets it separately, and nothing
else does (docs/adr/0009). Missing values stay None and are reported, never guessed (docs/adr/0007).
"""

import re
from dataclasses import dataclass, field
from typing import Any

from app.jobs.requirements import parse_degree
from app.profile.skills import skill_key

SENTINELS = ("Not detected", "Not yet evaluated", "Skill not detected", "Insufficient evidence")
# "4 months", "2.5 yrs", "18 months", "6 weeks": durations a student states for internships or per skill
_DURATION = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(months?|mos?|mnths?|years?|yrs?|weeks?|wks?)\b", re.I)
_PER_YEAR = {"m": 12.0, "y": 1.0, "w": 52.0}
# career_interests holds a plan, not a field, for the placement cohort
_PLANS = {"placements", "masters", "higher studies"}


def _value(v: Any) -> Any:
    return None if v is None or (isinstance(v, str) and (v in SENTINELS or not v.strip())) else v


def stated_experience_years(profile: dict[str, Any]) -> tuple[float, str]:
    """The longest duration the student states anywhere (experience entries, per-skill evidence), in years.

    A conservative lower bound: date ranges ("Jan 2019 - Mar 2020") are not parsed, and durations are not
    summed (one internship can be mentioned twice). No stated duration means 0 years of evidence.
    """
    texts = [str(e) for e in (profile.get("experience") or []) if isinstance(profile.get("experience"), list)]
    skills = profile.get("skills")
    if isinstance(skills, list):
        texts += [s.get("evidence") or "" for s in skills]
    years = [float(n) / _PER_YEAR[unit[0].lower()] for t in texts for n, unit in _DURATION.findall(t)]
    if not years:
        return 0.0, "no duration stated"
    return round(max(years), 2), f"longest stated duration: {max(years):.2f} years"


def degree_family(code: str | None) -> str | None:
    """B.E = BE = B.Tech and M.E = M.Tech; other degrees by their own code."""
    if not code:
        return None
    c = code.replace(".", "").replace(" ", "").upper()
    return {"BE": "BTECH", "ME": "MTECH"}.get(c, c)


@dataclass(frozen=True)
class StudentView:
    student_id: str
    skills: tuple[str, ...]  # canonical names, de-duplicated, in profile order
    degree_code: str | None
    degree_level: int | None
    branch: str | None
    domain: str | None
    experience_years: float
    experience_basis: str
    country: str | None
    texts: dict[str, str] = field(default_factory=dict)  # embedding inputs, PII-masked upstream


@dataclass(frozen=True)
class JobView:
    job_id: str
    title: str | None
    role: str | None
    skills: tuple[tuple[str, str], ...]  # (canonical, "required" | "preferred")
    degree_code: str | None
    degree_level: int | None
    min_experience_years: float | None
    country: str | None
    near_duplicate_of: str | None
    visibility: Any  # the posting's visibility block, read only by app/jobs/visibility.py
    source: str | None  # jobs.source_dataset: the dataset, or "job_description_upload"
    texts: dict[str, str] = field(default_factory=dict)


def _unique(names) -> tuple[str, ...]:
    seen, out = set(), []
    for n in names:
        k = skill_key(n)
        if k and k not in seen:
            seen.add(k)
            out.append(n)
    return tuple(out)


def student_view(student_id: str, profile: dict[str, Any], record: dict[str, Any] | None = None) -> StudentView:
    from app.agents.profile_agent import section_texts  # the Profile Agent's own embedding texts (no import cycle)

    record = record or {}
    skills = profile.get("skills")
    names = _unique(s["canonical"] for s in skills) if isinstance(skills, list) else ()
    code, level = parse_degree(_value(profile.get("degree")) or _value(record.get("degree")) or "")
    years, basis = stated_experience_years(profile)
    domain = _value(profile.get("domain"))
    interests = _value(profile.get("career_interests"))
    if interests and interests.strip().lower() in _PLANS:
        interests = None
    texts = section_texts(profile)  # skills, experience, education, projects, certifications, full
    interest = " ".join(t for t in (interests, domain) if t)
    field_text = " ".join(t for t in (_value(profile.get("branch")), domain) if t)
    if interest:
        texts["interest"] = interest
    if field_text:
        texts["field"] = field_text
    return StudentView(student_id=student_id, skills=names, degree_code=code, degree_level=level,
                       branch=_value(profile.get("branch")), domain=domain, experience_years=years,
                       experience_basis=basis, country=_value(record.get("country")), texts=texts)


def job_view(job_id: str, posting: dict[str, Any], record: dict[str, Any] | None = None) -> JobView:
    from app.agents.job_intelligence_agent import job_section_texts  # the Job Agent's own embedding texts

    record = record or {}
    skills = posting.get("skills")
    pairs = []
    if isinstance(skills, list):
        seen = set()
        for s in skills:
            k = skill_key(s["canonical"])
            if k and k not in seen:
                seen.add(k)
                pairs.append((s["canonical"], s["requirement"]))
    level = posting.get("min_degree_level")
    min_years = posting.get("min_experience_years")
    title, role = _value(posting.get("job_title")), _value(posting.get("role"))
    texts = job_section_texts(posting)  # skills, required_skills, responsibilities, description, education, full
    title_text = " - ".join(dict.fromkeys(t for t in (title, role) if t))
    if title_text:
        texts["title"] = title_text
    return JobView(job_id=job_id, title=title, role=role, skills=tuple(pairs),
                   degree_code=_value(posting.get("min_degree")), degree_level=level if isinstance(level, int) else None,
                   min_experience_years=float(min_years) if isinstance(min_years, (int, float)) else None,
                   country=_value(posting.get("country")) or _value(record.get("country")),
                   near_duplicate_of=record.get("near_duplicate_of"), visibility=posting.get("visibility"),
                   source=record.get("source_dataset"), texts=texts)
