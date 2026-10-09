"""Stage 1, eligibility: remove only what clearly fails a hard requirement (spec §14), counted by reason.

Rules (config/matching_baseline_v1.yaml `eligibility`), checked in this order; an item is counted under
the first rule it fails:
  near_duplicate   a reposted job is recommended once, as its canonical posting (docs/adr/0010)
  not_visible      jobs for a student only: the employer's visibility preference (app/jobs/visibility.py,
                   docs/adr/0009). Never applied when ranking candidates for an employer.
  degree           the job's minimum degree level is above the student's
  experience       the job's minimum years exceed the student's stated experience + tolerance
  mandatory_skill  a skill configured as mandatory (by job_id or role) is missing. Only configured skills
                   exclude; every other skill gap is a score penalty (spec §14: "Do not make all
                   requirements hard filters").
Unknown values never exclude: a requirement that can't be checked is kept (docs/adr/0007).

Location (jobs for a student; docs/adr/0013): same-country jobs first. When fewer than min_same_country
remain, jobs in other countries are added as a second tier, and retrieval takes every same-country job
before any other. A student whose country is unknown gets no tiering.
"""

from dataclasses import dataclass, field

from app.jobs.visibility import posting_visible_to
from app.matching.config import MatchingConfig
from app.matching.entities import JobView, StudentView
from app.profile.skills import skill_key

REASONS = ("near_duplicate", "not_visible", "degree", "experience", "mandatory_skill")
SAME, OTHER, ANY = "same_country", "other_country", "any"


@dataclass
class EligibilityResult:
    eligible: list[str]
    tiers: dict[str, str] = field(default_factory=dict)  # id -> same_country | other_country | any
    excluded: dict[str, int] = field(default_factory=lambda: {r: 0 for r in REASONS})
    location_rule: str = "not applied"
    checked: int = 0


def _mandatory(job: JobView, cfg: MatchingConfig) -> list[str]:
    m = cfg.eligibility.get("mandatory_skills") or {}
    return list((m.get("by_job_id") or {}).get(job.job_id, [])) + list((m.get("by_role") or {}).get(job.role or "", []))


def failed_rules(student: StudentView, job: JobView, cfg: MatchingConfig, *, visibility_gender: str | None = None,
                 visibility_enabled: bool = True, apply_visibility: bool = True) -> list[str]:
    """Every hard rule this pair fails (empty: eligible)."""
    e = cfg.eligibility
    failed = []
    if e.get("exclude_near_duplicates", True) and job.near_duplicate_of:
        failed.append("near_duplicate")
    if apply_visibility and not posting_visible_to(job.visibility, visibility_gender, enabled=visibility_enabled):
        failed.append("not_visible")
    if (e.get("degree_filter", True) and job.degree_level is not None and student.degree_level is not None
            and job.degree_level > student.degree_level):
        failed.append("degree")
    if (e.get("experience_filter", True) and job.min_experience_years is not None
            and job.min_experience_years > student.experience_years + float(e.get("experience_tolerance_years", 0))):
        failed.append("experience")
    have = {skill_key(s) for s in student.skills}
    if any(skill_key(s) not in have for s in _mandatory(job, cfg)):
        failed.append("mandatory_skill")
    return failed


def jobs_for_student(student: StudentView, jobs: list[JobView], cfg: MatchingConfig, *,
                     visibility_gender: str | None, visibility_enabled: bool) -> EligibilityResult:
    result = EligibilityResult(eligible=[], checked=len(jobs))
    passed: list[JobView] = []
    for job in jobs:
        failed = failed_rules(student, job, cfg, visibility_gender=visibility_gender,
                              visibility_enabled=visibility_enabled)
        if failed:
            result.excluded[failed[0]] += 1
        else:
            passed.append(job)
    e = cfg.eligibility
    if not e.get("country_tiering", True):
        result.location_rule = "not applied: country tiering is off in the matching config"
        tiers = {j.job_id: ANY for j in passed}
    elif not student.country:
        result.location_rule = "not applied: the student's country is not recorded"
        tiers = {j.job_id: ANY for j in passed}
    else:
        country = student.country.strip().lower()
        same = [j for j in passed if (j.country or "").strip().lower() == country]
        minimum = int(e.get("min_same_country", cfg.top_n))
        if len(same) >= minimum:
            passed = same
            tiers = {j.job_id: SAME for j in same}
            result.location_rule = f"same country only: {len(same)} eligible jobs in {student.country}"
        else:
            tiers = {j.job_id: (SAME if j in same else OTHER) for j in passed}
            result.location_rule = (f"widened: only {len(same)} eligible job(s) in {student.country}, fewer than "
                                    f"{minimum}; jobs in other countries added after them")
    result.eligible = sorted(j.job_id for j in passed)
    result.tiers = tiers
    return result


def students_for_job(job: JobView, students: list[StudentView], cfg: MatchingConfig) -> EligibilityResult:
    """Candidates for an employer: no visibility filter and no location tiering (docs/adr/0009, 0013)."""
    result = EligibilityResult(eligible=[], checked=len(students),
                               location_rule="not applied: location is the employer's call, not a filter")
    for student in students:
        failed = [r for r in failed_rules(student, job, cfg, apply_visibility=False) if r != "near_duplicate"]
        if failed:
            result.excluded[failed[0]] += 1
        else:
            result.eligible.append(student.student_id)
    result.eligible.sort()
    result.tiers = {s: ANY for s in result.eligible}
    return result
