"""Stage 3, feature engineering: exactly the spec §11 features (FEATURE_NAMES), version FEATURE_VERSION.

Bump FEATURE_VERSION whenever a definition below changes. None means "can't be computed from the
evidence" (e.g. the job lists no preferred skills) and is never replaced by a default (docs/adr/0007).
Every vector is checked against the feature contract, which rejects protected attributes and PII.

Skill features (n = counts from the skill comparison; pc = partial_credit, rw = required_skill_weight):
  required_skill_match_score   (matched_req + pc * related_req) / required
  preferred_skill_match_score  (matched_pref + pc * related_pref) / preferred
  skill_match_score            (rw * (matched_req + pc * related_req) + matched_pref + pc * related_pref)
                               / (rw * required + preferred)
  matched_skill_count          job skills the profile names; missing_skill_count: job skills it doesn't
  required_skill_count         required skills of the job (inferred split, docs/adr/0008)
  matched_skill_ratio          matched / all job skills (no partial credit)
  preferred_skill_ratio        matched preferred / preferred (no partial credit)
  transferable_skill_score     related required skills / required skills the profile doesn't name: how much
                               of the required gap a related skill covers (None when nothing is missing)
Structured features:
  education_match   1 if the student's degree level meets the job's minimum, 0.5 one level below, else 0
  degree_match      1 if the same degree (B.E = B.Tech, M.E = M.Tech), else 0
  branch_match      1 if the student's branch fits the job's role family (config branch_rules), else 0
  experience_match  1 if stated experience >= the job's minimum (or the minimum is 0), else experience / minimum
  experience_gap    max(0, minimum - stated experience), in years (normalised later as 1 - min(gap, cap) / cap)
Similarity features, calibrated percentiles in [0, 1] (app/matching/calibration.py):
  semantic_similarity        retrieval's section-weighted similarity (skills, experience, projects)
  project_relevance          projects vs the job description
  certification_relevance    certifications vs the job's skills
  career_interest_alignment  the student's stated interest (career interest + domain) vs the job title and role
  role_similarity            past experience vs the job title and role
  domain_similarity          the student's branch and domain vs the full job text
"""

from app.matching.config import MatchingConfig
from app.matching.entities import JobView, StudentView, degree_family
from app.matching.features import FEATURE_NAMES, validate_feature_names
from app.matching.skill_gap import SkillComparison

FEATURE_VERSION = "features-v2"  # v2: a job skill stated in the profile text counts as matched
SIMILARITY_FEATURES = ("semantic_similarity", "project_relevance", "certification_relevance",
                       "career_interest_alignment", "role_similarity", "domain_similarity")


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 6) if den else None


def build_features(student: StudentView, job: JobView, skills: SkillComparison,
                   similarity_percentiles: dict[str, float | None], cfg: MatchingConfig) -> dict[str, float | None]:
    pc, rw = float(cfg.features["partial_credit"]), float(cfg.features["required_skill_weight"])
    n_req = sum(1 for _, r in job.skills if r == "required")
    n_pref = sum(1 for _, r in job.skills if r == "preferred")
    m_req, m_pref = skills.count("matched", "required"), skills.count("matched", "preferred")
    r_req, r_pref = skills.count("related", "required"), skills.count("related", "preferred")
    has_skills = bool(job.skills)

    if job.degree_level is None or student.degree_level is None:
        education = None
    else:
        gap = job.degree_level - student.degree_level
        education = 1.0 if gap <= 0 else 0.5 if gap == 1 else 0.0
    s_deg, j_deg = degree_family(student.degree_code), degree_family(job.degree_code)
    branch_families, job_families = cfg.families_of_branch(student.branch), cfg.families_of_role(job.role, job.title)
    minimum = job.min_experience_years
    if minimum is None:
        experience = gap_years = None
    else:
        experience = 1.0 if minimum <= 0 or student.experience_years >= minimum else round(student.experience_years / minimum, 6)
        gap_years = round(max(0.0, minimum - student.experience_years), 4)

    features = {
        "skill_match_score": _ratio(rw * (m_req + pc * r_req) + m_pref + pc * r_pref, rw * n_req + n_pref),
        "required_skill_match_score": _ratio(m_req + pc * r_req, n_req),
        "preferred_skill_match_score": _ratio(m_pref + pc * r_pref, n_pref),
        "education_match": education,
        "degree_match": None if not (s_deg and j_deg) else 1.0 if s_deg == j_deg else 0.0,
        "branch_match": None if branch_families is None or not job_families else float(bool(branch_families & job_families)),
        "experience_match": experience,
        "experience_gap": gap_years,
        "transferable_skill_score": _ratio(r_req, n_req - m_req),
        "matched_skill_count": float(m_req + m_pref) if has_skills else None,
        "missing_skill_count": float(n_req + n_pref - m_req - m_pref) if has_skills else None,
        "required_skill_count": float(n_req) if has_skills else None,
        "matched_skill_ratio": _ratio(m_req + m_pref, n_req + n_pref),
        "preferred_skill_ratio": _ratio(m_pref, n_pref),
        **{name: similarity_percentiles.get(name) for name in SIMILARITY_FEATURES},
    }
    validate_feature_names(features)  # the contract: no protected attribute, nothing outside spec §11
    if set(features) != set(FEATURE_NAMES):
        raise RuntimeError(f"{FEATURE_VERSION} must build every contract feature")
    return features


def normalise(name: str, value: float | None, cfg: MatchingConfig) -> float | None:
    """Each feature on [0, 1], higher = better fit. Counts are not good or bad on their own: None (weight 0)."""
    if value is None or name in ("matched_skill_count", "missing_skill_count", "required_skill_count"):
        return None
    if name == "experience_gap":
        cap = float(cfg.features["experience_gap_cap_years"])
        return round(1 - min(value, cap) / cap, 6)
    return round(min(max(value, 0.0), 1.0), 6)
