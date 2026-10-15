"""Evidence for one explanation, built deterministically from the agent's inputs (docs/adr/0019).

Each EvidenceItem is one claim the explanation may make, tied to the input value behind it:

    positive  matched skills (SkillGapReport.matched_skills), related student skills, education / experience / branch
              requirements the feature vector shows as met, similarity features in the top quarter of all pairs
    gaps      SkillGapReport.missing_required and missing_preferred, with their counterfactual importance
    model     the pair's largest SHAP contributions, each with its sign and size

`fact` is the claim in plain words, written here, not by the LLM: it is also the structured fallback line.
"""

import math
from typing import Any

from app.matching.entities import JobView, StudentView
from app.matching.features import FEATURE_NAMES
from app.schemas.agent_io import EvidenceBundle, EvidenceItem, SkillGapReport

LABELS: dict[str, str] = {
    "skill_match_score": "overall skill match",
    "required_skill_match_score": "required-skill match",
    "preferred_skill_match_score": "preferred-skill match",
    "semantic_similarity": "semantic similarity",
    "education_match": "education level",
    "degree_match": "degree match",
    "branch_match": "branch fit",
    "experience_match": "experience match",
    "experience_gap": "experience gap",
    "project_relevance": "project relevance",
    "certification_relevance": "certification relevance",
    "career_interest_alignment": "career interest alignment",
    "transferable_skill_score": "transferable-skill coverage",
    "matched_skill_count": "matched-skill count",
    "missing_skill_count": "missing-skill count",
    "required_skill_count": "required-skill count",
    "matched_skill_ratio": "matched-skill ratio",
    "preferred_skill_ratio": "preferred-skill ratio",
    "role_similarity": "role similarity",
    "domain_similarity": "domain similarity",
}
assert set(LABELS) == set(FEATURE_NAMES)
SIMILARITY = ("semantic_similarity", "project_relevance", "certification_relevance", "career_interest_alignment",
              "role_similarity", "domain_similarity")
COUNTS = ("matched_skill_count", "missing_skill_count", "required_skill_count")
SPACE = {"log-odds": "log-odds", "probability": "probability points", "suitability": "suitability points"}


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def pct(value: float) -> int:
    return int(round(value * 100))


def describe_value(feature: str, value: float | None) -> tuple[str, list[float]]:
    """The feature's value in words, and the numbers that description states."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "value not available", []
    if feature in SIMILARITY:
        return f"{ordinal(pct(value))} percentile of student-job pairs", [value, pct(value)]
    if feature == "education_match":
        return {1.0: "degree level meets the job's minimum", 0.5: "degree level one below the job's minimum"}.get(
            value, "degree level below the job's minimum"), []
    if feature == "degree_match":
        return ("same degree as the job names" if value == 1 else "a different degree from the one the job names"), []
    if feature == "branch_match":
        return ("branch fits the job's role family" if value == 1 else "branch outside the job's role family"), []
    if feature == "experience_match":
        if value >= 1:
            return "stated experience meets the job's minimum", []
        return f"stated experience is {pct(value)}% of the job's minimum", [value, pct(value)]
    if feature == "experience_gap":
        return ("no gap to the job's minimum experience" if value == 0
                else f"{value:g} years below the job's minimum experience"), [value]
    if feature in COUNTS:
        return f"{int(value)}", [value]
    return f"{value:.2f}", [value]


def _rounded(*values: float) -> list[float]:
    return [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


def build_evidence(*, suitability: float, features: dict[str, float | None], report: SkillGapReport,
                   contributions: dict[str, Any], semantic_raw: float | None, student: StudentView | None,
                   job: JobView | None, positive_similarity_percentile: float) -> EvidenceBundle:
    """`contributions` is get_feature_contributions' output: the selected positive and negative SHAP items."""
    requirement = {s.lower(): r for s, r in job.skills} if job is not None else {}
    positive: list[EvidenceItem] = []

    def add_positive(kind, fact, anchors, source, mentions=(), numbers=()):
        positive.append(EvidenceItem(id=f"P{len(positive) + 1}", kind=kind, fact=fact, anchors=list(anchors),
                                     mentions=list(mentions), numbers=_rounded(*numbers), source=source))

    for skill in report.matched_skills:
        req = requirement.get(skill.lower())
        fact = (f"{skill} requirement satisfied ({req} skill, detected in the profile)" if req
                else f"{skill} requirement satisfied (detected in the profile)")
        add_positive("matched_skill", fact, [skill], {"skill_gap_report": "matched_skills", "skill": skill,
                                                       "requirement": req or "not available"})
    for gap_list, req in ((report.missing_required, "required"), (report.missing_preferred, "preferred")):
        for g in gap_list:
            if g.transferable_candidate:
                add_positive("transferable_skill",
                             f"{g.transferable_candidate} detected, related to the {req} skill {g.skill} "
                             f"(name similarity {g.transferable_similarity:.2f})",
                             [g.transferable_candidate], {"skill_gap_report": f"missing_{req}", "skill": g.skill,
                                                          "transferable_candidate": g.transferable_candidate,
                                                          "similarity": g.transferable_similarity},
                             mentions=[g.skill], numbers=[g.transferable_similarity])
    if features.get("education_match") == 1.0:
        detail = (f": {student.degree_code} meets the job's minimum {job.degree_code}"
                  if student is not None and job is not None and student.degree_code and job.degree_code else "")
        add_positive("education", f"Education requirement satisfied (degree level meets the job's minimum{detail})",
                     ["education"], {"feature": "education_match", "value": 1.0,
                                     "student_degree": student.degree_code if student else "not available",
                                     "job_min_degree": job.degree_code if job else "not available"},
                     mentions=[LABELS["education_match"], LABELS["degree_match"]])
    if (features.get("experience_match") == 1.0 and job is not None and student is not None
            and (job.min_experience_years or 0) > 0):
        add_positive("experience",
                     f"Experience requirement satisfied (stated {student.experience_years:g} years, job minimum "
                     f"{job.min_experience_years:g} years)", ["experience"],
                     {"feature": "experience_match", "value": 1.0, "stated_years": student.experience_years,
                      "basis": student.experience_basis, "job_min_years": job.min_experience_years},
                     mentions=[LABELS["experience_match"], LABELS["experience_gap"]],
                     numbers=[student.experience_years, job.min_experience_years])
    if features.get("branch_match") == 1.0:
        branch = f" ({student.branch})" if student is not None and student.branch else ""
        add_positive("branch", f"Branch{branch} fits the job's role family", ["branch"],
                     {"feature": "branch_match", "value": 1.0, "branch": student.branch if student else "not available"},
                     mentions=[LABELS["branch_match"]])
    for feature in SIMILARITY:
        value = features.get(feature)
        if value is not None and value >= positive_similarity_percentile:
            label = LABELS[feature]
            add_positive("similarity", f"{label[0].upper()}{label[1:]} at the {ordinal(pct(value))} percentile of "
                                       "student-job pairs", [label], {"feature": feature, "percentile": value},
                         numbers=[value, pct(value)])

    gaps: list[EvidenceItem] = []
    related = {p.skill.lower(): p for p in report.partially_related}
    for gap_list, kind, req in ((report.missing_required, "missing_required", "required"),
                                (report.missing_preferred, "missing_preferred", "preferred")):
        for g in gap_list:
            parts = [f"{req} skill"]
            if g.importance_score is not None:
                parts.append(f"model importance {g.importance_score:+.3f}")
            mentions, numbers = [], [g.importance_score]
            if g.transferable_candidate:
                parts.append(f"related skill {g.transferable_candidate} detected")
                mentions.append(g.transferable_candidate)
                numbers.append(g.transferable_similarity)
            elif g.skill.lower() in related:
                p = related[g.skill.lower()]
                parts.append(f"partially related to {p.related_student_skill}, similarity {p.similarity:.2f}")
                mentions.append(p.related_student_skill)
                numbers.append(p.similarity)
            gaps.append(EvidenceItem(
                id=f"G{len(gaps) + 1}", kind=kind, fact=f"{g.skill} not detected ({'; '.join(parts)})",
                anchors=[g.skill], mentions=mentions, numbers=_rounded(*numbers),
                source={"skill_gap_report": kind, "skill": g.skill, "importance_score": g.importance_score,
                        "transferable_candidate": g.transferable_candidate}))

    space = SPACE.get(contributions["output_space"], contributions["output_space"])
    model: list[EvidenceItem] = []
    for c in contributions["positive"] + contributions["negative"]:
        feature, shap = c["feature"], c["shap"]
        label = LABELS[feature]
        words, numbers = describe_value(feature, features.get(feature))
        verb = "raised" if shap > 0 else "lowered"
        model.append(EvidenceItem(
            id=f"M{len(model) + 1}", kind="shap", sign=1 if shap > 0 else -1,
            fact=f"{label[0].upper()}{label[1:]} ({words}) {verb} the score (SHAP {shap:+.3f} {space})",
            anchors=[label], numbers=_rounded(*numbers, shap, abs(shap), c["share"], pct(c["share"])),
            source={"feature": feature, "value": features.get(feature), "shap": shap, "share": c["share"],
                    "output_space": contributions["output_space"], "tied_with": c.get("tied_with", [])}))

    # "How to Improve": required gaps by importance, then preferred (the report already orders each list).
    improve = [g.id for g in gaps if g.kind == "missing_required"] + [g.id for g in gaps if g.kind == "missing_preferred"]
    skills = set(report.matched_skills) | {g.source["skill"] for g in gaps}
    skills |= set(student.skills if student is not None else ())
    skills |= {s for s, _ in job.skills} if job is not None else set()
    skills |= {i.anchors[0] for i in positive if i.kind == "transferable_skill"}
    skills |= {m for g in gaps for m in g.mentions}
    shap_semantic = contributions["values"].get("semantic_similarity")
    notes = [report.note, "Positive evidence and gaps come from the profile, the posting and the skill-gap report; "
                          "model evidence comes from this prediction's SHAP values."]
    if contributions.get("skill_feature_caveat"):
        notes.append(contributions["skill_feature_caveat"])
    return EvidenceBundle(
        suitability_score=suitability, positive=positive, gaps=gaps, model=model, improve=improve,
        known_terms=sorted({*skills, *LABELS.values()}, key=lambda t: (-len(t), t.lower())),
        semantic={"percentile": features.get("semantic_similarity"), "raw_cosine": semantic_raw,
                  "shap": shap_semantic, "output_space": contributions["output_space"],
                  "note": "percentile among all student-job pairs for the configured embedding model; the raw "
                          "cosine is model-specific"},
        notes=notes)
