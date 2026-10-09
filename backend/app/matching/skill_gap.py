"""Skill comparison for one student and one job: shared by the features (Stage 3) and the skill gap (Stage 6).

  matched   a job skill the student's profile names (same canonical name after the shared normaliser)
  related   a job skill the profile doesn't name, but one of the student's skills is related to it: the
            embedding similarity of the two names reaches the calibrated threshold (app/matching/calibration.py).
            The student's skill is then "transferable" to the job's.
  missing   neither. A missing skill is "not detected" in the profile, which is not proof the student
            lacks it (spec §5.2).
Required vs preferred comes from the job's INFERRED split (docs/adr/0008).
"""

from dataclasses import dataclass, field

from app.embeddings.provider import EmbeddingProvider, Vector
from app.matching.vectors import cosine
from app.profile.skills import skill_key


@dataclass
class SkillComparison:
    matched: list[tuple[str, str]] = field(default_factory=list)  # (job skill, requirement)
    related: list[tuple[str, str, str, float]] = field(default_factory=list)  # (job skill, req, student skill, sim)
    missing: list[tuple[str, str]] = field(default_factory=list)  # (job skill, requirement)

    def count(self, kind: str, requirement: str | None = None) -> int:
        rows = getattr(self, kind)
        return sum(1 for r in rows if requirement is None or r[1] == requirement)


class SkillVectors:
    """Embeddings of skill names, cached per model version (job skill names repeat across postings)."""

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider
        self.version, self.dimension = provider.describe()
        self._cache: dict[str, Vector | None] = {}

    def get(self, names: list[str]) -> dict[str, Vector | None]:
        todo = sorted({skill_key(n): n for n in names if skill_key(n) and skill_key(n) not in self._cache}.items())
        if todo:
            vectors = self.provider.embed([n for _, n in todo])
            self._cache.update({k: v for (k, _), v in zip(todo, vectors)})
        return {n: self._cache.get(skill_key(n)) for n in names}


def compare_skills(student_skills: tuple[str, ...], job_skills: tuple[tuple[str, str], ...],
                   vectors: SkillVectors | None, threshold: float | None) -> SkillComparison:
    """vectors/threshold None: relatedness can't be judged, so nothing is "related" (all unmatched are missing)."""
    have = {skill_key(s): s for s in student_skills}
    out = SkillComparison()
    unmatched = []
    for name, requirement in job_skills:
        if skill_key(name) in have:
            out.matched.append((name, requirement))
        else:
            unmatched.append((name, requirement))
    if unmatched and student_skills and vectors is not None and threshold is not None:
        vecs = vectors.get([n for n, _ in unmatched] + list(student_skills))
        for name, requirement in unmatched:
            best = max(((s, cosine(vecs[name], vecs[s], vectors.dimension)) for s in student_skills),
                       key=lambda p: (p[1] if p[1] is not None else -1.0, p[0]), default=(None, None))
            if best[1] is not None and best[1] >= threshold:
                out.related.append((name, requirement, best[0], round(best[1], 4)))
            else:
                out.missing.append((name, requirement))
    else:
        out.missing = unmatched
    return out


def skill_gap(comparison: SkillComparison, job_skills: tuple[tuple[str, str], ...]) -> dict:
    """The SkillGap contract (app/schemas/agent_io.py), consumed unchanged by the Learning Agent (Phase 10)."""
    return {
        "matched_skills": [n for n, _ in comparison.matched],
        "partially_related_skills": [n for n, _, _, _ in comparison.related],
        "missing_skills": [n for n, _ in comparison.missing],
        "important_missing_skills": [n for n, r in comparison.missing if r == "required"],
        "transferable_skills": sorted({s for _, _, s, _ in comparison.related}),
        "relations": [{"job_skill": n, "requirement": r, "student_skill": s, "similarity": sim}
                      for n, r, s, sim in comparison.related],
        "required_skills": [n for n, r in job_skills if r == "required"],
        "preferred_skills": [n for n, r in job_skills if r == "preferred"],
        "method": ("exact match on normalised skill names; related = embedding similarity of skill names at or "
                   "above the calibrated threshold; required/preferred is the job's inferred split (ADR 0008)"),
    }
