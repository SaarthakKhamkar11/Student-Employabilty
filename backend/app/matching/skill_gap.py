"""Skill comparison for one student and one job, as the features (Stage 3) use it. The Stage 6 report is built by
src/scoring/skill_gap_analysis.py (Phase 9), which uses this comparison for its counterfactual re-scoring.

  matched   a job skill the student's profile names (same canonical name after the shared normaliser)
  related   a job skill the profile doesn't name, but one of the student's skills is related to it: the
            embedding similarity of the two names reaches the calibrated threshold (app/matching/calibration.py).
            The student's skill is then "transferable" to the job's.
  missing   neither. A missing skill is "not detected" in the profile, which is not proof the student
            lacks it (spec §5.2).
Required vs preferred comes from the job's INFERRED split (docs/adr/0008).
"""

import re
from dataclasses import dataclass, field

from app.embeddings.provider import QUERY, EmbeddingProvider, Vector
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
            vectors = self.provider.embed([n for _, n in todo], QUERY)  # skill names: symmetric, both query
            self._cache.update({k: v for (k, _), v in zip(todo, vectors)})
        return {n: self._cache.get(skill_key(n)) for n in names}


# ----------------------------------------------------------------------------------- evidence in the profile text
# Job skills are mostly phrases ("Data analysis and visualization tools", "SQL and database querying"), student
# skills mostly names ("Power BI", "SQL"), so an exact name match almost never fires. A job skill is also MATCHED
# when every content word of the phrase occurs in the student's own profile text (skills, projects, experience,
# certifications): the profile states it, in other words. Words are compared by a spelling-tolerant stem
# (visualisation = visualization, modelling = modeling); filler words ("skills", "tools", "strong") don't count.
_FILLER = {
    "and", "or", "the", "of", "in", "with", "for", "to", "a", "an", "on", "at", "by", "as", "etc", "eg", "e", "g",
    "ie", "skills", "skill", "tools", "tool", "concepts", "concept", "knowledge", "strong", "good", "excellent",
    "ability", "abilities", "proficiency", "proficient", "experience", "experienced", "understanding", "familiarity",
    "familiar", "using", "use", "related", "techniques", "technique", "methods", "solid", "basic", "basics",
    "advanced", "working", "hands", "plus", "making", "based", "related", "practices", "fundamentals", "principles",
    "expertise", "level", "high", "effective", "relevant", "various", "other", "similar", "like", "such", "including",
    "software", "systems", "platforms", "applications", "framework", "frameworks", "management",
}
_WORD = re.compile(r"[a-z0-9][a-z0-9+#.]*[a-z0-9+#]|[a-z0-9]")


def _stem(word: str) -> str:
    w = word.replace("isation", "ization").replace("ising", "izing").replace("ise", "ize").replace("ll", "l")
    return w if len(w) <= 5 else w[:5]


def profile_terms(text: str) -> frozenset[str]:
    """The stems of every word in a profile text (computed once per student)."""
    return frozenset(_stem(w) for w in _WORD.findall(text.lower()))


def phrase_terms(name: str) -> frozenset[str]:
    return frozenset(_stem(w) for w in _WORD.findall(name.lower()) if w not in _FILLER)


def evidenced(name: str, terms: frozenset[str] | None) -> bool:
    """Every content word of the job skill occurs in the student's profile text."""
    if not terms:
        return False
    wanted = phrase_terms(name)
    return bool(wanted) and wanted <= terms


def compare_skills(student_skills: tuple[str, ...], job_skills: tuple[tuple[str, str], ...],
                   vectors: SkillVectors | None, threshold: float | None,
                   terms: frozenset[str] | None = None) -> SkillComparison:
    """vectors/threshold None: relatedness can't be judged, so nothing is "related" (all unmatched are missing).
    terms: profile_terms() of the student's profile text; a job skill all of whose words are there is matched."""
    have = {skill_key(s): s for s in student_skills}
    out = SkillComparison()
    unmatched = []
    for name, requirement in job_skills:
        if skill_key(name) in have or evidenced(name, terms):
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

