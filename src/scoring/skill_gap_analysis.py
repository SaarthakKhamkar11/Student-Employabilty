"""Skill-gap analysis for one (student, job) pair (Phase 9; spec §5.4 Stage 6; docs/adr/0016).

The Matching Agent's `identify_skill_gaps` tool wraps this module; the logic lives here so it can be tested and
run on its own:   python -m src.scoring.skill_gap_analysis --student S00002 --job J0016561 [--model lightgbm]

Output: the SkillGapReport contract (app/schemas/agent_io.py), which the Learning Agent (Phase 10) consumes.

Method
1. Matched / missing. The student's skills (Profile Agent, Phase 4) and the job's skills (Job Intelligence Agent,
   Phase 5) are canonical names from the ONE shared normaliser, app/profile/skills.py `normalize_skills` (alias
   table), and are compared by its `skill_key`. Raw skill lists go through the same normaliser (`from_raw_skills`);
   there is no second normaliser here. A job skill whose key the profile doesn't have is MISSING, which means "not
   detected in the profile", never "the student lacks it" (spec §5.2). Missing skills are split by the job's
   INFERRED requirement (required / preferred, docs/adr/0008).
2. Importance by counterfactual re-scoring. For each missing skill s, the student's skill list with s added is run
   again through Stage 3 (`build_features`: the skill comparison, including Phase 7's calibrated relatedness, and
   every skill feature recomputed) and Stage 4 (the ACTIVE matching model, `predict_suitability`'s predictor, chosen
   by config/matching_model.yaml):
       importance(s) = suitability(profile + s) - suitability(profile)        (0-1 suitability scale)
   All variants of a pair are scored in one batch by the same model. The text-similarity features
   (semantic_similarity, project_relevance, ...) keep their actual values: the counterfactual is "this skill is
   detected in the skill list", not "the profile text was rewritten". No importance is a hand-set number: it is
   whatever the model returns, and the report names the model (`importance_model`). None = the model could not
   score the pair.
3. Transferable / partially related. For each missing skill, the student's possessed skill whose NAME is most
   similar (cosine, configured embedding model, skill-name vectors) is found. With similarity >= `transferable` it is
   the skill's transferable_candidate; with partial <= similarity < transferable the skill is listed in
   partially_related; below `partial` no relation is reported. Thresholds are per embedding model family, tuned on
   real output (config/skill_gap.yaml); a model without an entry gets no relation tiers rather than borrowed ones.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from app.matching.feature_builder import build_features
from app.matching.skill_gap import SkillVectors, compare_skills, evidenced
from app.matching.vectors import cosine
from app.profile.skills import normalize_skills, skill_key

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "skill_gap.yaml"
ACTUAL = "__as_is__"
METHOD = ("matched/missing: canonical skill names from the shared normaliser (app/profile/skills.py), compared by "
          "skill_key; missing = not detected in the profile; required/preferred = the job's inferred split (ADR 0008). "
          "importance_score: counterfactual re-scoring, suitability with the skill detected minus suitability as is, "
          "by the active matching model with text-similarity features held fixed. transferable / partially related: "
          "cosine similarity of skill names under the configured embedding model, thresholds from "
          "config/skill_gap.yaml (docs/adr/0016)")

Rescorer = Callable[[dict[str, tuple[str, ...]]], dict[str, float | None]]


@dataclass(frozen=True)
class Thresholds:
    transferable: float
    partial: float
    model_name: str
    tuned_on: str


def load_thresholds(model_name: str, path: Path = CONFIG_PATH) -> Thresholds | None:
    with open(path, encoding="utf-8") as fh:
        entry = (yaml.safe_load(fh).get("thresholds") or {}).get(model_name)
    if not entry:
        return None
    t = Thresholds(float(entry["transferable"]), float(entry["partial"]), model_name, str(entry.get("tuned_on", "")))
    if not 0 < t.partial < t.transferable <= 1:
        raise ValueError(f"{path}: need 0 < partial < transferable <= 1 for {model_name}")
    return t


def from_raw_skills(raw: list[str]) -> tuple[str, ...]:
    """Raw skill strings -> canonical names, through the Phase 4 normaliser (never a second one)."""
    return tuple(s.canonical for s in normalize_skills(raw))


def split_skills(student_skills: tuple[str, ...], job_skills: tuple[tuple[str, str], ...],
                 terms: frozenset[str] | None = None):
    """Matched = the same rule as the features (app/matching/skill_gap.compare_skills): the skill name, or every
    word of the job skill stated in the profile text."""
    have = {skill_key(s) for s in student_skills}
    ok = {n for n, _ in job_skills if skill_key(n) in have or evidenced(n, terms)}
    matched = [n for n, _ in job_skills if n in ok]
    missing = [(n, r) for n, r in job_skills if n not in ok]
    return matched, missing


def closest_skills(names: list[str], student_skills: tuple[str, ...],
                   vectors: SkillVectors | None) -> dict[str, tuple[str, float] | None]:
    """For each job skill name, the student's skill with the most similar name (ties: alphabetical)."""
    if vectors is None or not student_skills or not names:
        return {n: None for n in names}
    vecs = vectors.get(list(names) + list(student_skills))
    out = {}
    for n in names:
        scored = [(s, cosine(vecs[n], vecs[s], vectors.dimension)) for s in student_skills]
        scored = sorted(((s, sim) for s, sim in scored if sim is not None), key=lambda p: (-p[1], p[0]))
        out[n] = scored[0] if scored else None
    return out


def counterfactual_importance(student_skills: tuple[str, ...], missing: list[str],
                              rescore: Rescorer) -> dict[str, float | None]:
    variants = {ACTUAL: student_skills, **{s: student_skills + (s,) for s in missing}}
    scores = rescore(variants)
    base = scores.get(ACTUAL)
    return {s: (None if base is None or scores.get(s) is None else round(scores[s] - base, 6)) for s in missing}


def pipeline_rescorer(student, job, percentiles: dict[str, float | None], cfg, predictor,
                      vectors: SkillVectors | None, related_threshold: float | None) -> Rescorer:
    """Stage 3 + Stage 4 for skill-list variants of one pair: the same functions and model the Matching Agent uses."""

    def rescore(variants: dict[str, tuple[str, ...]]) -> dict[str, float | None]:
        vectors_by_variant = {}
        for key, skills in variants.items():
            variant = replace(student, skills=skills)
            # a counterfactual skill is in the skill list only: the profile text (terms) stays as it is
            comparison = compare_skills(variant.skills, job.skills, vectors, related_threshold, student.terms)
            vectors_by_variant[key] = build_features(variant, job, comparison, percentiles, cfg)
        return {k: p["suitability"] for k, p in predictor.predict_many(vectors_by_variant).items()}

    return rescore


def _gap(skill: str, importance: dict, closest: dict, t: Thresholds | None) -> dict:
    best = closest.get(skill)
    transferable = t is not None and best is not None and best[1] >= t.transferable
    return {"skill": skill, "importance_score": importance.get(skill),
            "transferable_candidate": best[0] if transferable else None,
            "transferable_similarity": round(best[1], 4) if transferable else None}


def _ranked(gaps: list[dict]) -> list[dict]:
    return sorted(gaps, key=lambda g: (g["importance_score"] is None, -(g["importance_score"] or 0.0), g["skill"]))


def analyse(student, job, *, rescore: Rescorer, vectors: SkillVectors | None, thresholds: Thresholds | None,
            importance_model: str, similarity_model: str) -> dict:
    """The SkillGapReport for one pair (a dict; validate with app.schemas.agent_io.SkillGapReport)."""
    matched, missing = split_skills(student.skills, job.skills, student.terms)
    names = [n for n, _ in missing]
    importance = counterfactual_importance(student.skills, names, rescore) if names else {}
    closest = closest_skills(names, student.skills, vectors if thresholds is not None else None)
    partial = []
    for n, r in missing:
        best = closest.get(n)
        if thresholds and best and thresholds.partial <= best[1] < thresholds.transferable:
            partial.append({"skill": n, "requirement": r, "related_student_skill": best[0],
                            "similarity": round(best[1], 4)})
    return {
        "matched_skills": matched,
        "missing_required": _ranked([_gap(n, importance, closest, thresholds) for n, r in missing if r == "required"]),
        "missing_preferred": _ranked([_gap(n, importance, closest, thresholds) for n, r in missing if r == "preferred"]),
        "partially_related": sorted(partial, key=lambda p: (-p["similarity"], p["skill"])),
        "importance_model": importance_model,
        "similarity_model": similarity_model,
        "thresholds": ({"transferable": thresholds.transferable, "partial": thresholds.partial} if thresholds else {}),
        "method": METHOD + ("" if thresholds else "; relation tiers Not yet evaluated: no thresholds for this "
                                                   "embedding model in config/skill_gap.yaml"),
    }


def main(argv=None) -> None:
    from app.core.config import get_settings
    from app.db.matching_repository import build_matching_repository
    from app.embeddings.provider import build_embedding_provider
    from app.matching import pipeline as P
    from app.schemas.agent_io import SkillGapReport
    from app.tools.matching_tools import build_context

    ap = argparse.ArgumentParser(description="Skill-gap report for one stored (student, job) pair")
    ap.add_argument("--student", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--model", default=None, help="matching model for importance (default: config/matching_model.yaml)")
    args = ap.parse_args(argv)
    settings = get_settings()
    ctx = build_context(build_matching_repository(settings), build_embedding_provider(settings), settings,
                        matching_model_override=args.model)
    stage2 = P.retrieval(ctx, "student", args.student, [args.job], {}, 1)
    sims = P.similarity(ctx, "student", args.student, [args.job], {r["id"]: r["score"] for r in stage2["top"]})
    report = P.skill_gaps(ctx, "student", args.student, [args.job], sims["percentiles"])["gaps"][args.job]
    print(json.dumps(SkillGapReport.model_validate(report).model_dump(), indent=1))


if __name__ == "__main__":
    main()
