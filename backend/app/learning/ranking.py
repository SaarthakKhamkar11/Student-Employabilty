"""Phase 10 ranking (docs/adr/0018): which gaps first, which resources for each gap, and why.

Prioritise gaps   Phase 9's SkillGapReport, never recomputed: missing required + missing preferred, ordered by
                  importance_score (counterfactual re-scoring by the active matching model) descending. Equal
                  importances share a priority and are reported as tied (features-v1 sees skills as counts, ADR
                  0016), never presented as a ranking. None (the model couldn't score) comes last.
Relevance         cosine similarity of the skill name (role QUERY) and the resource text (role PASSAGE: title,
                  the source's skill list, description) under the configured embedding model: computed per pair.
Rank resources    by relevance, descending (ties: catalogue course before video, then title). Only resources with
                  relevance >= min_relevance (config/learning.yaml, tuned on real output) are recommended; if none
                  is, the skill gets NO_RESOURCES, never a filler.
Optional fields   difficulty, prerequisites, estimated effort: as the source states them, else NOT_AVAILABLE.
                  They are shown, not scored: the sources state them for some resources only, and no evidence says
                  which difficulty or length suits a student better.
"""

from app.embeddings.provider import PASSAGE, QUERY, EmbeddingProvider
from app.matching.vectors import cosine

NO_RESOURCES = "No resources found for this skill in the current catalog"
NOT_AVAILABLE = "Not available in source data"


def prioritise(report: dict) -> list[dict]:
    """[{skill, requirement, importance_score, priority, tied_with, transferable_candidate, ...}] best first."""
    gaps = ([{**g, "requirement": "required"} for g in report["missing_required"]]
            + [{**g, "requirement": "preferred"} for g in report["missing_preferred"]])
    key = lambda g: (g["importance_score"] is None, -(g["importance_score"] or 0.0), g["requirement"] != "required")  # noqa: E731
    gaps.sort(key=lambda g: (*key(g), g["skill"]))
    out, priority, last = [], 0, None
    for g in gaps:
        if key(g) != last:
            priority, last = priority + 1, key(g)
        out.append({**g, "priority": priority})
    for g in out:
        g["tied_with"] = [h["skill"] for h in out if h["priority"] == g["priority"] and h["skill"] != g["skill"]]
    return out


def resource_text(r: dict) -> str:
    parts = [r["title"]]
    if r.get("skills"):
        parts.append("Skills: " + ", ".join(r["skills"]))
    if r.get("description"):
        parts.append(r["description"])
    return ". ".join(parts)


def relevance(provider: EmbeddingProvider, skills: list[str], resources: dict[str, list[dict]]) -> dict:
    """{skill: {resource_id: cosine}} for each skill against its candidate resources."""
    version, dimension = provider.describe()
    skill_vecs = dict(zip(skills, provider.embed(skills, QUERY)))
    pool = {r["resource_id"]: r for rs in resources.values() for r in rs}
    ids = sorted(pool)
    res_vecs = dict(zip(ids, provider.embed([resource_text(pool[i]) for i in ids], PASSAGE)))
    out = {}
    for s in skills:
        out[s] = {r["resource_id"]: cosine(skill_vecs.get(s), res_vecs.get(r["resource_id"]), dimension)
                  for r in resources.get(s, [])}
    return out


def _show(value) -> str:
    return value if value else NOT_AVAILABLE


def rank(gaps: list[dict], resources: dict[str, list[dict]], scores: dict, min_relevance: float,
         top_k: int) -> list[dict]:
    out = []
    for g in gaps:
        pool = {r["resource_id"]: r for r in resources.get(g["skill"], [])}
        scored = [(rid, s) for rid, s in scores.get(g["skill"], {}).items() if s is not None]
        scored.sort(key=lambda p: (-p[1], pool[p[0]]["kind"] != "course", pool[p[0]]["title"]))
        best = scored[0][1] if scored else None
        kept = [(rid, s) for rid, s in scored if s >= min_relevance][:top_k]
        out.append({
            **g,
            "resources": [{"resource_name": pool[rid]["title"], "resource_url": pool[rid]["url"],
                           "source": f"{rid} | {pool[rid]['source_reference']}", "relevance_score": round(s, 4),
                           "kind": pool[rid]["kind"], "provider": pool[rid].get("provider") or NOT_AVAILABLE,
                           "difficulty": _show(pool[rid].get("difficulty")),
                           "prerequisites": _show(pool[rid].get("prerequisites")),
                           "estimated_effort": _show(pool[rid].get("estimated_effort"))} for rid, s in kept],
            "message": None if kept else NO_RESOURCES,
            "candidates_considered": len(scored), "best_candidate_relevance": round(best, 4) if best is not None else None,
        })
    return out


def explain(item: dict, job_label: str, importance_model: str, embedding: str, min_relevance: float) -> str:
    """One paragraph, every clause from the report, the catalogue row or the computed score."""
    imp = item["importance_score"]
    parts = [f"'{item['skill']}' is a {item['requirement']} skill for {job_label} that was not detected in the "
             f"profile (not proof it is lacking)."]
    if imp is None:
        parts.append("Its importance could not be scored by the matching model.")
    else:
        parts.append(f"Detecting it would change the predicted suitability by {imp:+.4f} ({importance_model}, "
                     "counterfactual re-scoring, Phase 9).")
    if item["tied_with"]:
        parts.append(f"Same importance as {', '.join(item['tied_with'])}: the model can't tell these apart, so "
                     "their order is not a ranking.")
    if item.get("transferable_candidate"):
        parts.append(f"Related skill already in the profile: {item['transferable_candidate']} "
                     f"(name similarity {item['transferable_similarity']}).")
    if item["resources"]:
        top = item["resources"][0]
        parts.append(f"Top resource: '{top['resource_name']}' ({top['kind']}, {top['provider']}), relevance "
                     f"{top['relevance_score']:.3f} to the skill under {embedding}; difficulty: {top['difficulty']}; "
                     f"effort: {top['estimated_effort']}.")
    else:
        best = item["best_candidate_relevance"]
        parts.append(f"{NO_RESOURCES}: {item['candidates_considered']} candidates checked, the best relevance "
                     f"{'n/a' if best is None else f'{best:.3f}'} is below the {min_relevance} threshold.")
    return " ".join(parts)
