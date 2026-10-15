"""Cohort benchmarking (spec §18, Phase 12; docs/adr/0020; settings in config/cohort.yaml).

    benchmark(ctx, student_id, job_id=None) -> CohortBenchmark (as a dict)

Stage 1, filter. Peers with the student's branch, graduation year and target role. The target role is the job's
role family when a job is given (the job being evaluated against), else the student's own: the families of their
stated career interest, or of their domain when the interest is a plan such as "Placements" (role families and
domain -> family map: config/weak_label_rules.yaml, the ones Phase 6/7 use). A filter applies only when the
student's own value is known. If fewer than min_cohort_size peers remain, graduation year and then target role
are dropped (reported in `relaxed`); branch never is.

Stage 2, comparable students. Within that cohort, the top_n peers by cosine similarity of the stored profile-level
("full" section) embedding of the configured embedding model: the same vectors the Profile Agent stores.

Stage 3, statistics, over the comparable peers only:
- suitability: every peer and the student scored for the same job by the active matching model, through the
  Matching Agent's own Stages 2-4 (app/matching/pipeline.py). Hard eligibility rules are not applied: this
  compares suitability, not who could apply. Cohort average and median, and the student's percentile.
- skill coverage: the share of the cohort's most common skills the student has.
- relative skill gap: the student's missing job skills against how many peers miss each, and the student's missing
  count against theirs.

Privacy: only aggregates leave this module (counts, averages, percentiles, shares of peers). No peer id, score or
profile is returned or logged; statistics need at least min_cohort_size peers, and a common skill at least
min_skill_holders peers (tests/test_cohort_analytics.py checks the response against every real student_id).
"""

import argparse
import json
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "cohort.yaml"
ROLE_RULES_PATH = REPO_ROOT / "config" / "weak_label_rules.yaml"
_PLANS = {"placements", "masters", "higher studies"}  # career_interests that state a plan, not a field


@lru_cache(maxsize=2)
def load_config(path: str = str(CONFIG_PATH)) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def domain_families() -> dict[str, frozenset[str]]:
    """Each stated domain's PRIMARY role families (weak-label rules): what that student is aiming at."""
    with open(ROLE_RULES_PATH, encoding="utf-8") as fh:
        domains = yaml.safe_load(fh)["domains"]
    return {d: frozenset(v.get("primary") or ()) for d, v in domains.items()}


def student_target(view, mcfg) -> tuple[frozenset[str], str, str | None]:
    """(role families, source, label) of what a student is aiming at."""
    interest = view.career_interests
    if interest and interest.strip().lower() not in _PLANS:
        families = frozenset(mcfg.families_of_role(interest))
        if families:
            return families, "career_interests", interest
    families = domain_families().get(view.domain or "", frozenset())
    return families, "domain", view.domain


def benchmark(ctx, student_id: str, job_id: str | None = None, config: dict[str, Any] | None = None) -> dict:
    from app.matching import pipeline as P

    cfg = config or load_config()
    privacy = cfg["privacy"]
    student, _ = ctx.student(student_id)
    job = ctx.job(job_id) if job_id else None
    pool = [s for s, _ in ctx.repo.students() if s.student_id != student_id]

    # -- Stage 1: filter -----------------------------------------------------------------------------------------
    own_families, own_source, own_label = student_target(student, ctx.cfg)
    if job is not None and (job_families := frozenset(ctx.cfg.families_of_role(job.role, job.title))):
        families, source, label = job_families, "job", job.role or job.title
    else:
        families, source, label = own_families, own_source, own_label
    target = {"source": source, "label": label, "families": sorted(families)}
    if job is not None and source != "job":
        target["note"] = "the job's role matches no role family, so the student's own target role is used"

    def peer_matches(peer, name: str) -> bool:
        if name == "branch":
            return peer.branch == student.branch
        if name == "graduation_year":
            return peer.graduation_year == student.graduation_year
        return bool(student_target(peer, ctx.cfg)[0] & families)

    known = {"branch": student.branch is not None, "graduation_year": student.graduation_year is not None,
             "target_role": bool(families)}
    active = [n for n in ("branch", "graduation_year", "target_role") if known[n]]
    cohort = [p for p in pool if all(peer_matches(p, n) for n in active)]
    relaxed = []
    for name in cfg["filters"]["relax_order"]:
        if len(cohort) >= privacy["min_cohort_size"]:
            break
        if name in active:
            active.remove(name)
            relaxed.append(name)
            cohort = [p for p in pool if all(peer_matches(p, n) for n in active)]
    values = {"branch": student.branch, "graduation_year": None if student.graduation_year is None
              else str(student.graduation_year), "target_role": label}
    filters = [{"name": n, "value": values[n], "applied": n in active,
                "note": ("applied" if n in active else "dropped: the cohort was too small" if n in relaxed
                         else "not applied: not stated in the student's profile" if n != "target_role"
                         else "not applied: no role family for the student's interest or domain")}
               for n in ("branch", "graduation_year", "target_role")]
    version, dimension = ctx.provider.describe()
    section = cfg["comparable"]["section"]
    out: dict[str, Any] = {
        "student_id": student_id, "job_id": job_id, "method_version": cfg["version"], "target_role": target,
        "filters": filters, "relaxed": relaxed, "cohort_size": len(cohort), "comparable_size": 0,
        "similarity": {"embedding_model": ctx.provider.model_name, "embedding_version": version, "section": section,
                       "method": f"cosine of the stored '{section}' (profile-level) embedding, top "
                                 f"{cfg['comparable']['top_n']} within the filtered cohort"},
        "privacy": {"min_cohort_size": privacy["min_cohort_size"],
                    "min_skill_holders": privacy["min_skill_holders"],
                    "returned": "aggregates only: no other student's id, score or profile"},
        "notes": ["A missing skill is 'not detected' in the profile, which is not proof the student lacks it.",
                  "Hard eligibility rules are not applied: the comparison is of suitability, not of who may apply."],
    }
    if len(cohort) < privacy["min_cohort_size"]:
        out["notes"].insert(0, f"Only {len(cohort)} comparable peer(s) after filtering: fewer than "
                               f"{privacy['min_cohort_size']}, so no statistic is returned (privacy).")
        return _unavailable(out)

    # -- Stage 2: the N most similar peers by profile-level embedding --------------------------------------------
    text = student.texts.get(section)
    if not text:
        out["notes"].insert(0, f"The student's profile has no '{section}' text to embed, so comparable peers "
                               "can't be found.")
        return _unavailable(out)
    from app.embeddings.provider import QUERY

    query = ctx.provider.embed([text], QUERY)[0]  # the same role the Profile Agent stores student vectors with
    sims = ctx.repo.index.similarities("student", section, query, ctx.provider.model_name, version, dimension,
                                       ids=[p.student_id for p in cohort])
    ranked = sorted(sims.items(), key=lambda kv: (-kv[1], kv[0]))[:cfg["comparable"]["top_n"]]
    by_id = {p.student_id: p for p in cohort}
    comparable = [by_id[i] for i, _ in ranked]
    out["comparable_size"] = len(comparable)
    out["similarity"].update(peers_without_vector=len(cohort) - len(sims),
                             mean=round(statistics.fmean(v for _, v in ranked), 4) if ranked else None,
                             lowest=round(ranked[-1][1], 4) if ranked else None)
    if len(comparable) < privacy["min_cohort_size"]:
        out["notes"].insert(0, f"Only {len(comparable)} peer(s) have a stored embedding: fewer than "
                               f"{privacy['min_cohort_size']}, so no statistic is returned (privacy).")
        return _unavailable(out)

    # -- Stage 3: statistics over the comparable peers -----------------------------------------------------------
    out["skill_coverage"] = _coverage(student, comparable, cfg)
    if job is None:
        out["notes"].append("No target job was given: suitability and the relative skill gap need one.")
        return out
    ids = [student_id] + [p.student_id for p in comparable]
    stage2 = P.retrieval(ctx, "job", job_id, ids, {}, len(ids))
    scored_ids = [r["id"] for r in stage2["top"]]
    signals = P.similarity(ctx, "job", job_id, scored_ids, {r["id"]: r["score"] for r in stage2["top"]})
    feats = P.features(ctx, "job", job_id, scored_ids, signals["percentiles"])
    preds = P.prediction(ctx, feats["features"])  # refuses without complete spec §24 metadata
    scores = {i: p["suitability"] for i, p in preds["predictions"].items() if p["suitability"] is not None}
    out["model_versions"] = preds["model_versions"]
    peer_scores = [scores[p.student_id] for p in comparable if p.student_id in scores]
    if student_id not in scores or len(peer_scores) < privacy["min_cohort_size"]:
        out["notes"].insert(0, "The student or too few peers could be scored for this job: no suitability "
                               "statistics.")
    else:
        own = scores[student_id]
        out["suitability"] = {
            "job_id": job_id, "job_title": job.title, "student_score": round(own, 6),
            "cohort_average": round(statistics.fmean(peer_scores), 6),
            "cohort_median": round(statistics.median(peer_scores), 6),
            "percentile": _percentile_below(own, peer_scores), "peers_scored": len(peer_scores),
            "peers_unscored": len(comparable) - len(peer_scores)}
    out["relative_skill_gap"] = _relative_gap(ctx, student, comparable, job, cfg)
    return out


def _unavailable(out: dict) -> dict:
    for key in ("suitability", "skill_coverage", "relative_skill_gap"):
        out[key] = "Insufficient evidence"
    return out


def _percentile_below(value: float, others: list[float]) -> float:
    below = sum(1 for o in others if o < value)
    ties = sum(1 for o in others if o == value)
    return round(100 * (below + 0.5 * ties) / len(others), 1)


def _coverage(student, peers, cfg) -> dict:
    from app.profile.skills import skill_key

    holders: dict[str, int] = {}
    names: dict[str, str] = {}
    for p in peers:
        for skill in {skill_key(s): s for s in p.skills if skill_key(s)}.items():
            holders[skill[0]] = holders.get(skill[0], 0) + 1
            names.setdefault(skill[0], skill[1])
    common = sorted((k for k, n in holders.items() if n >= cfg["privacy"]["min_skill_holders"]),
                    key=lambda k: (-holders[k], k))[:cfg["statistics"]["common_skills"]]
    have = {skill_key(s) for s in student.skills}
    rows = [{"skill": names[k], "peers_with_share": round(holders[k] / len(peers), 3)} for k in common]
    return {"common_skills": rows,
            "coverage": round(sum(k in have for k in common) / len(common), 3) if common else "Insufficient evidence",
            "covered": [names[k] for k in common if k in have], "missing": [names[k] for k in common if k not in have]}


def _relative_gap(ctx, student, peers, job, cfg) -> dict:
    from app.matching import pipeline as P
    from app.profile.skills import skill_key

    def missing(view) -> set[str]:
        return {skill_key(name) for name, _ in P.comparison(ctx, view, job).missing}

    own = missing(student)
    peer_missing = [missing(p) for p in peers]
    names = {skill_key(n): n for n, _ in job.skills}
    share_with = {k: round(sum(k not in m for m in peer_missing) / len(peers), 3) for k in names}
    threshold = cfg["statistics"]["shared_gap_share"]
    row = lambda k: {"skill": names[k], "peers_with_share": share_with[k]}  # noqa: E731
    order = lambda keys: sorted(keys, key=lambda k: (-share_with[k], names[k]))  # noqa: E731
    counts = [len(m) for m in peer_missing]
    return {
        "job_id": job.job_id, "student_missing_count": len(own),
        "cohort_average_missing_count": round(statistics.fmean(counts), 3),
        "fewer_missing_than_percentile": _percentile_below(-len(own), [-c for c in counts]),
        "distinctive_gaps": [row(k) for k in order(k for k in own if share_with[k] > 1 - threshold)],
        "shared_gaps": [row(k) for k in order(k for k in own if share_with[k] <= 1 - threshold)],
        "advantages": [row(k) for k in order(k for k in names if k not in own and share_with[k] < threshold)],
    }


def main(argv=None) -> None:
    from app.core.config import get_settings
    from app.db.matching_repository import build_matching_repository
    from app.embeddings.provider import build_embedding_provider
    from app.matching import pipeline as P
    from app.schemas.agent_io import CohortBenchmark
    from app.tools.matching_tools import build_context

    ap = argparse.ArgumentParser(description="Cohort benchmark for one stored student (aggregates only)")
    ap.add_argument("--student", required=True)
    ap.add_argument("--job", default=None, help="target job (default with --top-job: the student's top match)")
    ap.add_argument("--top-job", action="store_true", help="benchmark against the student's top-ranked job")
    args = ap.parse_args(argv)
    settings = get_settings()
    ctx = build_context(build_matching_repository(settings), build_embedding_provider(settings), settings)
    job_id = args.job
    if job_id is None and args.top_job:
        ranked = P.run_pipeline(ctx, "jobs_for_student", args.student, None, 1)["ranked"]
        job_id = ranked[0]["id"] if ranked else None
    result = CohortBenchmark.model_validate(benchmark(ctx, args.student, job_id))
    print(json.dumps(result.model_dump(mode="json"), indent=1))


if __name__ == "__main__":
    main()
