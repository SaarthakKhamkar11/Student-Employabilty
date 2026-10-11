"""The matching stages as plain functions. The Matching Agent calls them through its tools
(app/tools/matching_tools.py); the evaluation and benchmark commands call them directly.

A "query" is the side being matched: a student (jobs_for_student) or a job (candidates_for_job). The
"candidates" are the other side. Features, skills and scores are always computed for (student, job).
"""

import hashlib
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.embeddings.provider import EmbeddingProvider
from app.matching import eligibility as elig
from app.matching.calibration import Calibration, cached_calibration
from app.matching.config import MatchingConfig
from app.matching.entities import JobView, StudentView
from app.matching.feature_builder import FEATURE_VERSION, build_features
from app.matching.predictor import build_predictor
from app.matching.similarity import compute_signals, retrieve
from app.matching.skill_gap import SkillVectors, compare_skills, skill_gap
from app.schemas.agent_state import PredictionVersions


class MatchingInputError(LookupError):
    """The request can't be matched as asked (unknown id, or no structured profile/posting yet)."""


@lru_cache(maxsize=4)
def dataset_version(processed_dir: str) -> str:
    """Same format as the Phase 6 engine: the processed data the matching configuration was built on."""
    digest = lambda name: hashlib.sha256(Path(processed_dir, name).read_bytes()).hexdigest()[:12]  # noqa: E731
    return f"students:{digest('students_unified.csv')} jobs:{digest('jobs.csv')}"


@dataclass
class MatchingContext:
    repo: object  # MatchingRepository
    provider: EmbeddingProvider
    cfg: MatchingConfig
    processed_dir: str
    visibility_enabled: bool = True
    calibration_loader: object = cached_calibration
    matching_model_override: str | None = None  # the MATCHING_MODEL setting; None = config/matching_model.yaml
    predictor: object = None  # Stage 4's model (app/matching/predictor.py); built from the config flag when None
    _skill_vectors: SkillVectors | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.predictor is None:
            self.predictor = build_predictor(self.cfg, self.matching_model_override)

    def calibration(self) -> Calibration:
        version, _ = self.provider.describe()
        return self.calibration_loader(self.provider.model_name, version)

    def skill_vectors(self) -> SkillVectors:
        if self._skill_vectors is None:
            self._skill_vectors = SkillVectors(self.provider)
        return self._skill_vectors

    def versions(self) -> PredictionVersions:
        """Spec §24 metadata. Refused (ValueError) unless the active model was registered by an experiment and was
        built for these feature and embedding versions."""
        version, _ = self.provider.describe()
        self.predictor.check(version)
        return PredictionVersions(
            embedding_model=self.provider.model_name, embedding_version=version,
            matching_model=self.predictor.name, matching_model_version=self.predictor.version,
            feature_version=FEATURE_VERSION, dataset_version=dataset_version(self.processed_dir),
            experiment_id=self.predictor.experiment_id)

    # -- loading ---------------------------------------------------------------
    def student(self, student_id: str) -> tuple[StudentView, str | None]:
        found = self.repo.student(student_id)
        if found is None:
            raise MatchingInputError(f"student {student_id} has no structured profile yet (run the Profile Agent)")
        return found

    def job(self, job_id: str) -> JobView:
        found = self.repo.job(job_id)
        if found is None:
            raise MatchingInputError(f"job {job_id} has no structured posting yet (run the Job Intelligence Agent)")
        return found

    def pairs(self, query_kind: str, query_id: str, candidates: list[str]) -> dict[str, tuple[StudentView, JobView]]:
        """(student, job) for every candidate, read from the cached full lists (one query, not one per candidate)."""
        if query_kind == "student":
            student, pool = self.student(query_id)[0], {j.job_id: j for j in self.repo.jobs()}
            missing = [c for c in candidates if c not in pool]
            out = {c: (student, pool[c]) for c in candidates if c in pool}
        else:
            job, pool = self.job(query_id), {s.student_id: s for s, _ in self.repo.students()}
            missing = [c for c in candidates if c not in pool]
            out = {c: (pool[c], job) for c in candidates if c in pool}
        if missing:
            raise MatchingInputError(f"no structured record for candidate(s) {missing[:5]}")
        return out


# -- Stage 1 -------------------------------------------------------------------
UPLOAD = "job_description_upload"
CORPUS = {"all": "all stored postings", "dataset": "dataset postings only (no uploaded JDs)",
          "uploaded": "uploaded JDs only"}


def eligibility(ctx: MatchingContext, mode: str, student_id: str | None, job_id: str | None,
                job_source: str = "all") -> dict:
    corpus = CORPUS[job_source] if mode == "jobs_for_student" else "all stored students with a profile"
    if mode == "jobs_for_student":
        student, gender = ctx.student(student_id)
        jobs = [j for j in ctx.repo.jobs() if job_source == "all" or (j.source == UPLOAD) == (job_source == "uploaded")]
        result = elig.jobs_for_student(student, jobs, ctx.cfg, visibility_gender=gender,
                                       visibility_enabled=ctx.visibility_enabled)
        query_id, pair = student_id, None
    elif mode == "candidates_for_job":
        job = ctx.job(job_id)
        result = elig.students_for_job(job, [s for s, _ in ctx.repo.students()], ctx.cfg)
        query_id, pair = job_id, None
    else:  # one named pair: the rules are checked and reported, never used to drop the pair
        student, gender = ctx.student(student_id)
        job = ctx.job(job_id)
        failed = elig.failed_rules(student, job, ctx.cfg, visibility_gender=gender,
                                   visibility_enabled=ctx.visibility_enabled)
        result = elig.EligibilityResult(eligible=[job_id], tiers={job_id: elig.ANY}, checked=1,
                                        location_rule="not applied: one named job")
        query_id, pair = student_id, failed
    return {"mode": mode, "query_id": query_id, "eligible": result.eligible, "tiers": result.tiers,
            "excluded": result.excluded, "checked": result.checked, "location_rule": result.location_rule,
            "pair_failed_rules": pair, "corpus": corpus if mode != "pair" else "one named job"}


# -- Stage 2 -------------------------------------------------------------------
def retrieval(ctx: MatchingContext, query_kind: str, query_id: str, eligible: list[str], tiers: dict[str, str],
              top_n: int) -> dict:
    view = ctx.student(query_id)[0] if query_kind == "student" else ctx.job(query_id)
    result = retrieve(query_kind, view.texts, eligible, tiers, ctx.repo.index, ctx.provider, top_n)
    version, _ = ctx.provider.describe()
    views = ctx.pairs(query_kind, query_id, [r.entity_id for r in result.top])
    return {"query_id": query_id, "scored": result.scored, "unscored": result.unscored,
            "query_sections": result.query_sections, "embedding_model": ctx.provider.model_name,
            "embedding_version": version,
            "top": [{"id": r.entity_id, "score": round(r.score, 6), "sections": r.sections, "tier": r.tier,
                     "summary": summary(query_kind, *views[r.entity_id])} for r in result.top]}


def summary(query_kind: str, student: StudentView, job: JobView) -> dict[str, str | None]:
    """What a person needs to recognise the candidate: never PII (no names, contacts or free text)."""
    if query_kind == "student":
        return {"job_title": job.title, "role": job.role, "country": job.country}
    return {"branch": student.branch, "domain": student.domain, "degree": student.degree_code}


# -- similarity signals (spec §5.4: embedding similarity) ------------------------
def similarity(ctx: MatchingContext, query_kind: str, query_id: str, candidates: list[str],
               retrieval_scores: dict[str, float]) -> dict:
    cal = ctx.calibration()
    pairs = ctx.pairs(query_kind, query_id, candidates)
    if query_kind == "student":
        query_texts = ctx.student(query_id)[0].texts
        candidate_texts = {c: job.texts for c, (_, job) in pairs.items()}
    else:
        query_texts = ctx.job(query_id).texts
        candidate_texts = {c: student.texts for c, (student, _) in pairs.items()}
    raw = compute_signals(query_kind, query_texts, candidate_texts, ctx.repo.index, ctx.provider)
    percentiles = {}
    for c in candidates:
        raw[c]["semantic_similarity"] = retrieval_scores.get(c)
        percentiles[c] = {name: cal.percentile(name, value) for name, value in raw[c].items()}
    return {"raw": raw, "percentiles": percentiles, "calibration": f"{cal.model_name} {cal.model_version}"}


# -- Stage 3 -------------------------------------------------------------------
def comparison(ctx: MatchingContext, student: StudentView, job: JobView):
    return compare_skills(student.skills, job.skills, ctx.skill_vectors(), ctx.calibration().transferable_threshold)


def features(ctx: MatchingContext, query_kind: str, query_id: str, candidates: list[str],
             percentiles: dict[str, dict[str, float | None]]) -> dict:
    out = {}
    for c, (student, job) in ctx.pairs(query_kind, query_id, candidates).items():
        out[c] = build_features(student, job, comparison(ctx, student, job), percentiles.get(c, {}), ctx.cfg)
    return {"feature_version": FEATURE_VERSION, "features": out,
            "transferable_threshold": ctx.calibration().transferable_threshold}


# -- Stage 4 -------------------------------------------------------------------
def prediction(ctx: MatchingContext, feature_vectors: dict[str, dict[str, float | None]], *,
               with_versions: bool = True) -> dict:
    """Refuses (ValueError) to predict without complete spec §24 metadata: see MatchingContext.versions().
    with_versions=False only for the evaluation that registers a configuration (app/matching/evaluate.py):
    its scores are metrics, never returned as predictions."""
    model = ctx.predictor
    out = {"matching_model": model.name, "matching_model_version": model.version, "method": model.method,
           "weights": model.weights, "predictions": model.predict_many(feature_vectors)}
    if with_versions:
        out["model_versions"] = ctx.versions().model_dump()
    return out


# -- Stage 5 -------------------------------------------------------------------
def ranking(suitability: dict[str, float | None], top_k: int, groups: dict[str, str] | None = None,
            max_per_group: int | None = None) -> dict:
    """Highest suitability first; ties broken by id so the order never depends on input order. With groups
    (a job's role), at most max_per_group of one group are returned; the rest are counted, not shown."""
    order = sorted(suitability, key=lambda c: (suitability[c] is None, -(suitability[c] or 0.0), c))
    kept, seen, collapsed = [], {}, 0
    for c in order:
        group = (groups or {}).get(c)
        if max_per_group and group is not None and seen.get(group, 0) >= max_per_group:
            collapsed += 1
            continue
        seen[group] = seen.get(group, 0) + 1
        kept.append(c)
    return {"ranked": [{"id": c, "rank": i, "suitability": suitability[c]} for i, c in enumerate(kept[:top_k], start=1)],
            "collapsed": collapsed}


def role_groups(retrieved_top: list[dict]) -> dict[str, str]:
    """A job's group for the per-role cap: its role, else its title (uploaded JDs may have no role)."""
    return {r["id"]: (r["summary"].get("role") or r["summary"].get("job_title") or r["id"]).strip().lower()
            for r in retrieved_top if "job_title" in r["summary"]}


# -- Stage 6 -------------------------------------------------------------------
def skill_gaps(ctx: MatchingContext, query_kind: str, query_id: str, candidates: list[str]) -> dict:
    out = {}
    for c, (student, job) in ctx.pairs(query_kind, query_id, candidates).items():
        out[c] = skill_gap(comparison(ctx, student, job), job.skills)
    return {"gaps": out}


def run_pipeline(ctx: MatchingContext, mode: str, student_id: str | None, job_id: str | None, top_k: int,
                 with_versions: bool = True, job_source: str = "all") -> dict:
    """All stages in order, with per-stage timings: the benchmark and evaluation path (no agent loop)."""
    timings, t = {}, time.perf_counter()

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        timings[name] = round((now - t) * 1000, 2)
        t = now

    kind = "job" if mode == "candidates_for_job" else "student"
    query_id = job_id if kind == "job" else student_id
    stage1 = eligibility(ctx, mode, student_id, job_id, job_source)
    lap("eligibility")
    stage2 = retrieval(ctx, kind, query_id, stage1["eligible"], stage1["tiers"], 1 if mode == "pair" else ctx.cfg.top_n)
    lap("retrieval")
    ids = [r["id"] for r in stage2["top"]]
    sims = similarity(ctx, kind, query_id, ids, {r["id"]: r["score"] for r in stage2["top"]})
    lap("similarity")
    feats = features(ctx, kind, query_id, ids, sims["percentiles"])
    lap("features")
    preds = prediction(ctx, feats["features"], with_versions=with_versions)
    lap("prediction")
    ranked = ranking({c: p["suitability"] for c, p in preds["predictions"].items()}, top_k,
                     role_groups(stage2["top"]), ctx.cfg.ranking.get("max_per_role"))
    lap("ranking")
    gaps = skill_gaps(ctx, kind, query_id, [r["id"] for r in ranked["ranked"]])
    lap("skill_gaps")
    return {"eligibility": stage1, "retrieval": stage2, "similarity": sims, "features": feats,
            "prediction": preds, "ranked": ranked["ranked"], "collapsed": ranked["collapsed"], "gaps": gaps,
            "timings_ms": timings}
