"""Matching Agent tools (spec §5.4, §20): one tool per stage, each with validated input and output.

    filter_eligible       Stage 1  hard rules + location tiering + visibility (jobs for a student only)
    retrieve_jobs         Stage 2  semantic retrieval of jobs for a student (embedding similarity)
    retrieve_candidates   Stage 2  semantic retrieval of students for a job
    compute_similarity             per-pair similarity signals, calibrated to percentiles
    generate_features     Stage 3  the versioned feature vector (spec §11)
    predict_suitability   Stage 4  the model named by config/matching_model.yaml (Baseline 3 or a trained Phase 8 model)
    rank_candidates       Stage 5  deterministic ranking (ties by id)
    identify_skill_gaps   Stage 6  matched / related / missing / important missing / transferable
    evaluate_model                 ranking metrics of the current matching configuration on labelled pairs
    benchmark_matching             per-stage latency of the whole pipeline
"""

import statistics
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings
from app.db.matching_repository import MatchingRepository
from app.embeddings.provider import EmbeddingProvider
from app.matching import pipeline as P
from app.matching.config import get_matching_config
from app.schemas.agent_io import SkillGap
from app.schemas.agent_state import PredictionVersions
from app.tools.registry import ToolInputRejected, ToolRegistry
from src.evaluation.retrieval_metrics import evaluate

QueryKind = Literal["student", "job"]
Features = dict[str, float | None]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FilterIn(_Model):
    mode: Literal["jobs_for_student", "candidates_for_job", "pair"]
    student_id: str | None = None
    job_id: str | None = None
    job_source: Literal["all", "dataset", "uploaded"] = "all"


class FilterOut(_Model):
    mode: str
    query_id: str
    eligible: list[str]
    tiers: dict[str, str]
    excluded: dict[str, int]
    checked: int
    location_rule: str
    pair_failed_rules: list[str] | None
    corpus: str


class RetrieveJobsIn(_Model):
    student_id: str
    eligible: list[str]
    tiers: dict[str, str] = Field(default_factory=dict)
    top_n: int = Field(ge=1, le=500)


class RetrieveCandidatesIn(_Model):
    job_id: str
    eligible: list[str]
    top_n: int = Field(ge=1, le=500)


class Retrieved(_Model):
    id: str
    score: float
    sections: dict[str, float]
    tier: str
    summary: dict[str, str | None]


class RetrieveOut(_Model):
    query_id: str
    top: list[Retrieved]
    scored: int
    unscored: int
    query_sections: list[str]
    embedding_model: str
    embedding_version: str


class SimilarityIn(_Model):
    query_kind: QueryKind
    query_id: str
    candidates: list[str]
    retrieval_scores: dict[str, float]


class SimilarityOut(_Model):
    raw: dict[str, dict[str, float | None]]
    percentiles: dict[str, dict[str, float | None]]
    calibration: str


class FeaturesIn(_Model):
    query_kind: QueryKind
    query_id: str
    candidates: list[str]
    percentiles: dict[str, dict[str, float | None]]


class FeaturesOut(_Model):
    feature_version: str
    features: dict[str, Features]
    transferable_threshold: float


class PredictIn(_Model):
    features: dict[str, Features]


class Prediction(_Model):
    suitability: float | None
    overall: float | str
    normalised: Features
    contributions: dict[str, float]
    weight_covered: float
    missing_features: list[str]
    dimensions: dict[str, float | str]


class PredictOut(_Model):
    matching_model: str
    matching_model_version: str
    method: str
    weights: dict[str, float]
    model_versions: PredictionVersions  # strict: an unregistered configuration fails here, before any result
    predictions: dict[str, Prediction]


class RankIn(_Model):
    suitability: dict[str, float | None]
    top_k: int = Field(ge=1, le=100)
    groups: dict[str, str] = Field(default_factory=dict)  # jobs: candidate -> role, for the per-role cap


class Ranked(_Model):
    id: str
    rank: int
    suitability: float | None


class RankOut(_Model):
    ranked: list[Ranked]
    collapsed: int  # postings left out because their role was already shown (ranking.max_per_role)


class GapsIn(_Model):
    query_kind: QueryKind
    query_id: str
    candidates: list[str]


class GapsOut(_Model):
    gaps: dict[str, SkillGap]


class LabelledPair(_Model):
    student_id: str
    job_id: str
    relevance_label: int = Field(ge=0, le=3)


class EvaluateIn(_Model):
    pairs: list[LabelledPair] = Field(min_length=1)
    ks: list[int] = Field(default_factory=lambda: [5, 10])
    relevance_threshold: int = 2


class EvaluateOut(_Model):
    matching_model_version: str
    metrics: dict[str, float | None]
    per_student: dict[str, dict[str, float]]
    students_evaluated: int
    students_excluded_no_relevant: int
    pairs: int
    unscored_pairs: int


class BenchmarkIn(_Model):
    student_ids: list[str] = Field(min_length=1, max_length=200)
    top_k: int = Field(10, ge=1, le=100)


class BenchmarkOut(_Model):
    runs: int
    stage_ms: dict[str, dict[str, float]]  # stage -> {median, p95}
    total_ms: dict[str, float]


def build_context(repo: MatchingRepository | None, embedder: EmbeddingProvider, settings: Settings,
                  **overrides: Any) -> P.MatchingContext | None:
    if repo is None:
        return None
    return P.MatchingContext(repo=repo, provider=embedder, cfg=overrides.pop("cfg", None) or get_matching_config(),
                             processed_dir=str(settings.processed_data_dir),
                             matching_model_override=overrides.pop("matching_model_override", settings.matching_model),
                             visibility_enabled=settings.job_gender_visibility_filter, **overrides)


def score_labelled(ctx: P.MatchingContext, pairs: list[dict[str, Any]], ks=(5, 10), threshold: int = 2,
                   reference: dict | None = None) -> dict:
    """Stages 2-4 on each student's judged jobs only (condensed lists, as in Phase 6). Hard filters are a
    business rule, not a quality signal, so they are not applied here. `reference`, if given, also collects
    the raw retrieval similarity per pair (semantic similarity alone, for comparison)."""
    frame = pd.DataFrame(pairs)
    scores: dict[tuple[str, str], float] = {}
    for sid, group in frame.groupby("student_id"):
        jobs = sorted(set(group["job_id"]))
        stage2 = P.retrieval(ctx, "student", sid, jobs, {}, len(jobs))
        ids = [r["id"] for r in stage2["top"]]
        if reference is not None:
            reference.update({(sid, r["id"]): r["score"] for r in stage2["top"]})
        sims = P.similarity(ctx, "student", sid, ids, {r["id"]: r["score"] for r in stage2["top"]})
        preds = P.prediction(ctx, P.features(ctx, "student", sid, ids, sims["percentiles"])["features"],
                             with_versions=False)
        for jid, p in preds["predictions"].items():
            if p["suitability"] is not None:
                scores[(sid, jid)] = p["suitability"]
    result = evaluate(scores, frame.astype({"relevance_label": int}), tuple(ks), threshold)
    result["scores"] = scores
    return result


def register_matching_tools(registry: ToolRegistry, *, repo: MatchingRepository | None, embedder: EmbeddingProvider,
                            settings: Settings, **context_overrides: Any) -> None:
    holder: dict[str, P.MatchingContext | None] = {}

    def ctx() -> P.MatchingContext:
        if "ctx" not in holder:
            holder["ctx"] = build_context(repo, embedder, settings, **context_overrides)
        if holder["ctx"] is None:
            raise RuntimeError("no database configured (DATABASE_URL)")
        return holder["ctx"]

    def guarded(tool: str, fn, *args):
        try:
            return fn(*args)
        except P.MatchingInputError as exc:  # deterministic: retrying can't help
            raise ToolInputRejected(tool, str(exc)) from exc

    @registry.register("filter_eligible", input_schema=FilterIn, output_schema=FilterOut)
    def tool_filter_eligible(mode, student_id=None, job_id=None, job_source="all"):
        """Stage 1: remove only what clearly fails a hard requirement; count exclusions by reason."""
        return guarded("filter_eligible", P.eligibility, ctx(), mode, student_id, job_id, job_source)

    @registry.register("retrieve_jobs", input_schema=RetrieveJobsIn, output_schema=RetrieveOut)
    def tool_retrieve_jobs(student_id, eligible, tiers, top_n):
        """Stage 2: top-N eligible jobs by section-weighted embedding similarity (same country first)."""
        return guarded("retrieve_jobs", P.retrieval, ctx(), "student", student_id, eligible, tiers, top_n)

    @registry.register("retrieve_candidates", input_schema=RetrieveCandidatesIn, output_schema=RetrieveOut)
    def tool_retrieve_candidates(job_id, eligible, top_n):
        """Stage 2: top-N eligible students for a job by section-weighted embedding similarity."""
        return guarded("retrieve_candidates", P.retrieval, ctx(), "job", job_id, eligible, {}, top_n)

    @registry.register("compute_similarity", input_schema=SimilarityIn, output_schema=SimilarityOut)
    def tool_compute_similarity(query_kind, query_id, candidates, retrieval_scores):
        """Per-pair embedding similarity signals, raw and as calibrated percentiles."""
        return guarded("compute_similarity", P.similarity, ctx(), query_kind, query_id, candidates, retrieval_scores)

    @registry.register("generate_features", input_schema=FeaturesIn, output_schema=FeaturesOut)
    def tool_generate_features(query_kind, query_id, candidates, percentiles):
        """Stage 3: the versioned spec §11 feature vector for each (student, job) pair."""
        return guarded("generate_features", P.features, ctx(), query_kind, query_id, candidates, percentiles)

    @registry.register("predict_suitability", input_schema=PredictIn, output_schema=PredictOut)
    def tool_predict_suitability(features):
        """Stage 4: suitability from the model the config flag names (config/matching_model.yaml)."""
        try:
            return P.prediction(ctx(), features)
        except ValueError as exc:  # unregistered or mismatched model: deterministic, never retried
            raise ToolInputRejected("predict_suitability", str(exc)) from exc

    @registry.register("rank_candidates", input_schema=RankIn, output_schema=RankOut)
    def tool_rank_candidates(suitability, top_k, groups):
        """Stage 5: rank by predicted suitability; ties broken by id; at most max_per_role postings per role."""
        return P.ranking(suitability, top_k, groups or None, ctx().cfg.ranking.get("max_per_role") if groups else None)

    @registry.register("identify_skill_gaps", input_schema=GapsIn, output_schema=GapsOut)
    def tool_identify_skill_gaps(query_kind, query_id, candidates):
        """Stage 6: matched, partially related, missing, important missing and transferable skills."""
        return guarded("identify_skill_gaps", P.skill_gaps, ctx(), query_kind, query_id, candidates)

    @registry.register("evaluate_model", input_schema=EvaluateIn, output_schema=EvaluateOut)
    def tool_evaluate_model(pairs, ks, relevance_threshold):
        """Ranking metrics (MRR, NDCG@K, P@K, R@K, Hit Rate) of the current configuration on labelled pairs."""
        c = ctx()
        result = guarded("evaluate_model", score_labelled, c, pairs, ks, relevance_threshold)
        return {"matching_model_version": c.cfg.matching_model_version, "metrics": result["mean"],
                "per_student": result["per_student"], "students_evaluated": result["students_evaluated"],
                "students_excluded_no_relevant": result["students_excluded_no_relevant"],
                "pairs": result["pairs"], "unscored_pairs": result["unscored_pairs"]}

    @registry.register("benchmark_matching", input_schema=BenchmarkIn, output_schema=BenchmarkOut)
    def tool_benchmark_matching(student_ids, top_k):
        """Per-stage latency of the full pipeline (jobs for each given student)."""
        c = ctx()
        runs = [guarded("benchmark_matching", P.run_pipeline, c, "jobs_for_student", s, None, top_k, False)["timings_ms"]
                for s in student_ids]
        stages = {k: [r[k] for r in runs] for k in runs[0]}
        totals = [sum(r.values()) for r in runs]
        summary = lambda v: {"median": round(statistics.median(v), 2), "p95": round(float(np.percentile(v, 95)), 2)}  # noqa: E731
        return {"runs": len(runs), "stage_ms": {k: summary(v) for k, v in stages.items()}, "total_ms": summary(totals)}
