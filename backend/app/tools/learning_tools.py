"""Learning Agent tools (spec §17, §20; docs/adr/0018): one tool per stage, validated input and output.

    prioritize_gaps               Phase 9's SkillGapReport -> gaps by importance (ties kept as ties)
    retrieve_courses              catalogue courses + YouTube Data API results for each skill (real rows only)
    compute_course_similarity     cosine(skill, resource text) under the configured embedding model
    rank_courses                  per skill: relevance order, threshold, "No resources found" when none qualifies
    explain_recommendations       one evidence-only explanation per skill
    save_learning_recommendations learning_recommendations rows (a database failure never fails the request)
"""

from functools import lru_cache
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import REPO_ROOT, Settings
from app.embeddings.provider import EmbeddingProvider
from app.learning import ranking as R
from app.learning.youtube import YouTubeUnavailable, search_videos
from app.schemas.agent_io import LearningRecommendation, SkillGapReport
from app.tools.registry import ToolInputRejected, ToolRegistry

CONFIG_PATH = REPO_ROOT / "config" / "learning.yaml"


@lru_cache(maxsize=1)
def learning_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def min_relevance(model_name: str, cfg: dict | None = None) -> float | None:
    return ((cfg or learning_config())["ranking"]["min_relevance"] or {}).get(model_name)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Gap(_Model):
    skill: str
    requirement: Literal["required", "preferred"]
    importance_score: float | None
    transferable_candidate: str | None
    transferable_similarity: float | None
    priority: int
    tied_with: list[str]


class PrioritizeIn(_Model):
    skill_gap_report: SkillGapReport


class PrioritizeOut(_Model):
    gaps: list[Gap]
    importance_model: str


class Resource(_Model):
    resource_id: str
    title: str
    url: str
    provider: str | None
    description: str | None
    skills: list[str]
    difficulty: str | None
    prerequisites: str | None
    estimated_effort: str | None
    source_reference: str
    kind: Literal["course", "video"]


class RetrieveIn(_Model):
    skills: list[str] = Field(min_length=1)


class RetrieveOut(_Model):
    resources: dict[str, list[Resource]]
    catalog_size: int
    youtube: dict[str, str]  # skill -> "searched (n videos)" / "cached (n videos)" / why not


class SimilarityIn(_Model):
    skills: list[str]
    resources: dict[str, list[Resource]]


class SimilarityOut(_Model):
    scores: dict[str, dict[str, float | None]]
    embedding: str


class RankIn(_Model):
    gaps: list[Gap]
    resources: dict[str, list[Resource]]
    scores: dict[str, dict[str, float | None]]


class RankOut(_Model):
    ranked: list[dict[str, Any]]
    min_relevance: float
    top_k: int


class ExplainIn(_Model):
    ranked: list[dict[str, Any]]
    job_label: str
    importance_model: str
    embedding: str
    min_relevance: float


class ExplainOut(_Model):
    recommendations: list[LearningRecommendation]


class SaveIn(_Model):
    student_id: str
    job_id: str
    recommendations: list[LearningRecommendation]


class SaveOut(_Model):
    saved: bool
    rows: int
    reason: str | None = None


def register_learning_tools(registry: ToolRegistry, *, repo, embedder: EmbeddingProvider, settings: Settings,
                            youtube_search=search_videos) -> None:

    @registry.register("prioritize_gaps", input_schema=PrioritizeIn, output_schema=PrioritizeOut)
    def tool_prioritize_gaps(skill_gap_report):
        """Order Phase 9's missing skills by importance_score (descending); equal scores share a priority."""
        report = skill_gap_report.model_dump() if hasattr(skill_gap_report, "model_dump") else skill_gap_report
        gaps = R.prioritise(report)
        if not gaps:
            raise ToolInputRejected("prioritize_gaps", "the skill-gap report lists no missing skill")
        return {"gaps": gaps, "importance_model": report.get("importance_model") or "not stated"}

    @registry.register("retrieve_courses", input_schema=RetrieveIn, output_schema=RetrieveOut)
    def tool_retrieve_courses(skills):
        """Candidates for each skill: every catalogue course, plus YouTube Data API results (cached per query)."""
        cfg = learning_config()["youtube"]
        courses = repo.courses()
        out, status = {}, {}
        key = settings.youtube_api_key.get_secret_value() if settings.youtube_api_key else ""
        searched = 0
        for skill in skills:
            videos: list[dict] = []
            query = cfg["query"].format(skill=skill)
            if not cfg["enabled"]:
                status[skill] = "not searched: YouTube disabled in config/learning.yaml"
            else:
                cached = repo.cached_search(query, cfg["cache_days"])
                if cached is not None:
                    videos, status[skill] = cached, f"cached search ({len(cached)} videos)"
                elif searched >= cfg["max_searches_per_request"]:
                    status[skill] = f"not searched: quota guard ({cfg['max_searches_per_request']} searches per request)"
                else:
                    try:
                        videos, request = youtube_search(query, key, cfg["max_results"])
                        searched += 1
                        repo.store_search(query, videos, request)
                        status[skill] = f"searched ({len(videos)} videos)"
                    except YouTubeUnavailable as exc:
                        status[skill] = f"not searched: {exc}"
            out[skill] = courses + videos
        return {"resources": out, "catalog_size": len(courses), "youtube": status}

    @registry.register("compute_course_similarity", input_schema=SimilarityIn, output_schema=SimilarityOut)
    def tool_compute_course_similarity(skills, resources):
        """Cosine similarity of each skill name (query) and each candidate's text (passage), configured model."""
        rows = {s: [r.model_dump() if hasattr(r, "model_dump") else r for r in rs] for s, rs in resources.items()}
        version, _ = embedder.describe()
        return {"scores": R.relevance(embedder, skills, rows), "embedding": f"{embedder.model_name} {version}"}

    @registry.register("rank_courses", input_schema=RankIn, output_schema=RankOut)
    def tool_rank_courses(gaps, resources, scores):
        """Per skill: resources by relevance at or above the tuned threshold, top_k; none -> "No resources found"."""
        threshold = min_relevance(embedder.model_name)
        if threshold is None:
            raise ToolInputRejected("rank_courses", f"no min_relevance tuned for {embedder.model_name} "
                                                    "in config/learning.yaml")
        dump = lambda x: x.model_dump() if hasattr(x, "model_dump") else x  # noqa: E731
        top_k = int(learning_config()["ranking"]["top_k"])
        ranked = R.rank([dump(g) for g in gaps], {s: [dump(r) for r in rs] for s, rs in resources.items()},
                        scores, threshold, top_k)
        return {"ranked": ranked, "min_relevance": threshold, "top_k": top_k}

    @registry.register("explain_recommendations", input_schema=ExplainIn, output_schema=ExplainOut)
    def tool_explain_recommendations(ranked, job_label, importance_model, embedding, min_relevance):
        """One explanation per skill, every clause from the report, a stored row or a computed score."""
        recs = []
        for item in ranked:
            recs.append({
                "skill": item["skill"], "requirement": item["requirement"], "priority": item["priority"],
                "importance_score": item["importance_score"], "tied_with": item["tied_with"],
                "transferable_candidate": item.get("transferable_candidate"), "resources": item["resources"],
                "message": item["message"],
                "explanation": R.explain(item, job_label, importance_model, embedding, min_relevance),
            })
        return {"recommendations": recs}

    @registry.register("save_learning_recommendations", input_schema=SaveIn, output_schema=SaveOut)
    def tool_save_learning_recommendations(student_id, job_id, recommendations):
        """Store one row per recommended resource (resource_id NULL = none found). Never fails the request."""
        rows = []
        for rec in recommendations:
            rec = rec.model_dump() if hasattr(rec, "model_dump") else rec
            ids = [r["source"].split(" | ", 1)[0] for r in rec["resources"]] or [None]
            rows += [{"student_id": student_id, "job_id": job_id, "skill": rec["skill"], "resource_id": rid,
                      "priority": rec["priority"], "reason": rec["explanation"]} for rid in ids]
        try:
            return {"saved": True, "rows": repo.save_recommendations(rows)}
        except Exception as exc:  # noqa: BLE001 - ADR 0003: a database failure never breaks a request
            return {"saved": False, "rows": 0, "reason": f"{type(exc).__name__}"}
