"""Matching Agent (spec §5.4, §14), implemented in Phase 7.

Pipeline, each stage a validated tool call through the BaseAgent loop:

    filter_eligible → retrieve_jobs | retrieve_candidates → compute_similarity → generate_features
    → predict_suitability → rank_candidates → identify_skill_gaps

Modes: jobs_for_student, candidates_for_job, and "pair" (jobs_for_student with a job_id: one named job,
whose hard-rule failures are reported instead of applied). The agent is not a model: the embedding model
is whatever EMBEDDING_MODEL configures (Phase 6 has not selected one yet, docs/adr/0012), and the
prediction model is whatever config/matching_model.yaml names (docs/adr/0015): Baseline 3,
weighted_baseline_v1, until Phase 8's engine selects a trained model on human labels. Every prediction
carries the seven spec §24 version fields, or is not returned.
"""

from typing import Any, ClassVar

from pydantic import BaseModel

from app.agents.base import GUARDRAILS, AgentContext, BaseAgent
from app.matching.features import FEATURE_NAMES
from app.schemas.agent_io import CallTool, Finish, MatchingAgentInput, MatchingAgentOutput
from app.schemas.agent_state import INSUFFICIENT_EVIDENCE

PIPELINE = ("filter_eligible", "retrieve", "compute_similarity", "generate_features", "predict_suitability",
            "rank_candidates", "identify_skill_gaps")
NOTES = (
    "Similarity values are percentiles among all student-job pairs for the configured embedding model, "
    "never raw cosine (raw values are model-specific; Phase 6).",
    "Required vs preferred job skills are inferred, not ground truth (docs/adr/0008).",
    "A missing skill is 'not detected' in the profile, which is not proof the student lacks it.",
)


class MatchingAgent(BaseAgent):
    name = "matching_agent"
    role_description = (
        "You are the Matching Agent. Match students and jobs by calling tools in order: "
        "eligibility filtering, semantic retrieval, structured feature generation, "
        "suitability prediction, ranking and skill-gap identification. Use only the embedding "
        "and prediction models currently selected by the model-selection engines, and attach "
        "their version metadata to every result. You orchestrate the tools; you never score "
        "or rank anything yourself. "
        + GUARDRAILS
    )
    allowed_tools: ClassVar[list[str]] = [
        "filter_eligible", "retrieve_jobs", "retrieve_candidates", "compute_similarity", "generate_features",
        "predict_suitability", "rank_candidates", "identify_skill_gaps", "evaluate_model", "benchmark_matching",
    ]
    planned_tools = (  # spec §20
        "retrieve_candidates",
        "retrieve_jobs",
        "compute_similarity",
        "generate_features",
        "predict_suitability",
        "rank_candidates",
        "identify_skill_gaps",
        "evaluate_model",
    )
    # The features this agent builds, validated against protected attributes (app/matching/features.py).
    # The employer's stated preference is checked BEFORE scoring by the visibility filter; it is never a
    # feature (tests/test_fairness_guards.py scans this file for it).
    planned_features: ClassVar[tuple[str, ...]] = FEATURE_NAMES
    input_schema = MatchingAgentInput
    output_schema = MatchingAgentOutput
    implemented = True
    implemented_in_phase = 7

    def max_steps(self) -> int:
        return len(PIPELINE) + 2

    def max_tool_retries(self) -> int:
        return 1

    # -- decisions -------------------------------------------------------------
    async def decide_next_action(self, ctx: AgentContext) -> CallTool | Finish:
        x = self._init(ctx)
        for stage in PIPELINE:
            if stage in x["done"]:
                continue
            x["done"].add(stage)
            arguments = self._arguments(stage, ctx)
            if arguments is None:  # nothing left to match: stop early with what the funnel shows
                return Finish(reason=f"nothing to {stage}")
            tool = stage if stage != "retrieve" else ("retrieve_candidates" if x["kind"] == "job" else "retrieve_jobs")
            return CallTool(tool_name=tool, arguments=arguments)
        return Finish(reason="matching pipeline complete")

    async def finalize(self, ctx: AgentContext) -> MatchingAgentOutput:
        output = self._output(ctx)
        if not isinstance(output.model_versions, str):  # spec §24: the state records what produced the result
            ctx.state.model_versions.update(output.model_versions.model_dump())
        return output

    # -- helpers ---------------------------------------------------------------
    def _init(self, ctx: AgentContext) -> dict[str, Any]:
        x = ctx.extras
        if "mode" not in x:
            req: MatchingAgentInput = ctx.input
            pair = req.mode == "jobs_for_student" and req.job_id is not None
            x["mode"] = "pair" if pair else req.mode
            x["kind"] = "job" if req.mode == "candidates_for_job" else "student"
            x["query_id"] = req.job_id if x["kind"] == "job" else req.student_id
            x["top_k"] = 1 if pair else req.top_k
            x["done"] = set()
        return x

    def _out(self, ctx: AgentContext, tool: str) -> BaseModel | None:
        for obs in reversed(ctx.observations):
            if obs.tool_name == tool:
                return obs.output
        return None

    def _arguments(self, stage: str, ctx: AgentContext) -> dict[str, Any] | None:
        req: MatchingAgentInput = ctx.input
        x = ctx.extras
        kind, qid = x["kind"], x["query_id"]
        if stage == "filter_eligible":
            return {"mode": x["mode"], "student_id": req.student_id, "job_id": req.job_id, "job_source": req.job_source}
        eligible = self._out(ctx, "filter_eligible")
        if stage == "retrieve":
            if not eligible.eligible:
                return None
            top_n = 1 if x["mode"] == "pair" else self._top_n()
            if kind == "job":
                return {"job_id": qid, "eligible": eligible.eligible, "top_n": top_n}
            return {"student_id": qid, "eligible": eligible.eligible, "tiers": eligible.tiers, "top_n": top_n}
        retrieved = self._out(ctx, "retrieve_candidates" if kind == "job" else "retrieve_jobs")
        candidates = [r.id for r in retrieved.top] if retrieved else []
        if not candidates:
            return None
        if stage == "compute_similarity":
            return {"query_kind": kind, "query_id": qid, "candidates": candidates,
                    "retrieval_scores": {r.id: r.score for r in retrieved.top}}
        if stage == "generate_features":
            return {"query_kind": kind, "query_id": qid, "candidates": candidates,
                    "percentiles": self._out(ctx, "compute_similarity").percentiles}
        if stage == "predict_suitability":
            return {"features": self._out(ctx, "generate_features").features}
        if stage == "rank_candidates":
            predictions = self._out(ctx, "predict_suitability").predictions
            groups = ({r.id: (r.summary.get("role") or r.summary.get("job_title") or r.id).strip().lower()
                       for r in retrieved.top} if kind == "student" else {})
            return {"suitability": {c: p.suitability for c, p in predictions.items()}, "top_k": x["top_k"],
                    "groups": groups}
        if stage == "identify_skill_gaps":
            return {"query_kind": kind, "query_id": qid,
                    "candidates": [r.id for r in self._out(ctx, "rank_candidates").ranked]}
        return None

    def _top_n(self) -> int:
        from app.matching.config import get_matching_config

        return get_matching_config().top_n

    def _output(self, ctx: AgentContext) -> MatchingAgentOutput:
        x = ctx.extras
        eligible = self._out(ctx, "filter_eligible")
        if eligible is None:
            return MatchingAgentOutput()
        retrieved = self._out(ctx, "retrieve_candidates" if x["kind"] == "job" else "retrieve_jobs")
        sims, feats = self._out(ctx, "compute_similarity"), self._out(ctx, "generate_features")
        preds, ranked, gaps = (self._out(ctx, t) for t in ("predict_suitability", "rank_candidates",
                                                          "identify_skill_gaps"))
        matches = []
        if ranked is not None and preds is not None:
            by_id = {r.id: (i, r) for i, r in enumerate(retrieved.top, start=1)}
            for item in ranked.ranked:
                c, p = item.id, preds.predictions[item.id]
                retrieval_rank, r = by_id[c]
                student_id, job_id = (x["query_id"], c) if x["kind"] == "student" else (c, x["query_id"])
                matches.append({
                    "student_id": student_id, "job_id": job_id, "rank": item.rank,
                    "suitability_score": p.suitability if p.suitability is not None else INSUFFICIENT_EVIDENCE,
                    "semantic_similarity": _or_sentinel(sims.percentiles[c].get("semantic_similarity")),
                    "employability": {"overall": p.overall, "dimensions": p.dimensions, "method": preds.method},
                    "skill_gap": gaps.gaps[c].model_dump() if gaps is not None and c in gaps.gaps else INSUFFICIENT_EVIDENCE,
                    "summary": {**r.summary, "location_tier": r.tier},
                    "trace": {
                        "eligibility": {"location_tier": r.tier, "location_rule": eligible.location_rule,
                                        "failed_rules": eligible.pair_failed_rules or []},
                        "retrieval": {"rank": retrieval_rank, "raw_score": r.score, "sections": r.sections,
                                      "embedding": f"{retrieved.embedding_model} {retrieved.embedding_version}"},
                        "similarity_raw": sims.raw[c], "features": feats.features[c], "normalised": p.normalised,
                        "weights": preds.weights, "contributions": p.contributions,
                        "weight_covered": p.weight_covered, "missing_features": p.missing_features,
                    },
                    "model_versions": preds.model_versions.model_dump(),
                })
        funnel = {"corpus": eligible.corpus, "checked": eligible.checked, "excluded": eligible.excluded,
                  "eligible": len(eligible.eligible),
                  "location_rule": eligible.location_rule, "retrieved": len(retrieved.top) if retrieved else 0,
                  "unscored": retrieved.unscored if retrieved else 0,
                  "collapsed_same_role": ranked.collapsed if ranked is not None else 0, "returned": len(matches)}
        pair = None
        if x["mode"] == "pair":
            failed = eligible.pair_failed_rules or []
            pair = {"eligible": not failed, "failed_rules": failed,
                    "note": "reported, not applied: this job was asked for by name"}
        return MatchingAgentOutput(
            mode=x["mode"], query_id=x["query_id"], matches=matches, funnel=funnel, pair_check=pair,
            model_versions=preds.model_versions if preds is not None else INSUFFICIENT_EVIDENCE,
            notes=[preds.method + ".", *NOTES] if preds is not None else list(NOTES))


def _or_sentinel(value: float | None) -> float | str:
    return INSUFFICIENT_EVIDENCE if value is None else value
