"""Learning Agent (spec §17), implemented in Phase 10 (docs/adr/0018).

Student → Target Job → Required Skills → Student Skills → Skill Gaps   (Phase 9's SkillGapReport, consumed as is)
→ prioritize_gaps → retrieve_courses → compute_course_similarity → rank_courses → explain_recommendations
→ save_learning_recommendations

Each stage is a validated tool call through the BaseAgent loop. Resources come only from the course catalogue
(learning_resources, built from the owner's curated list) and the YouTube Data API: a skill with no resource above
the tuned relevance threshold gets "No resources found for this skill in the current catalog", never a filler.
"""

from typing import Any, ClassVar

from pydantic import BaseModel

from app.agents.base import GUARDRAILS, AgentContext, BaseAgent
from app.schemas.agent_io import CallTool, Finish, LearningAgentInput, LearningAgentOutput

PIPELINE = ("prioritize_gaps", "retrieve_courses", "compute_course_similarity", "rank_courses",
            "explain_recommendations", "save_learning_recommendations")


class LearningAgent(BaseAgent):
    name = "learning_agent"
    role_description = (
        "You are the Learning Agent. Given a target job, its required skills, the student's "
        "detected skills and the skill gaps, prioritise the gaps by importance and recommend "
        "learning resources retrieved from the course catalogue, considering relevance, "
        "prerequisites, difficulty and effort where available. Never invent a course, title, "
        "provider or URL. "
        + GUARDRAILS
    )
    allowed_tools: ClassVar[list[str]] = list(PIPELINE)
    planned_tools = (  # spec §20
        "retrieve_courses",
        "compute_course_similarity",
        "rank_courses",
    )
    input_schema = LearningAgentInput
    output_schema = LearningAgentOutput
    implemented = True
    implemented_in_phase = 10

    def trace_model_versions(self, data: dict) -> dict[str, str]:
        embedding = (data.get("ranking") or {}).get("embedding")
        return {"embedding": embedding} if embedding else {}

    def max_steps(self) -> int:
        return len(PIPELINE) + 2

    def max_tool_retries(self) -> int:
        return 1

    async def decide_next_action(self, ctx: AgentContext) -> CallTool | Finish:
        done = ctx.extras.setdefault("done", set())
        for stage in PIPELINE:
            if stage in done:
                continue
            done.add(stage)
            arguments = self._arguments(stage, ctx)
            if arguments is None:
                return Finish(reason=f"nothing to {stage}")
            return CallTool(tool_name=stage, arguments=arguments)
        return Finish(reason="learning pipeline complete")

    def _out(self, ctx: AgentContext, tool: str) -> BaseModel | None:
        for obs in reversed(ctx.observations):
            if obs.tool_name == tool:
                return obs.output
        return None

    def _arguments(self, stage: str, ctx: AgentContext) -> dict[str, Any] | None:
        req: LearningAgentInput = ctx.input
        if stage == "prioritize_gaps":
            if req.skill_gap_report is None:
                return None  # Phase 10 consumes Phase 9's report; it never recomputes gaps from names
            return {"skill_gap_report": req.skill_gap_report.model_dump()}
        gaps = self._out(ctx, "prioritize_gaps")
        if gaps is None:
            return None
        skills = [g.skill for g in gaps.gaps]
        retrieved = self._out(ctx, "retrieve_courses")
        if stage == "retrieve_courses":
            return {"skills": skills}
        resources = {s: [r.model_dump() for r in rs] for s, rs in retrieved.resources.items()}
        if stage == "compute_course_similarity":
            return {"skills": skills, "resources": resources}
        sims = self._out(ctx, "compute_course_similarity")
        if stage == "rank_courses":
            return {"gaps": [g.model_dump() for g in gaps.gaps], "resources": resources, "scores": sims.scores}
        ranked = self._out(ctx, "rank_courses")
        if stage == "explain_recommendations":
            label = f"{req.job_title} ({req.job_id})" if req.job_title else f"job {req.job_id}"
            return {"ranked": ranked.ranked, "job_label": label, "importance_model": gaps.importance_model,
                    "embedding": sims.embedding, "min_relevance": ranked.min_relevance}
        if stage == "save_learning_recommendations":
            recs = self._out(ctx, "explain_recommendations").recommendations
            return {"student_id": req.student_id, "job_id": req.job_id, "recommendations": [r.model_dump() for r in recs]}
        return None

    async def finalize(self, ctx: AgentContext) -> LearningAgentOutput:
        req: LearningAgentInput = ctx.input
        explained = self._out(ctx, "explain_recommendations")
        if explained is None:
            return LearningAgentOutput(student_id=req.student_id, job_id=req.job_id)
        retrieved, sims, ranked = (self._out(ctx, t) for t in ("retrieve_courses", "compute_course_similarity",
                                                                "rank_courses"))
        saved = self._out(ctx, "save_learning_recommendations")
        resources = [r for rs in retrieved.resources.values() for r in rs]
        unique = {r.resource_id: r for r in resources}.values()
        avail = lambda f: sum(1 for r in unique if getattr(r, f))  # noqa: E731
        return LearningAgentOutput(
            student_id=req.student_id, job_id=req.job_id, recommendations=explained.recommendations,
            sources={"catalog_courses": retrieved.catalog_size,
                     "catalog": "owner-curated Coursera list (data/learning/curated_courses.csv), fields from each "
                                "course page (data/learning/course_catalog.json)",
                     "youtube": retrieved.youtube},
            ranking={"embedding": sims.embedding, "min_relevance": ranked.min_relevance, "top_k": ranked.top_k,
                     "order": "skills by Phase 9 importance_score (ties share a priority); resources by relevance",
                     "factor_availability": {
                         "relevance": "computed for every candidate",
                         "skill_importance": "from the SkillGapReport",
                         "difficulty": f"stated by the source for {avail('difficulty')} of {len(unique)} candidates",
                         "prerequisites": f"stated for {avail('prerequisites')} of {len(unique)}",
                         "estimated_effort": f"stated for {avail('estimated_effort')} of {len(unique)}"}},
            saved=bool(saved and saved.saved))
