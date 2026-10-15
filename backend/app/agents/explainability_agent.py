"""Explainability Agent (spec §16), implemented in Phase 11 (docs/adr/0019).

Two separate pipelines, each a sequence of validated tool calls through the BaseAgent loop:

GLOBAL (the whole active model):
    get_global_explanation  -> importance ranking, SHAP distributions over the validation split, summary-plot data

MATCH (one student-job prediction):
    get_model_prediction -> get_shap_values -> get_feature_contributions -> generate_evidence
    -> generate_explanation -> validate_explanation
       (failed: generate_explanation again, strictly, with the errors -> validate_explanation)
    -> get_match_explanation  (the validated lines, or, after two failures or without an LLM, the structured
                               evidence itself: an unvalidated sentence is never returned)

The scores, SHAP values and skill lists come from Phases 7-9's deterministic tools. The LLM (Gemini) only restates
evidence items as prose, and every line it writes must cite and match the items it states.
"""

from typing import Any, ClassVar

from pydantic import BaseModel

from app.agents.base import GUARDRAILS, AgentContext, BaseAgent
from app.schemas.agent_io import CallTool, ExplainabilityAgentInput, ExplainabilityAgentOutput, Finish

MATCH_PIPELINE = ("get_model_prediction", "get_shap_values", "get_feature_contributions", "generate_evidence",
                  "generate_explanation", "validate_explanation", "get_match_explanation")
GLOBAL_PIPELINE = ("get_global_explanation",)
MAX_GENERATIONS = 2  # the first draft plus one stricter retry


class ExplainabilityAgent(BaseAgent):
    name = "explainability_agent"
    role_description = (
        "You are the Explainability Agent. Explain a match using only the evidence you are "
        "given or retrieve through tools: the prediction, feature values, SHAP or other "
        "feature contributions, matched and missing skills, semantic similarity and model "
        "versions. Separate positive evidence, gaps, model evidence and improvement "
        "suggestions. Every statement must trace to that evidence; if evidence is missing, "
        "say 'Insufficient evidence'. "
        + GUARDRAILS
    )
    allowed_tools: ClassVar[list[str]] = list(MATCH_PIPELINE + GLOBAL_PIPELINE)
    planned_tools = (  # spec §20
        "get_model_prediction",
        "get_shap_values",
        "get_feature_contributions",
        "generate_evidence",
        "validate_explanation",
    )
    input_schema = ExplainabilityAgentInput
    output_schema = ExplainabilityAgentOutput
    implemented = True
    implemented_in_phase = 11

    def trace_model_versions(self, data: dict) -> dict[str, str]:
        m, g = data.get("match_explanation"), data.get("global_explanation")
        if isinstance(m, dict):
            v = m["model_versions"]
            return {"matching": v["matching_model_version"], "embedding": f"{v['embedding_model']} {v['embedding_version']}"}
        if isinstance(g, dict):
            return {"matching": g["matching_model_version"]}
        return {}

    def max_steps(self) -> int:
        return len(MATCH_PIPELINE) + 2 * (MAX_GENERATIONS - 1) + 1

    def max_tool_retries(self) -> int:
        return 1

    # -- decisions -------------------------------------------------------------
    async def decide_next_action(self, ctx: AgentContext) -> CallTool | Finish:
        req: ExplainabilityAgentInput = ctx.input
        if req.mode == "global":
            if self._out(ctx, "get_global_explanation") is None:
                return CallTool(tool_name="get_global_explanation", arguments={})
            return Finish(reason="global explanation complete")
        return self._next_match_step(ctx)

    def _next_match_step(self, ctx: AgentContext) -> CallTool | Finish:
        req: ExplainabilityAgentInput = ctx.input
        prediction = self._out(ctx, "get_model_prediction")
        if prediction is None:
            return CallTool(tool_name="get_model_prediction", arguments={"match": req.match.model_dump()})
        shap = self._out(ctx, "get_shap_values")
        if shap is None:
            return CallTool(tool_name="get_shap_values", arguments={
                "features": prediction.features, "model_versions": prediction.model_versions.model_dump(),
                "suitability": prediction.suitability})
        contributions = self._out(ctx, "get_feature_contributions")
        if contributions is None:
            return CallTool(tool_name="get_feature_contributions", arguments={
                "values": shap.values, "output_space": shap.output_space, "matching_model": shap.matching_model})
        evidence = self._out(ctx, "generate_evidence")
        if evidence is None:
            return CallTool(tool_name="generate_evidence", arguments={
                "prediction": prediction.model_dump(), "skill_gap_report": req.match.skill_gap.model_dump(),
                "contributions": contributions.model_dump()})

        drafts = self._all(ctx, "generate_explanation")
        checks = self._all(ctx, "validate_explanation")
        if not drafts:
            return CallTool(tool_name="generate_explanation", arguments={"evidence": evidence.model_dump()})
        last = drafts[-1]
        # each draft is validated once; an attempt that returned no draft has nothing to validate
        checked = bool(checks) and checks[-1].attempt == len(drafts)
        if last.draft is not None and not checked:
            return CallTool(tool_name="validate_explanation", arguments={
                "evidence": evidence.model_dump(), "draft": last.draft.model_dump(), "attempt": len(drafts)})
        valid = last.draft is not None and checked and checks[-1].valid
        if not valid and last.llm is not None and len(drafts) < MAX_GENERATIONS:
            errors = checks[-1].errors if last.draft is not None else [last.reason or "no draft"]
            return CallTool(tool_name="generate_explanation", arguments={
                "evidence": evidence.model_dump(), "strict": True, "previous_errors": errors})
        if self._out(ctx, "get_match_explanation") is None:
            reason = None
            if not valid:
                if last.llm is None:
                    reason = last.reason
                elif checks:
                    reason = f"no draft passed validate_explanation in {len(drafts)} attempt(s)"
                else:
                    reason = f"the LLM returned no draft in {len(drafts)} attempt(s): {last.reason}"
                if last.llm is not None:  # the LLM was configured but its prose could not be used
                    self.record_error(ctx, "explanation_fallback", reason, tool_name="validate_explanation")
            return CallTool(tool_name="get_match_explanation", arguments={
                "prediction": prediction.model_dump(), "shap": shap.model_dump(), "evidence": evidence.model_dump(),
                "draft": last.draft.model_dump() if valid else None,
                "validations": [c.model_dump() for c in checks], "llm": last.llm,
                "llm_attempts": len(drafts) if last.llm is not None else 0, "fallback_reason": reason})
        return Finish(reason="match explanation complete")

    # -- output ----------------------------------------------------------------
    async def finalize(self, ctx: AgentContext) -> ExplainabilityAgentOutput:
        req: ExplainabilityAgentInput = ctx.input
        if req.mode == "global":
            found = self._out(ctx, "get_global_explanation")
            return ExplainabilityAgentOutput(mode="global", **({"global_explanation": found} if found else {}))
        found = self._out(ctx, "get_match_explanation")
        return ExplainabilityAgentOutput(mode="match", **({"match_explanation": found} if found else {}))

    def _out(self, ctx: AgentContext, tool: str) -> BaseModel | None:
        found = self._all(ctx, tool)
        return found[-1] if found else None

    @staticmethod
    def _all(ctx: AgentContext, tool: str) -> list[Any]:
        return [o.output for o in ctx.observations if o.tool_name == tool]
