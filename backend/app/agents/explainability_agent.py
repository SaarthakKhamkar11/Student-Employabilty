"""Explainability Agent (spec §16). Phase 2 shell: its logic is implemented in Phase 11.

Converts model predictions, feature contributions (SHAP for tree models) and
skill evidence into human-readable explanations, where every statement is traceable
to evidence.
"""

from typing import Any, ClassVar

from pydantic import BaseModel

from app.agents.base import GUARDRAILS, AgentContext, BaseAgent
from app.schemas.agent_io import CallTool, Finish, ExplainabilityAgentInput, ExplainabilityAgentOutput


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
    allowed_tools: ClassVar[list[str]] = []  # tools are registered in Phase 11
    planned_tools = (  # spec §20
        "get_model_prediction",
        "get_shap_values",
        "get_feature_contributions",
        "generate_evidence",
        "validate_explanation",
    )
    input_schema = ExplainabilityAgentInput
    output_schema = ExplainabilityAgentOutput
    implemented = False
    implemented_in_phase = 11

    async def decide_next_action(self, ctx: AgentContext) -> CallTool | Finish:
        raise NotImplementedError(f"{self.name}: Phase 11")

    async def finalize(self, ctx: AgentContext) -> BaseModel | dict[str, Any]:
        raise NotImplementedError(f"{self.name}: Phase 11")
