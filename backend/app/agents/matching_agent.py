"""Matching Agent (spec §5.4). Phase 2 shell: its logic is implemented in Phase 7.

Orchestrates retrieval, similarity, feature engineering, prediction, ranking and
skill-gap tools. The agent is not a model: which embedding and prediction models it
uses is decided by the Phase 6 and Phase 8 experiments, not here (spec §41).
"""

from typing import Any, ClassVar

from pydantic import BaseModel

from app.agents.base import GUARDRAILS, AgentContext, BaseAgent
from app.matching.features import FEATURE_NAMES
from app.schemas.agent_io import CallTool, Finish, MatchingAgentInput, MatchingAgentOutput


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
    allowed_tools: ClassVar[list[str]] = []  # tools are registered in Phase 7
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
    # The features Phase 7 may build, validated against protected attributes (app/matching/features.py).
    # The employer's stated gender preference is checked BEFORE scoring by the check_job_visibility tool;
    # it is never a feature (tests/test_fairness_guards.py scans this file for it).
    planned_features: ClassVar[tuple[str, ...]] = FEATURE_NAMES
    input_schema = MatchingAgentInput
    output_schema = MatchingAgentOutput
    implemented = False
    implemented_in_phase = 7

    async def decide_next_action(self, ctx: AgentContext) -> CallTool | Finish:
        raise NotImplementedError(f"{self.name}: Phase 7")

    async def finalize(self, ctx: AgentContext) -> BaseModel | dict[str, Any]:
        raise NotImplementedError(f"{self.name}: Phase 7")
