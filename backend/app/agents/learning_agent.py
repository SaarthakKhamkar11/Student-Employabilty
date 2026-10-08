"""Learning Agent (spec §17). Phase 2 shell: its logic is implemented in Phase 10.

Prioritises a student's skill gaps for a target job and recommends learning
resources from the course catalogue only.
"""

from typing import Any, ClassVar

from pydantic import BaseModel

from app.agents.base import GUARDRAILS, AgentContext, BaseAgent
from app.schemas.agent_io import CallTool, Finish, LearningAgentInput, LearningAgentOutput


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
    allowed_tools: ClassVar[list[str]] = []  # tools are registered in Phase 10
    planned_tools = (  # spec §20
        "retrieve_courses",
        "compute_course_similarity",
        "rank_courses",
    )
    input_schema = LearningAgentInput
    output_schema = LearningAgentOutput
    implemented = False
    implemented_in_phase = 10

    async def decide_next_action(self, ctx: AgentContext) -> CallTool | Finish:
        raise NotImplementedError(f"{self.name}: Phase 10")

    async def finalize(self, ctx: AgentContext) -> BaseModel | dict[str, Any]:
        raise NotImplementedError(f"{self.name}: Phase 10")
