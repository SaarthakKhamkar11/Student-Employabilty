"""Job Intelligence Agent (spec §5.3). Phase 2 shell: its logic is implemented in Phase 5.

Turns a job description into a structured, skill-normalised job representation,
keeping required and preferred skills separate because matching treats them differently.
"""

from typing import Any, ClassVar

from pydantic import BaseModel

from app.agents.base import GUARDRAILS, AgentContext, BaseAgent
from app.schemas.agent_io import CallTool, Finish, JobIntelligenceInput, JobIntelligenceOutput


class JobIntelligenceAgent(BaseAgent):
    name = "job_intelligence_agent"
    role_description = (
        "You are the Job Intelligence Agent. Turn a job description into a structured job: "
        "title, role, responsibilities, required skills, preferred skills, education and "
        "experience requirements, work type and location. Keep required and preferred skills "
        "strictly separate, normalise skills, flag likely duplicate jobs, and request a job "
        "embedding from the currently selected embedding model through a tool. "
        + GUARDRAILS
    )
    allowed_tools: ClassVar[list[str]] = []  # tools are registered in Phase 5
    planned_tools = (  # spec §20
        "parse_job_description",
        "extract_requirements",
        "extract_skills",
        "normalize_skills",
        "generate_job_embedding",
        "save_job",
    )
    input_schema = JobIntelligenceInput
    output_schema = JobIntelligenceOutput
    implemented = False
    implemented_in_phase = 5

    async def decide_next_action(self, ctx: AgentContext) -> CallTool | Finish:
        raise NotImplementedError(f"{self.name}: Phase 5")

    async def finalize(self, ctx: AgentContext) -> BaseModel | dict[str, Any]:
        raise NotImplementedError(f"{self.name}: Phase 5")
