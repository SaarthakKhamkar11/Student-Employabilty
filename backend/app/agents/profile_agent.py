"""Profile Agent (spec §5.2). Phase 2 shell: its logic is implemented in Phase 4.

Turns a resume or stored profile into a structured, skill-normalised student
representation with an embedding from the currently selected model.
"""

from typing import Any, ClassVar

from pydantic import BaseModel

from app.agents.base import GUARDRAILS, AgentContext, BaseAgent
from app.schemas.agent_io import CallTool, Finish, ProfileAgentInput, ProfileAgentOutput


class ProfileAgent(BaseAgent):
    name = "profile_agent"
    role_description = (
        "You are the Profile Agent. Turn a student's resume or profile into a structured "
        "representation: education, technical and soft skills, projects, internships and "
        "experience, certifications and career interests. Detect PII so it can be kept out of "
        "the ML pipeline, normalise skills and their aliases, and request an embedding from "
        "the currently selected embedding model through a tool. A skill you cannot find is "
        "'not detected', never 'not possessed'. "
        + GUARDRAILS
    )
    allowed_tools: ClassVar[list[str]] = []  # tools are registered in Phase 4
    planned_tools = (  # spec §20
        "parse_resume",
        "extract_profile",
        "extract_skills",
        "normalize_skills",
        "generate_embedding",
        "save_profile",
    )
    input_schema = ProfileAgentInput
    output_schema = ProfileAgentOutput
    implemented = False
    implemented_in_phase = 4

    async def decide_next_action(self, ctx: AgentContext) -> CallTool | Finish:
        raise NotImplementedError(f"{self.name}: Phase 4")

    async def finalize(self, ctx: AgentContext) -> BaseModel | dict[str, Any]:
        raise NotImplementedError(f"{self.name}: Phase 4")
