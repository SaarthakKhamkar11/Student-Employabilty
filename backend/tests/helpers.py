"""Test doubles for Supervisor tests: a fake LLM and minimal stub specialists.

The stubs exist only so routing, ordering, retries and aggregation can be
tested. They are not implementations of Phases 4-11. The fake LLM pushes its
canned responses through the same parse_structured() validation as the real
Gemini client.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.agents import SPECIALIST_CLASSES, SupervisorAgent
from app.agents.base import AgentContext
from app.db.store import InMemoryAgentRunStore
from app.llm import parse_structured
from app.schemas.agent_io import (
    AgentOutput, CallTool, ExplainabilityAgentOutput, Finish, JobIntelligenceOutput, LearningAgentOutput,
    MatchingAgentOutput, MatchResult, ProfileAgentOutput, SkillGap,
)
from app.schemas.agent_state import ModelVersionInfo
from app.schemas.job import JobRecord
from app.schemas.student import StudentRecord
from app.tools.registry import ToolRegistry

STUDENT = dict(student_id="S00001", branch="Computer Science and Engineering (Data Science)",
               career_interests="Data analysis", domain="Data Analyst and Machine Learning",
               source_dataset="placement_cell", is_synthetic=False)
JOB = dict(job_id="J0000001", source_job_id="1", experience_required="0 to 2 Years",
           qualification_required="B.Tech", salary_range="$50K-$70K", location="Mumbai", country="India",
           work_type="Intern", job_title="Data Analyst", role="Data Analyst", job_portal="Test",
           job_description="Analyse data.", skills="Python SQL Power BI", responsibilities="Reports",
           company_name="Example Co", gender_preference="both")

STUB_OUTPUTS: dict[str, BaseModel] = {
    "profile_agent": ProfileAgentOutput(student=StudentRecord(**STUDENT), normalized_skills=["python", "sql"]),
    "job_intelligence_agent": JobIntelligenceOutput(job=JobRecord(**JOB), required_skills=["python", "sql"],
                                                    preferred_skills=["power bi"]),
    "matching_agent": MatchingAgentOutput(matches=[MatchResult(
        student_id="S00001", job_id="J0000001", rank=1, model_versions=ModelVersionInfo(),
        skill_gap=SkillGap(matched_skills=["python", "sql"], partially_related_skills=[], missing_skills=["power bi"],
                           important_missing_skills=["power bi"], transferable_skills=[]))]),
    "learning_agent": LearningAgentOutput(),
    "explainability_agent": ExplainabilityAgentOutput(),
}


@dataclass
class Sleep:
    seconds: float


class FakeLLM:
    """Scripted LLM. Each call consumes the next response; the last one repeats."""

    model_name = "fake-gemini"

    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.calls: list[dict[str, str]] = []

    async def generate_structured(self, *, system: str, prompt: str, schema: type[BaseModel], timeout_s: float):
        self.calls.append({"system": system, "prompt": prompt})
        response = self.responses[min(len(self.calls), len(self.responses)) - 1]
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, Sleep):
            await asyncio.sleep(response.seconds)
            raise AssertionError("the router should have timed out first")
        return parse_structured(response if isinstance(response, str) else json.dumps(response), schema)


def stub_registry() -> ToolRegistry:
    registry = ToolRegistry()

    class Nothing(BaseModel):
        ok: bool

    @registry.register("always_fails", output_schema=Nothing)
    def always_fails():
        raise ConnectionError("simulated outage")

    return registry


def make_stub_class(base: type, behaviour: str, timeline: list, inputs: dict, delay: float):
    """behaviour: ok | transient | flaky | permanent | rogue."""
    runs = {"n": 0}

    class Stub(base):
        implemented = True
        allowed_tools = ["always_fails"] if behaviour in ("transient", "flaky") else []

        async def run(self, state, input_payload):
            runs["n"] += 1
            if behaviour == "rogue":  # claims success with data that violates its output schema
                now = time.perf_counter()
                timeline.append((self.name, now, now))
                return AgentOutput(agent_name=self.name, status="success", data={"bogus": True},
                                   termination_reason="finished", started_at="2026-01-01T00:00:00Z",
                                   finished_at="2026-01-01T00:00:00Z", duration_ms=0.0)
            return await super().run(state, input_payload)

        async def decide_next_action(self, ctx: AgentContext):
            if ctx.step == 1:
                inputs[self.name] = ctx.input
                start = time.perf_counter()
                await asyncio.sleep(delay)
                timeline.append((self.name, start, time.perf_counter()))
                if behaviour == "permanent":
                    raise RuntimeError("deterministic bug")
                if behaviour == "transient" or (behaviour == "flaky" and runs["n"] == 1):
                    return CallTool(tool_name="always_fails")
            return Finish(reason="stub done")

        async def finalize(self, ctx: AgentContext):
            return STUB_OUTPUTS[self.name]

    Stub.__name__ = f"Stub{base.__name__}"
    return Stub


def build_supervisor(settings, llm=None, behaviours: dict[str, str] | None = None, store=None,
                     delay: float = 0.0, timeline: list | None = None, inputs: dict | None = None) -> SupervisorAgent:
    behaviours = behaviours or {}
    timeline = timeline if timeline is not None else []
    inputs = inputs if inputs is not None else {}
    registry = stub_registry()
    agents = {cls.name: make_stub_class(cls, behaviours.get(cls.name, "ok"), timeline, inputs, delay)(
        registry=registry, settings=settings) for cls in SPECIALIST_CLASSES}
    return SupervisorAgent(agents=agents, llm=llm, store=store if store is not None else InMemoryAgentRunStore(),
                           registry=registry, settings=settings)


RESUME_REQUEST = {"actor": "student", "request_kind": "resume_upload",
                  "resume_text": "Data science student. Skills: Python, SQL, pandas. Project: sales dashboard."}
JD_REQUEST = {"actor": "company", "request_kind": "jd_upload",
              "jd_text": "Data Analyst intern. Required: Python, SQL. Preferred: Power BI."}
