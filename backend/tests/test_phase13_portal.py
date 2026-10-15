"""Phase 13: the dashboard rules that live in the backend.

- Gemini 429: exponential backoff on the same model, at most 3 calls, then the model's cooldown and the next model.
- An uploaded JD asked for again by id reuses its stored posting (so a company's confirmed corrections survive).
- The Ask-AI message on a quota failure is the fixed, visible one.
"""

import asyncio

from pydantic import BaseModel

from app.llm import FallbackLLMClient, LLMQuotaError
from app.portal.service import QUOTA_MESSAGE
from tests.helpers import FakeLLM


class Answer(BaseModel):
    ok: bool


class Model(FakeLLM):
    def __init__(self, name, *responses):
        super().__init__(*responses)
        self.model_name = name


def quota(model):
    return LLMQuotaError(f"ClientError 429 RESOURCE_EXHAUSTED: {model}", retry_after_s=60.0, model=model)


def _client(*models, attempts=3):
    sleeps: list[float] = []

    async def sleep(s):
        sleeps.append(s)

    return FallbackLLMClient(list(models), quota_attempts=attempts, backoff_s=1.0, sleep=sleep), sleeps


def _ask(client):
    return asyncio.run(client.generate_structured(system="s", prompt="p", schema=Answer, timeout_s=5))


def test_a_429_is_retried_on_the_same_model_with_exponential_backoff():
    primary = Model("primary", quota("primary"), quota("primary"), {"ok": True})
    client, sleeps = _client(primary)
    assert _ask(client).ok is True
    assert len(primary.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_after_three_429s_the_model_cools_down_and_the_next_model_answers():
    primary = Model("primary", quota("primary"))
    backup = Model("backup", {"ok": True})
    client, sleeps = _client(primary, backup)
    assert _ask(client).ok is True
    assert len(primary.calls) == 3  # max 3 attempts, never more
    assert sleeps == [1.0, 2.0]
    assert [s["available"] for s in client.status()] == [False, True]


def test_the_default_keeps_the_pre_phase_13_behaviour_of_one_call_per_model():
    primary = Model("primary", quota("primary"))
    backup = Model("backup", {"ok": True})
    client = FallbackLLMClient([primary, backup])
    assert _ask(client).ok is True
    assert len(primary.calls) == 1


def test_the_quota_message_is_the_fixed_visible_one():
    assert QUOTA_MESSAGE == "AI summary unavailable (quota exceeded). Showing stored explanation."


def test_an_uploaded_job_asked_for_by_id_reuses_its_stored_posting(settings, embedder):
    from app.agents import build_agents
    from app.db.job_repository import InMemoryJobRepository
    from app.db.profile_repository import InMemoryProfileRepository
    from app.schemas.agent_state import AgentState
    from tests.test_job_agent import JD

    repo = InMemoryJobRepository()
    agents = build_agents(settings=settings, profile_repo=InMemoryProfileRepository(), job_repo=repo, embedder=embedder)
    agent = agents["job_intelligence_agent"]
    first = asyncio.run(agent.run(AgentState(request_id="r1", task_type="ingest_job"), {"jd_text": JD}))
    job_id = first.data["posting"]["job_id"]
    # the company confirms a correction (Phase 13): the stored posting now differs from what the text would give
    repo.postings[job_id] = dict(repo.postings[job_id], job_title="Junior Data Analyst (confirmed)")
    again = asyncio.run(agent.run(AgentState(request_id="r2", task_type="explain_match"), {"job_id": job_id}))
    assert again.status == "success"
    assert [c.tool_name for c in again.tool_calls] == ["load_job_record"]  # nothing rebuilt, nothing re-saved
    assert again.data["posting"]["job_title"] == "Junior Data Analyst (confirmed)"
    assert again.data["saved"] is False
