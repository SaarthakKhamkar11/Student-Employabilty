"""Gemini free-tier quota handling (Phase 12 follow-up): a per-model fallback chain with a quota circuit breaker,
and no routing call when the UI already names the task."""

import asyncio

import pytest
from pydantic import BaseModel

from app.core.tracing import TraceRecorder, trace_var
from app.db.store import InMemoryAgentRunStore
from app.llm import FallbackLLMClient, LLMQuotaError, capture_served_model
from tests.helpers import FakeLLM, build_supervisor

ROUTE = {"task_type": "ingest_student_profile", "rationale": "Resume upload.",
         "agents": ["profile_agent", "matching_agent", "learning_agent", "explainability_agent"]}


class Answer(BaseModel):
    task_type: str
    agents: list[str]
    rationale: str


class Model(FakeLLM):
    def __init__(self, name, *responses):
        super().__init__(*responses)
        self.model_name = name


def quota(model, delay=120.0):
    return LLMQuotaError(f"ClientError 429 RESOURCE_EXHAUSTED: {model} quota exhausted", retry_after_s=delay, model=model)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def ask(llm):
    async def go():
        with capture_served_model() as info:
            result = await llm.generate_structured(system="s", prompt="p", schema=Answer, timeout_s=1)
        return result, info
    return asyncio.run(go())


def test_a_429_moves_to_the_next_model_and_the_answer_says_which_model_served_it():
    primary, lite = Model("gemini-3.8-flash", quota("gemini-3.8-flash")), Model("gemini-3.5-flash-lite", ROUTE)
    chain = FallbackLLMClient([primary, lite], clock=Clock())
    result, info = ask(chain)
    assert result.task_type == "ingest_student_profile" and info["model"] == "gemini-3.5-flash-lite"
    assert len(primary.calls) == 1 and len(lite.calls) == 1


def test_an_exhausted_model_is_not_called_again_until_its_reset():
    clock = Clock()
    primary, lite = Model("gemini-3.8-flash", quota("gemini-3.8-flash", 120), ROUTE), Model("lite", ROUTE)
    chain = FallbackLLMClient([primary, lite], clock=clock)
    ask(chain)
    ask(chain)
    assert len(primary.calls) == 1 and len(lite.calls) == 2  # the primary was skipped, not re-asked
    assert [s["available"] for s in chain.status()] == [False, True]
    clock.now += 121  # the provider's own reset delay has passed
    _, info = ask(chain)
    assert info["model"] == "gemini-3.8-flash" and len(primary.calls) == 2


def test_when_every_model_is_exhausted_the_call_fails_at_once_without_a_network_call():
    clock = Clock()
    a, b = Model("a", quota("a", 300)), Model("b", quota("b", 60))
    chain = FallbackLLMClient([a, b], clock=clock)
    with pytest.raises(LLMQuotaError):
        ask(chain)
    with pytest.raises(LLMQuotaError, match="first resets in 60s"):
        ask(chain)
    assert len(a.calls) == len(b.calls) == 1  # the second call cost nothing


def test_without_a_reset_delay_the_configured_cooldown_applies():
    clock = Clock()
    a = Model("a", LLMQuotaError("429", retry_after_s=None), ROUTE)
    chain = FallbackLLMClient([a], cooldown_s=3600, clock=clock)
    with pytest.raises(LLMQuotaError):
        ask(chain)
    assert chain.status()[0]["cooldown_remaining_s"] == 3600


def test_each_real_provider_call_is_its_own_trace_span_under_its_own_model():
    trace = TraceRecorder("req-q")
    token = trace_var.set(trace)
    try:
        ask(FallbackLLMClient([Model("gemini-3.8-flash", quota("gemini-3.8-flash")), Model("lite", ROUTE)],
                              clock=Clock()))
    finally:
        trace_var.reset(token)
    assert [(s.kind, s.name, s.status) for s in trace.spans] == [("llm", "gemini-3.8-flash", "failed"),
                                                                  ("llm", "lite", "success")]


def test_the_router_reports_the_model_that_answered_and_does_not_retry_a_quota_error(sup_settings):
    chain = FallbackLLMClient([Model("gemini-3.8-flash", quota("gemini-3.8-flash")), Model("lite", ROUTE)],
                              clock=Clock())
    resp = asyncio.run(build_supervisor(sup_settings, llm=chain).handle(
        {"actor": "student", "request_kind": "resume_upload", "resume_text": "Skills: SQL"}))
    assert resp.routing.source == "llm" and resp.routing.model_used == "lite"
    exhausted = FallbackLLMClient([Model("a", quota("a"))], clock=Clock())
    resp = asyncio.run(build_supervisor(sup_settings, llm=exhausted).handle(
        {"actor": "student", "request_kind": "resume_upload", "resume_text": "Skills: SQL"}))
    assert resp.routing.fallback_reason == "llm_quota_exhausted" and resp.routing.llm_attempts == 1


def test_an_explicit_task_type_spends_no_llm_call_and_records_no_error(sup_settings):
    llm = FakeLLM(ROUTE)
    store = InMemoryAgentRunStore()
    resp = asyncio.run(build_supervisor(sup_settings, llm=llm, store=store).handle(
        {"actor": "student", "request_kind": "question", "request_text": "Which jobs suit me?",
         "task_type": "recommend_jobs_for_student", "student_id": "S00001"}))
    assert llm.calls == [] and resp.routing.fallback_reason == "llm_not_needed: explicit task_type"
    assert not [e for e in resp.errors if e["code"] == "llm_routing_fallback"]
    # opting back in routes it by the LLM
    llm2 = FakeLLM({"task_type": "recommend_jobs_for_student", "agents": ["profile_agent", "matching_agent"],
                    "rationale": "jobs"})
    settings = sup_settings.model_copy(update={"llm_route_explicit_tasks": True})
    asyncio.run(build_supervisor(settings, llm=llm2).handle(
        {"actor": "student", "request_kind": "question", "request_text": "Which jobs suit me?",
         "task_type": "recommend_jobs_for_student", "student_id": "S00001"}))
    assert len(llm2.calls) == 1


def test_gemini_429_details_are_read_before_the_message_is_redacted(monkeypatch):
    from app.llm.client import GeminiClient

    class Boom(Exception):
        code, status = 429, "RESOURCE_EXHAUSTED"

    raw = ("429 RESOURCE_EXHAUSTED. {'error': {'details': [{'violations': [{'quotaId': "
           "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '20'}]}, {'retryDelay': '61890s'}]}}"
           + " padding" * 50)

    class Models:
        async def generate_content(self, **_):
            raise Boom(raw)

    client = GeminiClient.__new__(GeminiClient)
    client.model_name, client._api_key = "gemini-3.8-flash", "k"
    client._client = type("C", (), {"aio": type("A", (), {"models": Models()})()})()
    with pytest.raises(LLMQuotaError) as err:
        asyncio.run(client.generate_structured(system="s", prompt="p", schema=Answer, timeout_s=1))
    assert err.value.retry_after_s == 61890 and err.value.model == "gemini-3.8-flash"
    assert "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in str(err.value)
