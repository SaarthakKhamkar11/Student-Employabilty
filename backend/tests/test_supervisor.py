"""Supervisor Agent (Phase 3): LLM routing, deterministic fallback, ordering, failure policy, state, persistence."""

import asyncio
import json
import logging
import uuid

from app.agents import build_agents
from app.agents.routing import FALLBACK_ROUTES
from app.core.logging import JsonFormatter
from app.db.store import InMemoryAgentRunStore
from app.llm import LLMError
from app.schemas.agent_io import LLMRoutingDecision, SupervisorResponse
from app.schemas.agent_state import INSUFFICIENT_EVIDENCE, NOT_YET_EVALUATED
from tests.helpers import JD_REQUEST, RESUME_REQUEST, FakeLLM, Sleep, build_supervisor

RESUME_ROUTE = {"task_type": "ingest_student_profile", "rationale": "Resume upload: profile, match, learn, explain.",
                # deliberately out of order: the Supervisor, not the LLM, orders by dependencies
                "agents": ["explainability_agent", "learning_agent", "matching_agent", "profile_agent"]}
JD_ROUTE = {"task_type": "ingest_job", "rationale": "JD upload: job intelligence, match, explain.",
            "agents": ["job_intelligence_agent", "matching_agent", "explainability_agent"]}


def handle(supervisor, request):
    return asyncio.run(supervisor.handle(request))


# -- LLM routing (success case) --------------------------------------------------------------
def test_llm_routes_resume_upload_and_skips_job_intelligence(sup_settings):
    store = InMemoryAgentRunStore()
    llm = FakeLLM(RESUME_ROUTE)
    resp = handle(build_supervisor(sup_settings, llm=llm, store=store), RESUME_REQUEST)

    assert resp.routing.source == "llm" and resp.routing.model_used == "fake-gemini"
    assert resp.routing.agents == ["profile_agent", "matching_agent", "learning_agent", "explainability_agent"]
    assert "job_intelligence_agent" not in resp.results
    assert resp.status == "success" and all(o.status == "success" for o in resp.results.values())
    record = store.records[resp.request_id]
    assert record["agents_called"] == ["supervisor", "profile_agent", "matching_agent",
                                       "learning_agent", "explainability_agent"]
    assert len(llm.calls) == 1


def test_llm_routes_jd_upload_and_skips_profile_and_learning(sup_settings):
    resp = handle(build_supervisor(sup_settings, llm=FakeLLM(JD_ROUTE)), JD_REQUEST)
    assert resp.routing.source == "llm"
    assert resp.routing.agents == ["job_intelligence_agent", "matching_agent", "explainability_agent"]
    assert resp.status == "success"


# -- LLM failure → deterministic fallback --------------------------------------------------
def test_llm_exception_falls_back_to_deterministic_route(sup_settings):
    llm = FakeLLM(LLMError("Gemini unavailable: 503"))
    resp = handle(build_supervisor(sup_settings, llm=llm), RESUME_REQUEST)

    SupervisorResponse.model_validate(resp.model_dump())  # a complete, valid response
    assert resp.routing.source == "deterministic_fallback" and resp.routing.fallback_reason == "llm_error"
    assert resp.routing.agents == list(FALLBACK_ROUTES["ingest_student_profile"])
    assert resp.routing.llm_attempts == 2 and len(llm.calls) == 2  # one retry, no more
    assert resp.status == "success" and resp.persisted
    assert any(e["code"] == "llm_routing_fallback" for e in resp.errors)


def test_unexpected_client_crash_also_falls_back(sup_settings):
    resp = handle(build_supervisor(sup_settings, llm=FakeLLM(RuntimeError("socket closed"))), JD_REQUEST)
    assert resp.routing.source == "deterministic_fallback" and resp.routing.fallback_reason == "llm_error"
    assert resp.routing.agents == list(FALLBACK_ROUTES["ingest_job"])


def test_llm_timeout_is_retried_once_then_falls_back(sup_settings):
    llm = FakeLLM(Sleep(5))
    resp = handle(build_supervisor(sup_settings, llm=llm), RESUME_REQUEST)
    assert resp.routing.fallback_reason == "llm_timeout" and len(llm.calls) == 2
    assert resp.routing.duration_ms < 2000


def test_llm_output_failing_schema_falls_back(sup_settings):
    llm = FakeLLM('{"task_type": "delete_everything", "agents": ["all"], "rationale": "x"}')
    resp = handle(build_supervisor(sup_settings, llm=llm), RESUME_REQUEST)
    assert resp.routing.fallback_reason == "llm_invalid_output"


def test_llm_decision_violating_routing_rules_falls_back(sup_settings):
    bad = {"task_type": "ingest_student_profile", "agents": ["matching_agent"], "rationale": "skip parsing"}
    resp = handle(build_supervisor(sup_settings, llm=FakeLLM(bad)), RESUME_REQUEST)
    assert resp.routing.fallback_reason == "llm_decision_rejected: a resume upload must go through profile_agent"


def test_without_llm_configured_routing_is_deterministic(sup_settings):
    resp = handle(build_supervisor(sup_settings, llm=None), RESUME_REQUEST)
    assert resp.routing.source == "deterministic_fallback" and resp.routing.fallback_reason == "llm_not_configured"
    assert resp.routing.llm_attempts == 0
    assert not any(e["code"] == "llm_routing_fallback" for e in resp.errors)  # configuration, not a failure


def test_llm_routing_schema_cannot_carry_numbers():
    """Rule 5 by construction: the LLM's routing output has no numeric fields."""
    props = LLMRoutingDecision.model_json_schema()["properties"]
    assert set(props) == {"task_type", "agents", "rationale"}
    assert "number" not in json.dumps(props) and "integer" not in json.dumps(props)


# -- execution order -------------------------------------------------------------------------
def test_waves_follow_dependencies_and_independent_agents_run_concurrently(sup_settings):
    timeline: list = []
    route = {"task_type": "skill_gap_analysis", "rationale": "gap analysis",
             "agents": ["profile_agent", "job_intelligence_agent", "matching_agent", "learning_agent",
                        "explainability_agent"]}
    request = {"actor": "student", "request_kind": "question", "request_text": "What am I missing for this job?",
               "student_id": "S00001", "job_id": "J0000001"}
    resp = handle(build_supervisor(sup_settings, llm=FakeLLM(route), delay=0.1, timeline=timeline), request)
    assert resp.status == "success"
    t = {name: (start, end) for name, start, end in timeline}
    # wave 1 overlaps (concurrent), wave 2 starts after both, wave 3 overlaps after wave 2
    assert t["job_intelligence_agent"][0] < t["profile_agent"][1]
    assert t["matching_agent"][0] >= max(t["profile_agent"][1], t["job_intelligence_agent"][1])
    assert min(t["learning_agent"][0], t["explainability_agent"][0]) >= t["matching_agent"][1]
    assert t["explainability_agent"][0] < t["learning_agent"][1]


def test_agents_exchange_structured_inputs_built_from_upstream_outputs(sup_settings):
    inputs: dict = {}
    handle(build_supervisor(sup_settings, llm=FakeLLM(RESUME_ROUTE), inputs=inputs), RESUME_REQUEST)
    assert inputs["matching_agent"].student_id == "S00001"  # taken from profile_agent's structured output
    assert inputs["matching_agent"].mode == "jobs_for_student"
    assert inputs["learning_agent"].skill_gaps == ["power bi"]  # taken from matching_agent's skill gap
    assert inputs["explainability_agent"].match.job_id == "J0000001"


# -- failure policy ----------------------------------------------------------------------------
def test_optional_agent_failure_degrades_to_partial_and_keeps_matches(sup_settings):
    resp = handle(build_supervisor(sup_settings, llm=FakeLLM(RESUME_ROUTE),
                                   behaviours={"learning_agent": "transient"}), RESUME_REQUEST)
    assert resp.status == "partial"
    assert resp.results["matching_agent"].status == "success" and resp.results["matching_agent"].data["matches"]
    learning = resp.results["learning_agent"]
    assert learning.status == "failed" and learning.attempts == 2 and learning.data == INSUFFICIENT_EVIDENCE


def test_core_agent_failure_fails_request_and_skips_dependants(sup_settings):
    resp = handle(build_supervisor(sup_settings, llm=FakeLLM(RESUME_ROUTE),
                                   behaviours={"matching_agent": "transient"}), RESUME_REQUEST)
    assert resp.status == "failed"
    assert resp.results["matching_agent"].attempts == 2
    for dependant in ("learning_agent", "explainability_agent"):
        outcome = resp.results[dependant]
        assert outcome.status == "skipped" and outcome.data == INSUFFICIENT_EVIDENCE
        assert outcome.skip_reason == "upstream matching_agent failed"


def test_transient_failure_recovers_on_the_single_retry(sup_settings):
    resp = handle(build_supervisor(sup_settings, llm=FakeLLM(RESUME_ROUTE),
                                   behaviours={"profile_agent": "flaky"}), RESUME_REQUEST)
    assert resp.status == "success" and resp.results["profile_agent"].attempts == 2


def test_deterministic_failure_is_not_retried(sup_settings):
    resp = handle(build_supervisor(sup_settings, llm=FakeLLM(RESUME_ROUTE),
                                   behaviours={"profile_agent": "permanent"}), RESUME_REQUEST)
    profile = resp.results["profile_agent"]
    assert profile.attempts == 1 and profile.termination_reason == "invalid_decision"
    assert resp.status == "failed"


def test_agent_output_violating_its_schema_is_rejected(sup_settings):
    resp = handle(build_supervisor(sup_settings, llm=FakeLLM(RESUME_ROUTE),
                                   behaviours={"matching_agent": "rogue"}), RESUME_REQUEST)
    assert resp.results["matching_agent"].status == "failed"
    assert resp.results["matching_agent"].termination_reason == "invalid_output"
    assert any(e["code"] == "invalid_agent_output" for e in resp.errors)
    assert resp.results["learning_agent"].status == "skipped"


# -- honesty with the real Phase 2 shells -----------------------------------------------------
def _leaves(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _leaves(v)
    elif isinstance(value, list):
        for v in value:
            yield from _leaves(v)
    else:
        yield value


def test_real_shells_yield_sentinels_never_invented_results(sup_settings):
    store = InMemoryAgentRunStore()
    supervisor = build_agents(settings=sup_settings, store=store)["supervisor"]
    resp = handle(supervisor, RESUME_REQUEST)
    assert resp.status == "failed"
    assert resp.results["profile_agent"].termination_reason == "not_implemented"
    assert resp.results["matching_agent"].skip_reason == "upstream profile_agent failed"
    for outcome in resp.results.values():
        assert outcome.data in (NOT_YET_EVALUATED, INSUFFICIENT_EVIDENCE)
    assert set(resp.model_versions.values()) == {NOT_YET_EVALUATED}


def test_unroutable_question_runs_no_agent_and_says_so(sup_settings):
    resp = handle(build_supervisor(sup_settings), {"actor": "student", "request_kind": "question",
                                                   "request_text": "hello there"})
    assert resp.task_type == "unknown" and resp.results == {} and resp.status == "failed"
    assert "Insufficient evidence" in resp.summary


# -- state and persistence ----------------------------------------------------------------------
def test_final_state_is_persisted_with_lifecycle_fields(sup_settings):
    store = InMemoryAgentRunStore()
    resp = handle(build_supervisor(sup_settings, llm=FakeLLM(RESUME_ROUTE), store=store), RESUME_REQUEST)
    record = store.records[resp.request_id]
    uuid.UUID(record["request_id"])
    assert record["task_type"] == "ingest_student_profile" and record["status"] == "success"
    assert record["routing_source"] == "llm" and record["model_used"] == "fake-gemini"
    assert set(record["agent_results"]) == {"supervisor", "profile_agent", "matching_agent",
                                            "learning_agent", "explainability_agent"}
    assert set(record["model_versions"].values()) == {NOT_YET_EVALUATED}


def test_database_failure_does_not_break_the_request(sup_settings):
    class BrokenStore:
        async def save(self, record):
            raise ConnectionRefusedError("db down")

        async def get(self, request_id):
            return None

    resp = handle(build_supervisor(sup_settings, llm=FakeLLM(RESUME_ROUTE), store=BrokenStore()), RESUME_REQUEST)
    assert resp.status == "success" and resp.persisted is False
    assert any(e["code"] == "persistence_failed" for e in resp.errors)


def test_routing_and_agent_runs_are_logged_without_pii(sup_settings, caplog):
    caplog.set_level(logging.INFO, logger="employability")
    request = dict(RESUME_REQUEST, resume_text="Contact: asha.k@college.edu, +91 98765 43210. Skills: Python.")
    resp = handle(build_supervisor(sup_settings, llm=FakeLLM(RESUME_ROUTE)), request)
    lines = [JsonFormatter().format(r) for r in caplog.records]
    routing = next(json.loads(line) for line in lines if '"event": "routing_decision"' in line)
    for key in ("request_id", "agent_name", "task", "duration_ms", "model_used", "retry_count", "error", "source"):
        assert key in routing
    assert routing["request_id"] == resp.request_id
    assert sum('"event": "agent_run"' in line for line in lines) == 5  # supervisor + 4 specialists
    assert not any("asha.k@college.edu" in line or "98765" in line for line in lines)
