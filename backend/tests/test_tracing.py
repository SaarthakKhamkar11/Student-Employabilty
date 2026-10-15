"""Phase 12: agent tracing (spec §39, docs/adr/0020).

1. THE PII test: a resume with a real-looking name, email and phone goes through the real Supervisor -> Profile ->
   Matching -> Learning -> Explainability chain, and the captured log output and every persisted trace record are
   grepped for those literal strings.
2. The trace tree is rebuilt from the persisted rows and matches what really ran (tools, attempts, durations).
3. Retries, failures, skips and LLM calls appear in the trace as they happened.
"""

import asyncio
import io
import json
import logging
import os
import uuid

import pytest

from app.agents import build_agents
from app.agents.profile_agent import section_texts
from app.core.config import Settings
from app.core.logging import JsonFormatter
from app.core.trace_tree import build_trace_tree, get_agent_trace, render_trace
from app.db.profile_repository import InMemoryProfileRepository
from app.db.store import InMemoryAgentRunStore
from app.learning.repository import InMemoryLearningRepository
from app.matching.entities import student_view
from app.tools import learning_tools
from tests.helpers import FakeLLM, build_supervisor
from tests.test_learning_agent import COURSES, FakeYouTube

NAME, EMAIL, PHONE = "Rohan Mehta", "rohan.mehta@example.com", "+91 98765 43210"
RESUME = f"""{NAME}
Email: {EMAIL} | Phone: {PHONE}

Education: B.E. Computer Engineering, 2027
Skills: Python, SQL, Power BI, Machine learning, Excel
Projects: {NAME} built a sales dashboard in Power BI with Python and SQL for a retail client.
Experience: Data Analyst Intern at Example Analytics, 6 months
"""
# the literal strings, plus the forms they could leak in: each name word, the phone's bare digits
FORBIDDEN = [NAME, EMAIL, PHONE, "Rohan", "Mehta", "rohan.mehta", "9876543210", "919876543210"]
ROUTE = {"task_type": "ingest_student_profile", "rationale": "Resume upload: profile, match, learn, explain.",
         "agents": ["profile_agent", "matching_agent", "learning_agent", "explainability_agent"]}


class LinkedProfileRepository(InMemoryProfileRepository):
    """What Postgres does for free: a saved profile becomes matchable (a view and its vectors) right away."""

    def __init__(self, matching_repo, embedder):
        super().__init__()
        self.matching_repo, self.embedder = matching_repo, embedder

    def save_profile(self, profile, embeddings):
        result = super().save_profile(profile, embeddings)
        data = profile.model_dump(mode="json")
        self.matching_repo.add_student(student_view(profile.student_id, data, {"country": "India"}), None)
        version, _ = self.embedder.describe()
        texts = section_texts(data)
        for section, vec in zip(texts, self.embedder.embed(list(texts.values()), "query")):
            if vec is not None:
                self.matching_repo.index.add("student", profile.student_id, section, vec, self.embedder.model_name,
                                             version)
        return result


@pytest.fixture
def learning_cfg(monkeypatch):
    conf = {"ranking": {"top_k": 3, "min_relevance": {"tfidf_baseline": 0.1}},
            "youtube": {"enabled": True, "query": "{skill} tutorial", "max_results": 5, "cache_days": 7,
                        "max_searches_per_request": 10}}
    monkeypatch.setattr(learning_tools, "learning_config", lambda: conf)


def full_chain(embedder, store, llm=None):
    from tests.matching_world import build_world

    repo, cfg, cal = build_world(embedder)
    settings = Settings(_env_file=None, environment="test", tool_retry_backoff_seconds=0.0, agent_timeout_seconds=60.0,
                        tool_timeout_seconds=30.0, supervisor_timeout_seconds=120.0, llm_timeout_seconds=5.0,
                        matching_model="weighted_baseline_v1", youtube_api_key="k")
    agents = build_agents(settings=settings, llm=llm, store=store, profile_repo=LinkedProfileRepository(repo, embedder),
                          job_repo=None, embedder=embedder, fallback_embedder=None, matching_repo=repo,
                          learning_repo=InMemoryLearningRepository(COURSES), youtube_search=FakeYouTube(),
                          matching_overrides={"cfg": cfg, "calibration_loader": lambda *_: cal})
    return agents["supervisor"]


@pytest.fixture
def captured_logs():
    """Every log record of every logger, formatted exactly as production formats them (JSON, after redaction)."""
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JsonFormatter())
    handler.setLevel(logging.DEBUG)
    root, ours = logging.getLogger(), logging.getLogger("employability")
    levels = root.level, ours.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    ours.setLevel(logging.DEBUG)
    try:
        yield buffer
    finally:
        root.removeHandler(handler)
        root.setLevel(levels[0])
        ours.setLevel(levels[1])


# -- 1. THE PII test ----------------------------------------------------------------------------------------------
def test_no_pii_from_the_resume_reaches_any_log_line_or_trace_record(embedder, learning_cfg, captured_logs, capsys):
    store = InMemoryAgentRunStore()
    supervisor = full_chain(embedder, store, llm=FakeLLM(ROUTE))
    resp = asyncio.run(supervisor.handle({"actor": "student", "request_kind": "resume_upload", "resume_text": RESUME}))

    # the chain really ran: all five agents produced results, and the PII was there to be found. (Explainability
    # is "partial": FakeLLM's reply is no valid draft, so the validated structured explanation is used instead.)
    assert resp.status == "partial" and {e["code"] for e in resp.errors} == {"explanation_fallback"}, resp.errors
    assert all(isinstance(resp.results[a].data, dict) for a in ROUTE["agents"])
    assert [resp.results[a].status for a in ROUTE["agents"]] == ["success", "success", "success", "partial"]
    logs = captured_logs.getvalue() + capsys.readouterr().out
    events = [json.loads(line) for line in captured_logs.getvalue().splitlines() if line.startswith("{")]
    assert {e["agent_name"] for e in events if e["event"] == "agent_run"} == {"supervisor", *ROUTE["agents"]}
    flags = next(e for e in events if e["event"] == "pii_flags" and "resume" in e["fields_scanned"])["flags"]
    assert {f["type"] for f in flags} >= {"name", "email", "phone"}
    # not vacuous: the detector only masks the name on the first line, so the copy in the project text flows on
    # into the student's own profile (their response), and only the trace redaction keeps it out of the records
    assert NAME in resp.model_dump_json()

    trace = asyncio.run(get_agent_trace(resp.request_id, store))
    persisted = json.dumps({"runs": store.records, "executions": store.executions, "messages": store.messages},
                           default=str)
    texts = {"log output": logs, "persisted agent_runs + agent_messages": persisted,
             "trace tree": trace.model_dump_json() + render_trace(trace)}
    print(f"\ngrep over {len(logs.splitlines())} log lines, {len(store.executions) + 1} agent_runs rows, "
          f"{len(store.messages)} agent_messages rows:")
    for label, text in texts.items():
        hits = {s: text.lower().count(s.lower()) for s in FORBIDDEN}
        print(f"  {label:40s} " + "  ".join(f"{s!r}: {n}" for s, n in hits.items()))
        assert not any(hits.values()), f"PII leaked into the {label}: {hits}"
    assert "[REDACTED]" in persisted  # the name in the project text was replaced, not dropped silently


# -- 2. the tree is what actually ran ------------------------------------------------------------------------------
def test_the_trace_tree_is_rebuilt_from_the_persisted_rows_and_matches_the_real_run(embedder, learning_cfg):
    store = InMemoryAgentRunStore()
    resp = asyncio.run(full_chain(embedder, store, llm=FakeLLM(ROUTE)).handle(
        {"actor": "student", "request_kind": "resume_upload", "resume_text": RESUME}))
    tree = asyncio.run(get_agent_trace(resp.request_id, store))
    print("\n" + render_trace(tree))
    root = tree.root
    assert root.kind == "supervisor" and root.name == "supervisor" and tree.request_id == resp.request_id
    agents = [c for c in root.children if c.kind == "agent"]
    assert [a.name for a in agents] == ROUTE["agents"]  # in execution order: waves 1, 2, then 3 (in parallel)
    # each agent node's tools are exactly the tool calls its own AgentOutput reports, in order, with the same timings
    results = store.records[resp.request_id]["agent_results"]
    for node in agents:
        reported = results[node.name]["tool_calls"]
        tools = [c for c in node.children if c.kind == "tool"]
        assert [t.name for t in tools] == [c["tool_name"] for c in reported]
        assert node.tool_call_count == len(reported) and node.status == results[node.name]["status"]
        assert node.tool_latency_ms == pytest.approx(sum(c["duration_ms"] for c in reported), abs=1e-3)
        assert all(t.duration_ms >= 0 and t.started_at >= node.started_at for t in tools)
        assert node.finished_at >= max(t.finished_at for t in tools)
    names = {a.name: a for a in agents}
    assert [t.name for t in names["matching_agent"].children][-1] == "cohort_benchmark"
    assert names["matching_agent"].model_version.startswith("weighted_baseline_v1")
    assert names["profile_agent"].model_version.startswith("tfidf_baseline")
    # LLM calls: the routing call under the Supervisor; the explanation calls inside the explainability tool
    routing_calls = [c for c in root.children if c.kind == "llm"]
    assert len(routing_calls) == 1 and routing_calls[0].status == "success"
    generate = [t for t in names["explainability_agent"].children if t.name == "generate_explanation"]
    nested = [c for t in generate for c in t.children if c.kind == "llm"]
    assert nested and len(nested) == len(generate)  # FakeLLM answers with a routing JSON: invalid for a draft
    assert all(c.status == "failed" and c.error == "LLMInvalidOutputError" for c in nested)
    assert tree.counts["llm_calls"] == 1 + len(nested)
    assert "fake-gemini" in (names["explainability_agent"].model_used or "")


def test_retries_failures_and_skips_appear_as_they_happened(sup_settings):
    store = InMemoryAgentRunStore()
    supervisor = build_supervisor(sup_settings, store=store,
                                  behaviours={"profile_agent": "flaky", "learning_agent": "transient"})
    resp = asyncio.run(supervisor.handle({"actor": "student", "request_kind": "resume_upload",
                                          "resume_text": "Skills: Python"}))
    tree = asyncio.run(get_agent_trace(resp.request_id, store))
    print("\n" + render_trace(tree))
    runs = [c for c in tree.root.children if c.kind == "agent"]
    profile = [r for r in runs if r.name == "profile_agent"]
    learning = [r for r in runs if r.name == "learning_agent"]
    # flaky: the first execution's tool failed both its attempts (the Profile Agent retries a tool once), the
    # Supervisor re-ran the agent, and the second execution succeeded
    assert [(r.attempt, r.status) for r in profile] == [(1, "failed"), (2, "success")]
    assert [(t.name, t.attempt, t.status, t.error) for t in profile[0].children] == [
        ("always_fails", 1, "failed", "ToolExecutionError"), ("always_fails", 2, "failed", "ToolExecutionError")]
    assert profile[0].retry_count == 1 and profile[0].error == "tool_failure"
    # transient: both executions failed; the request degrades to partial
    assert [(r.attempt, r.status) for r in learning] == [(1, "failed"), (2, "failed")]
    assert tree.counts["agent_retries"] == 2 and resp.status == "partial"
    assert store.records[resp.request_id]["retry_count"] == 2
    no_llm = asyncio.run(get_agent_trace(resp.request_id, store)).counts["llm_calls"]
    assert no_llm == 0  # the deterministic router made no LLM call, and none is shown


def test_a_skipped_agent_is_recorded_as_a_decision_not_an_execution(sup_settings):
    store = InMemoryAgentRunStore()
    resp = asyncio.run(build_supervisor(sup_settings, store=store, behaviours={"matching_agent": "permanent"})
                       .handle({"actor": "student", "request_kind": "resume_upload", "resume_text": "Skills: SQL"}))
    tree = asyncio.run(get_agent_trace(resp.request_id, store))
    skipped = {c.name: c for c in tree.root.children if c.kind == "skipped"}
    assert set(skipped) == {"learning_agent", "explainability_agent"}
    assert all(s.duration_ms == 0 and "upstream matching_agent" in s.details["reason"] for s in skipped.values())
    assert not [r for r in store.executions.values() if r["agent_name"] in skipped]  # no execution row


def test_an_unpersisted_request_has_no_trace_rather_than_a_made_up_one():
    assert asyncio.run(get_agent_trace(str(uuid.uuid4()), InMemoryAgentRunStore())) is None


def test_llm_quota_errors_are_traced_with_their_http_status(sup_settings):
    from app.llm import LLMError

    store = InMemoryAgentRunStore()
    llm = FakeLLM(LLMError("ClientError 429 RESOURCE_EXHAUSTED: quota exceeded"))
    resp = asyncio.run(build_supervisor(sup_settings, llm=llm, store=store).handle(
        {"actor": "student", "request_kind": "resume_upload", "resume_text": "Skills: SQL"}))
    tree = asyncio.run(get_agent_trace(resp.request_id, store))
    calls = [c for c in tree.root.children if c.kind == "llm"]
    assert len(calls) == 1 + sup_settings.llm_max_retries == len(llm.calls)  # every real call, no more, no less
    assert all(c.status == "failed" and c.error == "LLMError 429" and c.details["http_status"] == 429 for c in calls)
    assert tree.routing_source == "deterministic_fallback"


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="needs TEST_DATABASE_URL (a migrated Postgres)")
def test_the_trace_round_trips_through_postgres(embedder, learning_cfg):
    from app.db.migrate import apply_migrations
    from app.db.store import PostgresAgentRunStore

    dsn = os.environ["TEST_DATABASE_URL"]
    apply_migrations(dsn)
    memory, pg = InMemoryAgentRunStore(), PostgresAgentRunStore(dsn)

    class Both:  # the same request written to both stores, so the two trees can be compared
        async def save(self, *a):
            await memory.save(*a)
            await pg.save(*a)

        async def get(self, rid):
            return await pg.get(rid)

    resp = asyncio.run(full_chain(embedder, Both(), llm=FakeLLM(ROUTE)).handle(
        {"actor": "student", "request_kind": "resume_upload", "resume_text": RESUME}))
    assert resp.persisted
    from_pg = asyncio.run(get_agent_trace(resp.request_id, pg))
    from_memory = asyncio.run(get_agent_trace(resp.request_id, memory))
    assert render_trace(from_pg) == render_trace(from_memory)
    records = asyncio.run(pg.get_trace(resp.request_id))
    assert not any(s.lower() in json.dumps(records.model_dump(), default=str).lower() for s in FORBIDDEN)
    assert build_trace_tree(records).counts == from_memory.counts
