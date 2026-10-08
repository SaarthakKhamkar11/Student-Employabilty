"""The agentic runtime contract: decide, call, validate, retry, fall back, terminate, trace."""

import asyncio
import json
import logging

import pytest
from pydantic import BaseModel

from app.agents.base import AgentConfigurationError, BaseAgent
from app.core.logging import JsonFormatter
from app.schemas.agent_io import CallTool, Finish
from app.schemas.agent_state import INSUFFICIENT_EVIDENCE
from app.tools.registry import ToolRegistry


class DemoIn(BaseModel):
    start: int


class DemoOut(BaseModel):
    total: int


class Num(BaseModel):
    value: int


def make_registry(calls: dict) -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register("add_one", output_schema=Num)
    def add_one(x: int, email: str = ""):
        calls["add_one"] = calls.get("add_one", 0) + 1
        return {"value": x + 1}

    @reg.register("flaky", output_schema=Num)
    def flaky(x: int):
        calls["flaky"] = calls.get("flaky", 0) + 1
        if calls["flaky"] == 1:
            raise ConnectionError("transient")
        return {"value": x}

    @reg.register("always_bad", output_schema=Num)
    def always_bad():
        calls["always_bad"] = calls.get("always_bad", 0) + 1
        return {"not": "a Num"}

    @reg.register("hangs", output_schema=Num)
    async def hangs():
        await asyncio.sleep(10)

    return reg


def make_agent(script, *, allowed=("add_one", "flaky", "always_bad", "hangs"), finalize=None, fallback=None):
    """An agent whose decisions follow a fixed script (Phase 3+ replaces this with real policies)."""

    class ScriptedAgent(BaseAgent):
        name = "scripted_agent"
        role_description = "test agent"
        allowed_tools = list(allowed)
        input_schema = DemoIn
        output_schema = DemoOut

        async def decide_next_action(self, ctx):
            step = script[min(ctx.step, len(script)) - 1]
            return step(ctx) if callable(step) else step

        async def finalize(self, ctx):
            if finalize:
                return finalize(ctx)
            return DemoOut(total=ctx.observations[-1].output.value if ctx.observations else ctx.input.start)

    if fallback:
        ScriptedAgent.fallback = fallback
    return ScriptedAgent


def run(agent, state, payload=None):
    return asyncio.run(agent.run(state, payload if payload is not None else {"start": 1}))


def test_success_path_validates_records_and_terminates(settings, state):
    calls = {}
    agent = make_agent([CallTool(tool_name="add_one", arguments={"x": 1}), Finish(reason="done")])(
        make_registry(calls), settings)
    out = run(agent, state)
    assert out.status == "success" and out.termination_reason == "finished"
    assert out.data == {"total": 2}
    assert [(c.tool_name, c.attempts, c.success) for c in out.tool_calls] == [("add_one", 1, True)]
    assert state.agents_called == ["scripted_agent"]
    assert state.agent_results["scripted_agent"]["status"] == "success"
    assert state.errors == []


def test_transient_tool_failure_is_retried_then_succeeds(settings, state):
    calls = {}
    agent = make_agent([CallTool(tool_name="flaky", arguments={"x": 7}), Finish(reason="done")])(
        make_registry(calls), settings)
    out = run(agent, state)
    assert out.status == "success" and out.data == {"total": 7}
    assert out.tool_calls[0].attempts == 2 and calls["flaky"] == 2


def test_invalid_tool_output_exhausts_retries_then_falls_back_without_bad_data(settings, state):
    calls = {}
    agent = make_agent([CallTool(tool_name="always_bad"), Finish(reason="done")])(make_registry(calls), settings)
    out = run(agent, state)
    assert out.status == "failed" and out.termination_reason == "tool_failure" and out.fallback_used
    assert out.data == INSUFFICIENT_EVIDENCE
    assert calls["always_bad"] == 1 + settings.agent_max_retries
    assert out.tool_calls[0].error_type == "ToolValidationError"
    assert state.errors and state.errors[0]["code"] == "tool_failure"


def test_max_retries_is_configurable(settings, state):
    calls = {}
    settings.agent_max_retries = 0
    agent = make_agent([CallTool(tool_name="always_bad")])(make_registry(calls), settings)
    run(agent, state)
    assert calls["always_bad"] == 1


def test_fallback_override_can_return_partial_data(settings, state):
    async def keep_what_we_have(self, ctx, error):
        return "partial", {"total": ctx.observations[-1].output.value}

    agent = make_agent(
        [CallTool(tool_name="add_one", arguments={"x": 1}), CallTool(tool_name="always_bad")],
        fallback=keep_what_we_have,
    )(make_registry({}), settings)
    out = run(agent, state)
    assert out.status == "partial" and out.data == {"total": 2} and out.errors


def test_tool_outside_allowed_tools_is_refused(settings, state):
    calls = {}
    agent = make_agent([CallTool(tool_name="add_one", arguments={"x": 1})], allowed=())(make_registry(calls), settings)
    out = run(agent, state)
    assert out.status == "failed" and out.termination_reason == "invalid_decision"
    assert "add_one" not in calls


def test_unregistered_allowed_tool_fails_at_construction(settings):
    with pytest.raises(AgentConfigurationError):
        make_agent([], allowed=("ghost_tool",))(make_registry({}), settings)


def test_malformed_decision_terminates_cleanly(settings, state):
    agent = make_agent([{"action": "call_tool"}])(make_registry({}), settings)  # missing tool_name
    out = run(agent, state)
    assert out.status == "failed" and out.termination_reason == "invalid_decision"


def test_decision_exception_never_escapes_run(settings, state):
    def explode(ctx):
        raise RuntimeError("policy crashed")

    out = run(make_agent([explode])(make_registry({}), settings), state)
    assert out.status == "failed" and out.termination_reason == "invalid_decision"


def test_output_failing_output_schema_is_rejected(settings, state):
    agent = make_agent([Finish(reason="done")], finalize=lambda ctx: {"total": "lots"})(make_registry({}), settings)
    out = run(agent, state)
    assert out.status == "failed" and out.termination_reason == "invalid_output"


def test_invalid_input_payload_is_rejected(settings, state):
    out = run(make_agent([Finish(reason="done")])(make_registry({}), settings), state, {"wrong": 1})
    assert out.status == "failed" and out.termination_reason == "invalid_input"


def test_hanging_tool_times_out_and_falls_back(settings, state):
    settings.agent_max_retries = 0
    out = run(make_agent([CallTool(tool_name="hangs")])(make_registry({}), settings), state)
    assert out.status == "failed" and out.termination_reason == "tool_failure"
    assert out.tool_calls[0].error_type == "ToolExecutionError"


def test_hanging_decision_hits_agent_timeout(settings, state):
    async def stall(ctx):
        await asyncio.sleep(10)

    class Stalling(make_agent([])):
        async def decide_next_action(self, ctx):
            return await stall(ctx)

    settings.agent_timeout_seconds = 0.2
    out = run(Stalling(make_registry({}), settings), state)
    assert out.status == "failed" and out.termination_reason == "timeout"


def test_loop_without_finish_stops_at_max_steps(settings, state):
    calls = {}
    agent = make_agent([CallTool(tool_name="add_one", arguments={"x": 0})])(make_registry(calls), settings)
    out = run(agent, state)
    assert out.status == "failed" and out.termination_reason == "max_steps"
    assert calls["add_one"] == settings.agent_max_steps


def test_logs_are_tagged_with_request_id_and_contain_no_pii(settings, state, caplog):
    caplog.set_level(logging.INFO, logger="employability")
    script = [CallTool(tool_name="add_one", arguments={"x": 1, "email": "jane.doe@example.com"}), Finish(reason="ok")]
    run(make_agent(script)(make_registry({}), settings), state)
    lines = [JsonFormatter().format(r) for r in caplog.records]
    assert any('"event": "agent_run"' in line for line in lines)
    assert all(json.loads(line)["request_id"] == state.request_id for line in lines)
    assert not any("jane.doe@example.com" in line for line in lines)
    agent_run = next(json.loads(line) for line in lines if '"event": "agent_run"' in line)
    for key in ("agent_name", "task", "start_time", "end_time", "duration_ms", "tool_calls", "status", "retry_count"):
        assert key in agent_run
