"""Agent roster: six agents with roles and contracts; unimplemented specialists are honest shells."""

import asyncio
import re
import socket
from pathlib import Path

import pytest

from app.agents import AGENT_CLASSES, SPECIALIST_CLASSES, build_agents
from app.schemas.agent_state import NOT_YET_EVALUATED, AgentState
from app.tools.registry import ToolRegistry

SPEC_AGENTS = {"supervisor", "profile_agent", "job_intelligence_agent", "matching_agent",
               "learning_agent", "explainability_agent"}
SHELLS = sorted(c.name for c in SPECIALIST_CLASSES if not c.implemented)
VALID_INPUT = {
    "job_intelligence_agent": {"job_id": "J0000001"},
    "matching_agent": {"mode": "jobs_for_student", "student_id": "S00001"},
    "learning_agent": {"student_id": "S00001", "job_id": "J0000001", "skill_gaps": ["SQL"]},
    "explainability_agent": {"match": {"student_id": "S00001", "job_id": "J0000001", "rank": 1,
                                       "model_versions": {}}},
}


@pytest.fixture
def agents(settings):
    return build_agents(ToolRegistry(), settings)


def test_exactly_the_six_spec_agents_exist(agents):
    assert set(agents) == SPEC_AGENTS


@pytest.mark.parametrize("cls", AGENT_CLASSES, ids=lambda c: c.name)
def test_each_agent_declares_role_contracts_and_planned_tools(cls):
    assert len(cls.role_description) > 200
    assert "untrusted data" in cls.role_description and "Never invent" in cls.role_description
    assert cls.input_schema is not None and cls.output_schema is not None
    assert cls.planned_tools and cls.implemented_in_phase


@pytest.mark.parametrize("name", SHELLS)
def test_shell_terminates_honestly_without_network(agents, state, name, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("specialist shells must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    out = asyncio.run(agents[name].run(state, VALID_INPUT[name]))
    assert out.status == "failed" and out.termination_reason == "not_implemented"
    assert out.data == NOT_YET_EVALUATED
    assert f"Phase {agents[name].implemented_in_phase}" in out.errors[0].message
    assert state.agents_called == [name] and state.agent_results[name]["status"] == "failed"


def test_no_model_is_hard_coded_as_the_winner():
    """Embedding and matching models are chosen from evidence in Phases 6 and 8 (spec rules 8, 9)."""
    app_dir = Path(__file__).resolve().parents[1] / "app"
    pattern = re.compile(r"sbert|xgboost|mpnet|minilm|e5-|bge-|lightgbm|random.?forest", re.I)
    hits = [f"{p.name}: {m.group(0)}" for p in app_dir.rglob("*.py") for m in pattern.finditer(p.read_text())]
    assert hits == []


def test_implemented_agents_and_delegation_rights():
    implemented = {cls.name for cls in AGENT_CLASSES if cls.implemented}
    assert implemented == {"supervisor", "profile_agent", "job_intelligence_agent", "matching_agent"}  # Phases 3-5, 7
    supervisor = next(cls for cls in AGENT_CLASSES if cls.name == "supervisor")
    assert set(supervisor.allowed_delegates) == SPEC_AGENTS - {"supervisor"}
    assert all(not cls.allowed_delegates for cls in SPECIALIST_CLASSES)
