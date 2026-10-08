import uuid

import pytest

from app.core.config import Settings
from app.schemas.agent_state import AgentState


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch):
    """Settings read environment variables even with _env_file=None, so a developer's shell
    (e.g. after `source .env`) must not leak into tests. The .env *file* is unaffected."""
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)


@pytest.fixture
def settings() -> Settings:
    """Fast, deterministic limits; ignores the developer's .env."""
    return Settings(
        _env_file=None,
        environment="test",
        agent_max_retries=2,
        agent_max_steps=5,
        agent_timeout_seconds=2.0,
        tool_timeout_seconds=0.5,
        tool_retry_backoff_seconds=0.0,
    )


@pytest.fixture
def sup_settings() -> Settings:
    """Supervisor tests: specialist tool retries off, so Supervisor-level retries are countable."""
    return Settings(
        _env_file=None,
        environment="test",
        agent_max_retries=0,
        agent_timeout_seconds=2.0,
        tool_timeout_seconds=0.5,
        tool_retry_backoff_seconds=0.0,
        llm_timeout_seconds=0.2,
        llm_max_retries=1,
        supervisor_timeout_seconds=10.0,
    )


@pytest.fixture(scope="session")
def embedder():
    """One fitted interim TF-IDF provider for the whole test session (fitting takes about 1 s)."""
    from app.embeddings.provider import TfidfBaselineProvider

    provider = TfidfBaselineProvider(Settings(_env_file=None).processed_data_dir)
    provider.describe()
    return provider


@pytest.fixture
def state() -> AgentState:
    return AgentState(request_id=f"req-{uuid.uuid4().hex[:8]}", task_type="test")
