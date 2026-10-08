"""The Gemini client wrapper, exercised with the real SDK types but no network."""

import asyncio
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.llm import LLMError, LLMInvalidOutputError, LLMTimeoutError, build_llm_client, parse_structured
from app.llm.client import GeminiClient
from app.schemas.agent_io import LLMRoutingDecision

FAKE_KEY = "fake-gemini-key-0123456789"
VALID = '{"task_type": "ingest_job", "agents": ["job_intelligence_agent"], "rationale": "JD upload"}'


def test_parse_structured_validates_locally():
    assert parse_structured(VALID, LLMRoutingDecision).agents == ["job_intelligence_agent"]
    for bad in ("", "not json", '{"task_type": "ingest_job"}'):
        with pytest.raises(LLMInvalidOutputError):
            parse_structured(bad, LLMRoutingDecision)


def test_client_is_built_only_when_key_and_model_are_set():
    assert build_llm_client(Settings(_env_file=None)) is None
    assert build_llm_client(Settings(_env_file=None, gemini_api_key=FAKE_KEY)) is None
    client = build_llm_client(Settings(_env_file=None, gemini_api_key=FAKE_KEY, gemini_model="some-model"))
    assert isinstance(client, GeminiClient) and client.model_name == "some-model"


def _client(monkeypatch, fake_generate):
    client = GeminiClient(FAKE_KEY, "some-model")
    monkeypatch.setattr(client._client.aio.models, "generate_content", fake_generate)
    return client


def call(client, timeout_s=1.0):
    return asyncio.run(client.generate_structured(system="sys", prompt="p", schema=LLMRoutingDecision,
                                                  timeout_s=timeout_s))


def test_request_is_json_constrained_deterministic_and_not_retried_by_the_sdk(monkeypatch):
    seen = {}

    async def fake_generate(*, model, contents, config):
        seen.update(model=model, config=config)
        return SimpleNamespace(text=VALID)

    assert call(_client(monkeypatch, fake_generate), timeout_s=10).task_type == "ingest_job"
    config = seen["config"]
    assert seen["model"] == "some-model" and config.temperature == 0.0
    assert config.response_mime_type == "application/json" and config.response_json_schema
    assert config.http_options.timeout == 10_000 and config.http_options.retry_options.attempts == 1


def test_provider_errors_are_wrapped_and_never_echo_the_key(monkeypatch):
    async def fake_generate(**kwargs):
        raise RuntimeError(f"401 invalid key {FAKE_KEY}")

    with pytest.raises(LLMError) as info:
        call(_client(monkeypatch, fake_generate))
    assert FAKE_KEY not in str(info.value)


def test_slow_provider_raises_timeout(monkeypatch):
    async def fake_generate(**kwargs):
        await asyncio.sleep(5)

    with pytest.raises(LLMTimeoutError):
        call(_client(monkeypatch, fake_generate), timeout_s=0.05)
