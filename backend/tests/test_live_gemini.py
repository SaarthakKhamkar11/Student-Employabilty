"""Opt-in live check against the real Gemini API (costs a request; needs a key).

Run: RUN_LIVE_LLM_TESTS=1 .venv/bin/python -m pytest backend/tests/test_live_gemini.py
with GEMINI_API_KEY and GEMINI_MODEL set in .env.
"""

import asyncio
import os

import pytest

from app.core.config import Settings
from app.llm import build_llm_client
from tests.helpers import RESUME_REQUEST, build_supervisor

settings = Settings()
pytestmark = pytest.mark.skipif(
    not (os.environ.get("RUN_LIVE_LLM_TESTS") == "1" and settings.llm_configured),
    reason="live LLM test: set RUN_LIVE_LLM_TESTS=1 plus GEMINI_API_KEY and GEMINI_MODEL",
)


def test_real_gemini_routes_a_resume_upload(sup_settings):
    live = sup_settings.model_copy(update={"llm_timeout_seconds": 10.0})
    supervisor = build_supervisor(live, llm=build_llm_client(settings))
    resp = asyncio.run(supervisor.handle(RESUME_REQUEST))
    assert resp.routing.source == "llm", resp.routing.fallback_reason
    assert "profile_agent" in resp.routing.agents
