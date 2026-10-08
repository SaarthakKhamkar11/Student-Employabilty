"""FastAPI entrypoint.

Phase 3: the Supervisor is live behind POST /requests. The specialist agents
are still shells, so every result reads "Not yet evaluated" or "Insufficient
evidence" until their phases. There is no authentication yet (Phase 16):
ports bind to 127.0.0.1 only.
"""

import asyncio
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException

from app.agents import SupervisorAgent, build_agents
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.store import build_store
from app.llm import build_llm_client
from app.schemas.agent_io import SupervisorInput, SupervisorResponse

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title="Student Employability Intelligence API", version="0.3.0")
# Fails at startup if an agent declares an unregistered tool.
agents = build_agents(llm=build_llm_client(settings), store=build_store(settings))


def get_supervisor() -> SupervisorAgent:
    return agents["supervisor"]  # type: ignore[return-value]


async def _reachable(url: str | None, default_port: int) -> str:
    """TCP reachability only; credentials in the URL are never returned or logged."""
    if not url:
        return "not_configured"
    parts = urlsplit(url)
    if not parts.hostname:
        return "invalid_url"
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(parts.hostname, parts.port or default_port), 1.0)
    except (OSError, TimeoutError):
        return "unreachable"
    writer.close()
    return "reachable"


@app.get("/health")
async def health() -> dict:
    db_url = settings.database_url.get_secret_value() if settings.database_url else None
    postgres, redis = await asyncio.gather(_reachable(db_url, 5432), _reachable(settings.redis_url, 6379))
    return {
        "status": "ok",
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "llm_key_configured": settings.llm_configured,
        # LLM routing is used only when configured; otherwise the deterministic router runs (spec §22)
        "llm_calls_enabled": settings.llm_configured,
        "dependencies": {"postgres": postgres, "redis": redis},
    }


@app.get("/agents")
async def list_agents() -> list[dict]:
    return [
        {
            "name": agent.name,
            "implemented": agent.implemented,
            "implemented_in_phase": agent.implemented_in_phase,
            "allowed_tools": list(agent.allowed_tools),
            "allowed_delegates": list(agent.allowed_delegates),
            "planned_tools": list(agent.planned_tools),
            "input_schema": agent.input_schema.__name__,
            "output_schema": agent.output_schema.__name__,
            "role_description": agent.role_description,
        }
        for agent in agents.values()
    ]


@app.post("/requests", response_model=SupervisorResponse)
async def create_request(body: SupervisorInput, supervisor: SupervisorAgent = Depends(get_supervisor)):
    return await supervisor.handle(body)


@app.get("/requests/{request_id}")
async def get_request(request_id: UUID, supervisor: SupervisorAgent = Depends(get_supervisor)) -> dict:
    if supervisor.store is None:
        raise HTTPException(503, "persistence is not configured")
    try:
        record = await supervisor.store.get(str(request_id))
    except Exception:
        raise HTTPException(503, "database unavailable") from None
    if record is None:
        raise HTTPException(404, "unknown request_id")
    return record
