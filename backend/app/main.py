"""FastAPI entrypoint.

Phase 3: the Supervisor is live behind POST /requests. Phase 4: the Profile
Agent is live, including POST /requests/resume-file for .pdf/.docx/.txt
uploads. The other specialists are still shells, so their results read "Not
yet evaluated" or "Insufficient evidence" until their phases. There is no
authentication yet (Phase 16): ports bind to 127.0.0.1 only.
"""

import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from app.agents import SupervisorAgent, build_agents
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger, log_event
from app.db.store import build_store
from app.embeddings.provider import build_embedding_provider
from app.llm import build_llm_client
from app.profile.text_extraction import MAX_FILE_BYTES
from app.schemas.agent_io import SupervisorInput, SupervisorResponse

settings = get_settings()
configure_logging(settings.log_level)

embedder = build_embedding_provider(settings)
# Fails at startup if an agent declares an unregistered tool.
agents = build_agents(llm=build_llm_client(settings), store=build_store(settings), embedder=embedder)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fit the embedding model once at startup, so the first request doesn't pay for it.
    try:
        version, dimension = await asyncio.to_thread(embedder.describe)
        log_event(get_logger("main"), "embedding_model_ready", model_name=embedder.model_name,
                  model_version=version, dimension=dimension, interim=embedder.interim)
    except Exception as exc:  # the API still serves; profiles are then saved without embeddings
        log_event(get_logger("main"), "embedding_model_unavailable", logging.ERROR, error=type(exc).__name__)
    yield


app = FastAPI(title="Student Employability Intelligence API", version="0.4.0", lifespan=lifespan)


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


@app.post("/requests/resume-file", response_model=SupervisorResponse)
async def upload_resume_file(file: UploadFile = File(...), student_id: str | None = Form(None),
                             supervisor: SupervisorAgent = Depends(get_supervisor)):
    """Upload a .pdf, .docx or .txt resume. The Profile Agent's extract_text tool reads it."""
    data = await file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"file larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
    request = SupervisorInput(actor="student", request_kind="resume_upload", student_id=student_id or None,
                              resume_file_b64=base64.b64encode(data).decode(), resume_filename=file.filename)
    return await supervisor.handle(request)


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
