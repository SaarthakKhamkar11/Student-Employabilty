"""Persistence of finished Supervisor runs (the agent_runs table).

Records are PII-scrubbed before storage: emails and phone numbers are masked
and sensitive keys are dropped, but long text is kept. A database failure
must never break a request (spec §21): the Supervisor reports persisted=false.
"""

from datetime import datetime
from typing import Any, Protocol

import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from app.core.config import Settings
from app.core.logging import PII_KEYS, redact


class AgentRunRecord(BaseModel):
    request_id: str
    user_id: str | None
    task_type: str
    student_id: str | None
    job_id: str | None
    status: str
    routing_source: str | None
    model_used: str | None
    agents_called: list[str]
    agent_results: dict[str, Any]
    model_versions: dict[str, str]
    errors: list[dict[str, Any]]
    started_at: datetime
    finished_at: datetime
    duration_ms: float

    def scrubbed(self) -> "AgentRunRecord":
        data = self.model_dump()
        for key in ("agent_results", "errors"):
            data[key] = redact(data[key], max_text=None, keys=PII_KEYS)
        return AgentRunRecord.model_validate(data)


class AgentRunStore(Protocol):
    async def save(self, record: AgentRunRecord) -> None: ...

    async def get(self, request_id: str) -> dict[str, Any] | None: ...


_COLUMNS = list(AgentRunRecord.model_fields)
_JSON_COLUMNS = {"agents_called", "agent_results", "model_versions", "errors"}


class PostgresAgentRunStore:
    def __init__(self, dsn: str, connect_timeout: int = 3):
        self._dsn = dsn
        self._connect_timeout = connect_timeout

    async def save(self, record: AgentRunRecord) -> None:
        row = record.scrubbed().model_dump()
        values = [Jsonb(row[c]) if c in _JSON_COLUMNS else row[c] for c in _COLUMNS]
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _COLUMNS if c != "request_id")
        sql = (f"INSERT INTO agent_runs ({', '.join(_COLUMNS)}) VALUES ({', '.join(['%s'] * len(_COLUMNS))}) "
               f"ON CONFLICT (request_id) DO UPDATE SET {updates}")
        async with await psycopg.AsyncConnection.connect(self._dsn, connect_timeout=self._connect_timeout) as conn:
            await conn.execute(sql, values)

    async def get(self, request_id: str) -> dict[str, Any] | None:
        async with await psycopg.AsyncConnection.connect(self._dsn, connect_timeout=self._connect_timeout) as conn:
            cur = await conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM agent_runs WHERE request_id = %s",
                                     (request_id,))
            row = await cur.fetchone()
        if row is None:
            return None
        return {c: (str(v) if c == "request_id" else v) for c, v in zip(_COLUMNS, row)}


class InMemoryAgentRunStore:
    """For tests and for running without a database."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    async def save(self, record: AgentRunRecord) -> None:
        self.records[record.request_id] = record.scrubbed().model_dump(mode="json")

    async def get(self, request_id: str) -> dict[str, Any] | None:
        return self.records.get(request_id)


def build_store(settings: Settings) -> AgentRunStore | None:
    if not settings.database_url:
        return None
    return PostgresAgentRunStore(settings.database_url.get_secret_value())
