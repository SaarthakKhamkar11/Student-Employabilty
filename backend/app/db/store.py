"""Persistence of finished Supervisor runs: agent_runs (the request and every agent execution) and agent_messages
(each tool attempt, LLM call and skipped agent), Phase 3 + Phase 12.

Records are PII-scrubbed before storage: emails and phone numbers are masked, sensitive keys are dropped, and the
request's own detected PII values are replaced (app/core/logging.py), but long text is kept. A database failure
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
    """One agent_runs row. The Supervisor's row is the request; an agent execution's row points at it."""

    request_id: str  # the request (Supervisor row), or this execution's own id (agent row)
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
    # Phase 12 (migrations 0009 + 0011): the execution tree and the spec §39 fields
    parent_run_id: str | None = None
    agent_name: str = "supervisor"
    attempt: int = 1
    termination_reason: str | None = None
    retry_count: int = 0
    used_fallback: bool = False
    tool_call_count: int = 0
    tool_latency_ms: float = 0.0
    model_version: str | None = None
    error_code: str | None = None

    def scrubbed(self) -> "AgentRunRecord":
        data = self.model_dump()
        for key in ("agent_results", "errors"):
            data[key] = redact(data[key], max_text=None, keys=PII_KEYS)
        for key in ("termination_reason", "model_used", "model_version", "error_code"):
            data[key] = redact(data[key], max_text=None, keys=PII_KEYS)
        return AgentRunRecord.model_validate(data)


class AgentMessageRecord(BaseModel):
    """One agent_messages row: something an agent execution did (a tool attempt, an LLM call) or a skip."""

    run_id: str  # the agent_runs row it belongs to
    sequence: int
    sender: str
    recipient: str
    message_type: str  # tool_call | llm_call | skip
    span_id: str
    parent_span_id: str | None
    status: str
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    content: dict[str, Any]

    def scrubbed(self) -> "AgentMessageRecord":
        data = self.model_dump()
        data["content"] = redact(data["content"], max_text=None, keys=PII_KEYS)
        return AgentMessageRecord.model_validate(data)


class TraceRecords(BaseModel):
    """Everything one request wrote: its agent_runs rows (the Supervisor's first) and its agent_messages rows."""

    runs: list[dict[str, Any]]
    messages: list[dict[str, Any]]


class AgentRunStore(Protocol):
    async def save(self, record: AgentRunRecord, executions: list[AgentRunRecord] | None = None,
                   messages: list[AgentMessageRecord] | None = None) -> None: ...

    async def get(self, request_id: str) -> dict[str, Any] | None: ...

    async def get_trace(self, request_id: str) -> TraceRecords | None: ...


_COLUMNS = list(AgentRunRecord.model_fields)
_JSON_COLUMNS = {"agents_called", "agent_results", "model_versions", "errors"}
_UUID_COLUMNS = {"request_id", "parent_run_id"}
_MESSAGE_COLUMNS = list(AgentMessageRecord.model_fields)


def _row(record: AgentRunRecord) -> list:
    row = record.scrubbed().model_dump()
    return [Jsonb(row[c]) if c in _JSON_COLUMNS else row[c] for c in _COLUMNS]


class PostgresAgentRunStore:
    def __init__(self, dsn: str, connect_timeout: int = 3):
        self._dsn = dsn
        self._connect_timeout = connect_timeout

    async def save(self, record: AgentRunRecord, executions: list[AgentRunRecord] | None = None,
                   messages: list[AgentMessageRecord] | None = None) -> None:
        """One transaction: the request row, then its execution rows, then their messages (foreign keys)."""
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _COLUMNS if c != "request_id")
        placeholders = ", ".join(["%s"] * len(_COLUMNS))
        upsert = (f"INSERT INTO agent_runs ({', '.join(_COLUMNS)}) VALUES ({placeholders}) "
                  f"ON CONFLICT (request_id) DO UPDATE SET {updates}")
        insert_message = (f"INSERT INTO agent_messages ({', '.join(_MESSAGE_COLUMNS)}) "
                          f"VALUES ({', '.join(['%s'] * len(_MESSAGE_COLUMNS))}) "
                          "ON CONFLICT (run_id, sequence) DO NOTHING")
        async with await psycopg.AsyncConnection.connect(self._dsn, connect_timeout=self._connect_timeout) as conn:
            async with conn.transaction():
                await conn.execute(upsert, _row(record))
                async with conn.cursor() as cur:
                    if executions:
                        await cur.executemany(upsert, [_row(r) for r in executions])
                    if messages:
                        rows = [m.scrubbed().model_dump() for m in messages]
                        await cur.executemany(insert_message, [
                            [Jsonb(r[c]) if c == "content" else r[c] for c in _MESSAGE_COLUMNS] for r in rows])

    async def get(self, request_id: str) -> dict[str, Any] | None:
        async with await psycopg.AsyncConnection.connect(self._dsn, connect_timeout=self._connect_timeout) as conn:
            cur = await conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM agent_runs WHERE request_id = %s",
                                     (request_id,))
            row = await cur.fetchone()
        if row is None:
            return None
        return _as_dict(row)

    async def get_trace(self, request_id: str) -> TraceRecords | None:
        """The request row and every row below it (recursive: an agent's own sub-executions, if any)."""
        cols = ", ".join(f"r.{c}" for c in _COLUMNS)
        sql = (f"WITH RECURSIVE tree AS (SELECT {cols} FROM agent_runs r WHERE r.request_id = %s "
               f"UNION ALL SELECT {cols} FROM agent_runs r JOIN tree t ON r.parent_run_id = t.request_id) "
               f"SELECT {', '.join(_COLUMNS)} FROM tree ORDER BY started_at, attempt")
        async with await psycopg.AsyncConnection.connect(self._dsn, connect_timeout=self._connect_timeout) as conn:
            runs = [_as_dict(r) for r in await (await conn.execute(sql, (request_id,))).fetchall()]
            if not runs:
                return None
            ids = [r["request_id"] for r in runs]
            cur = await conn.execute(f"SELECT {', '.join(_MESSAGE_COLUMNS)} FROM agent_messages "
                                     "WHERE run_id = ANY(%s::uuid[]) ORDER BY run_id, sequence", (ids,))
            messages = [{c: (str(v) if c == "run_id" else v) for c, v in zip(_MESSAGE_COLUMNS, row)}
                        for row in await cur.fetchall()]
        return TraceRecords(runs=runs, messages=messages)


def _as_dict(row) -> dict[str, Any]:
    return {c: (str(v) if c in _UUID_COLUMNS and v is not None else v) for c, v in zip(_COLUMNS, row)}


class InMemoryAgentRunStore:
    """For tests and for running without a database. Same rows, same scrubbing as Postgres."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}  # request rows, by request_id
        self.executions: dict[str, dict[str, Any]] = {}  # agent execution rows, by their own id
        self.messages: list[dict[str, Any]] = []

    async def save(self, record: AgentRunRecord, executions: list[AgentRunRecord] | None = None,
                   messages: list[AgentMessageRecord] | None = None) -> None:
        self.records[record.request_id] = record.scrubbed().model_dump(mode="json")
        for r in executions or []:
            self.executions[r.request_id] = r.scrubbed().model_dump(mode="json")
        self.messages += [m.scrubbed().model_dump(mode="json") for m in messages or []]

    async def get(self, request_id: str) -> dict[str, Any] | None:
        return self.records.get(request_id)

    async def get_trace(self, request_id: str) -> TraceRecords | None:
        root = self.records.get(request_id)
        if root is None:
            return None
        runs, todo = [root], [request_id]
        while todo:
            parent = todo.pop()
            children = [r for r in self.executions.values() if r["parent_run_id"] == parent]
            runs += children
            todo += [c["request_id"] for c in children]
        ids = {r["request_id"] for r in runs}
        return TraceRecords(runs=runs, messages=[m for m in self.messages if m["run_id"] in ids])


def build_store(settings: Settings) -> AgentRunStore | None:
    if not settings.database_url:
        return None
    return PostgresAgentRunStore(settings.database_url.get_secret_value())
