"""agent_runs persistence. The Postgres test runs only when TEST_DATABASE_URL is set."""

import asyncio
import os
import uuid
from datetime import datetime, timezone

import pytest

from app.db.store import AgentRunRecord, InMemoryAgentRunStore, PostgresAgentRunStore


def record(**overrides) -> AgentRunRecord:
    now = datetime.now(timezone.utc)
    data = dict(request_id=str(uuid.uuid4()), user_id=None, task_type="ingest_student_profile",
                student_id="S00001", job_id=None, status="failed", routing_source="deterministic_fallback",
                model_used="deterministic_router", agents_called=["supervisor", "profile_agent"],
                agent_results={"profile_agent": {"status": "failed", "data": "Not yet evaluated",
                                                 "note": "contact asha.k@college.edu"}},
                model_versions={"embedding_model": "Not yet evaluated"}, errors=[{"code": "not_implemented"}],
                started_at=now, finished_at=now, duration_ms=1.5)
    data.update(overrides)
    return AgentRunRecord(**data)


def test_records_are_pii_scrubbed_before_storage():
    store = InMemoryAgentRunStore()
    rec = record()
    asyncio.run(store.save(rec))
    stored = asyncio.run(store.get(rec.request_id))
    assert stored["agent_results"]["profile_agent"]["note"] == "contact [EMAIL]"


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to run against Postgres")
def test_postgres_migration_and_round_trip():
    import psycopg

    from app.db.migrate import apply_migrations

    dsn = os.environ["TEST_DATABASE_URL"]
    apply_migrations(dsn)
    assert apply_migrations(dsn) == []  # idempotent: nothing left to apply

    store, rec = PostgresAgentRunStore(dsn), record()
    try:
        asyncio.run(store.save(rec))
        asyncio.run(store.save(rec.model_copy(update={"status": "partial"})))  # upsert, not duplicate
        row = asyncio.run(store.get(rec.request_id))
        assert row["request_id"] == rec.request_id and row["status"] == "partial"
        assert row["agents_called"] == ["supervisor", "profile_agent"]
        assert row["agent_results"]["profile_agent"]["note"] == "contact [EMAIL]"
        assert asyncio.run(store.get(str(uuid.uuid4()))) is None
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("DELETE FROM agent_runs WHERE request_id = %s", (rec.request_id,))
