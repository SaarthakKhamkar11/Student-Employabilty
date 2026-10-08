-- Phase 3: one row per Supervisor request (spec §5.1 state, §19, §23 "agent_runs").
-- Values are PII-scrubbed before insert (emails and phone numbers masked).
CREATE TABLE IF NOT EXISTS agent_runs (
    request_id      UUID PRIMARY KEY,
    user_id         TEXT,
    task_type       TEXT NOT NULL,
    student_id      TEXT,
    job_id          TEXT,
    status          TEXT NOT NULL CHECK (status IN ('pending', 'running', 'success', 'partial', 'failed')),
    routing_source  TEXT CHECK (routing_source IN ('llm', 'deterministic_fallback')),
    model_used      TEXT,
    agents_called   JSONB NOT NULL,
    agent_results   JSONB NOT NULL,
    model_versions  JSONB NOT NULL,
    errors          JSONB NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL,
    finished_at     TIMESTAMPTZ NOT NULL,
    duration_ms     DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_runs_task_created_idx ON agent_runs (task_type, created_at DESC);
CREATE INDEX IF NOT EXISTS agent_runs_status_idx ON agent_runs (status);
