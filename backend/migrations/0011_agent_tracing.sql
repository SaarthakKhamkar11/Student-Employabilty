-- Phase 12: agent tracing (spec §39, docs/adr/0020).
--
-- agent_runs already holds the Supervisor -> agent tree (0009: parent_run_id, agent_name, retry_count,
-- used_fallback). Phase 12 writes it: the Supervisor's row is the request (request_id = the request), and each
-- agent EXECUTION gets its own row (its own UUID, parent_run_id = the Supervisor row). A Supervisor retry of an
-- agent is a second row with attempt 2. These columns hold the spec §39 fields that were only in JSON before.
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS attempt INTEGER NOT NULL DEFAULT 1 CHECK (attempt >= 1);
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS termination_reason TEXT;
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS tool_call_count INTEGER NOT NULL DEFAULT 0 CHECK (tool_call_count >= 0);
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS tool_latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0
    CHECK (tool_latency_ms >= 0);
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS model_version TEXT;
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS error_code TEXT;
CREATE INDEX IF NOT EXISTS agent_runs_agent_created_idx ON agent_runs (agent_name, created_at DESC);

-- agent_messages (0009) gets one row per thing an agent run did: each tool attempt, each LLM call, and each agent
-- the Supervisor decided to skip. The span ids rebuild the nesting (an LLM call inside a tool attempt).
-- content is PII-redacted before insert, and 0009's CHECK still rejects emails and PII keys.
ALTER TABLE agent_messages ADD COLUMN IF NOT EXISTS span_id TEXT;
ALTER TABLE agent_messages ADD COLUMN IF NOT EXISTS parent_span_id TEXT;
ALTER TABLE agent_messages ADD COLUMN IF NOT EXISTS status TEXT;
ALTER TABLE agent_messages ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ;
ALTER TABLE agent_messages ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ;
ALTER TABLE agent_messages ADD COLUMN IF NOT EXISTS duration_ms DOUBLE PRECISION CHECK (duration_ms >= 0);
ALTER TABLE agent_messages ADD CONSTRAINT agent_messages_type_known
    CHECK (message_type IN ('tool_call', 'tool_result', 'llm_call', 'handoff', 'skip', 'error'));
CREATE INDEX IF NOT EXISTS agent_messages_type_idx ON agent_messages (message_type, created_at DESC);
