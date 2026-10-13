-- Before Phase 10: the remaining spec §23 tables, plus the constraints the existing tables were missing.
-- Nothing is dropped or recreated (docs/adr/0017).
--
-- PII rule: email, phone and password_hash live ONLY in `users`. No other table has a PII column, and the
-- free-form JSON columns (agent state, messages, audit details, feedback payloads) refuse anything that looks
-- like an email address. That check is the same regex the app's redact() masks with (app/core/logging.py
-- EMAIL_RE), so a row only fails it when scrubbing was skipped.

-- True when the text contains something shaped like an email address.
CREATE OR REPLACE FUNCTION contains_email(t text) RETURNS boolean
    LANGUAGE sql IMMUTABLE PARALLEL SAFE
    AS $$ SELECT t ~ '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' $$;

-- True when a JSON object has a top-level PII field holding a value. redact() keeps the key and writes
-- "[REDACTED]", which passes.
CREATE OR REPLACE FUNCTION has_pii_key(j jsonb) RETURNS boolean
    LANGUAGE sql IMMUTABLE PARALLEL SAFE
    AS $$ SELECT jsonb_typeof(j) = 'object' AND EXISTS (
              SELECT 1 FROM jsonb_each(j) e
              WHERE e.key IN ('email', 'phone', 'phone_number', 'mobile', 'password', 'password_hash')
                AND e.value NOT IN ('null'::jsonb, '"[REDACTED]"'::jsonb)) $$;

-- ---------------------------------------------------------------------------------------------------------
-- Accounts
-- ---------------------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS companies (
    company_id      BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL CHECK (btrim(name) <> ''),
    industry        TEXT,
    website         TEXT CHECK (website ~* '^https?://'),
    country         TEXT,
    source_dataset  TEXT NOT NULL,          -- where the row came from; no contact details are stored here
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS companies_name_idx ON companies (lower(name));

-- The only table with PII. password_hash is a bcrypt or argon2 hash, never plaintext.
CREATE TABLE IF NOT EXISTS users (
    user_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           TEXT NOT NULL CHECK (email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    phone           TEXT CHECK (phone ~ '^\+?[0-9][0-9 ()-]{6,19}$'),
    password_hash   TEXT NOT NULL CHECK (
                        password_hash ~ '^\$2[aby]\$[0-9]{2}\$[./A-Za-z0-9]{53}$'   -- bcrypt
                        OR password_hash ~ '^\$argon2(id|i|d)\$v=[0-9]+\$'),      -- argon2 (PHC string)
    role            TEXT NOT NULL CHECK (role IN ('student', 'recruiter', 'placement_officer', 'admin')),
    student_id      TEXT UNIQUE REFERENCES student_profiles (student_id) ON DELETE SET NULL,
    company_id      BIGINT REFERENCES companies (company_id) ON DELETE SET NULL,
    is_active       BOOLEAN NOT NULL DEFAULT true,
    last_login_at   TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS users_email_idx ON users (lower(email));

-- jobs.company_name stays as the source text. The link is filled only where a company row really exists.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS company_id BIGINT REFERENCES companies (company_id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS jobs_company_id_idx ON jobs (company_id);

COMMENT ON COLUMN jobs.gender_preference IS
    'Visibility filter only, never a scoring feature (docs/adr/0009).';
COMMENT ON COLUMN student_profiles.gender IS
    'Protected attribute: fairness audit only, never a matching feature (spec §11).';

-- ---------------------------------------------------------------------------------------------------------
-- Embeddings: allowed sections, per entity type. Students: the spec §23 list. Jobs: the sections the Job
-- Intelligence Agent stores (app/matching/similarity.py STORED_JOB). Courses: set by Phase 10.
-- ---------------------------------------------------------------------------------------------------------
ALTER TABLE embeddings ADD CONSTRAINT embeddings_section_known CHECK (
    (entity_type = 'student' AND section IN ('skills', 'experience', 'education', 'projects', 'certifications', 'full'))
    OR (entity_type = 'job' AND section IN ('skills', 'required_skills', 'responsibilities', 'description',
                                            'education', 'full'))
    OR entity_type = 'course');

-- ---------------------------------------------------------------------------------------------------------
-- Agent runs: the Supervisor -> agent tree. A Supervisor row has parent_run_id NULL. Each agent it calls
-- may get its own row (its own UUID in request_id) pointing at the Supervisor row. The defaults keep
-- the Phase 3 writer (app/db/store.py) working unchanged.
-- ---------------------------------------------------------------------------------------------------------
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS parent_run_id UUID REFERENCES agent_runs (request_id) ON DELETE CASCADE;
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS agent_name TEXT NOT NULL DEFAULT 'supervisor';
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0);
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS used_fallback BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE agent_runs ADD CONSTRAINT agent_runs_not_own_parent CHECK (parent_run_id <> request_id);
ALTER TABLE agent_runs ADD CONSTRAINT agent_runs_state_redacted CHECK (
    NOT contains_email(agent_results::text) AND NOT contains_email(errors::text)
    AND NOT has_pii_key(agent_results));
CREATE INDEX IF NOT EXISTS agent_runs_parent_idx ON agent_runs (parent_run_id);

CREATE TABLE IF NOT EXISTS agent_messages (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID NOT NULL REFERENCES agent_runs (request_id) ON DELETE CASCADE,
    sequence        INTEGER NOT NULL CHECK (sequence >= 0),
    sender          TEXT NOT NULL,
    recipient       TEXT NOT NULL,
    message_type    TEXT NOT NULL,          -- e.g. tool_call, tool_result, handoff, error
    content         JSONB NOT NULL,         -- PII-redacted before insert
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, sequence),
    CHECK (NOT contains_email(content::text) AND NOT has_pii_key(content))
);

-- ---------------------------------------------------------------------------------------------------------
-- Relevance labels: a view, not a second table (docs/adr/0017). The rules are enforced by the tables it reads:
-- relevance_judgments (label 0-3, label_source) and relevance_pool (split). The three constraints below are
-- the ones those tables were missing.
-- ---------------------------------------------------------------------------------------------------------
ALTER TABLE relevance_pool ADD CONSTRAINT relevance_pool_student_fk
    FOREIGN KEY (student_id) REFERENCES student_profiles (student_id);
ALTER TABLE relevance_pool ADD CONSTRAINT relevance_pool_job_fk
    FOREIGN KEY (job_id) REFERENCES jobs (job_id);
-- One row per pair. A pair belongs to one pool version, so each pair has exactly one split and one label.
ALTER TABLE relevance_pool ADD CONSTRAINT relevance_pool_one_row_per_pair UNIQUE (student_id, job_id);

-- One row per judged pair. Same aggregation as the labelling export (app/labelling/service.py): the median of
-- the labellers' grades, ties rounded down. If labellers' sources differ, the weakest source is reported, so a
-- label is never presented as better evidence than it is. student_fully_judged matches the export's filter.
CREATE OR REPLACE VIEW relevance_labels AS
WITH per_pair AS (
    SELECT p.pool_version, p.student_id, p.job_id, p.split,
           floor(percentile_cont(0.5) WITHIN GROUP (ORDER BY j.relevance_label))::smallint AS relevance_label,
           CASE WHEN bool_or(j.label_source = 'synthetic_clearly_marked') THEN 'synthetic_clearly_marked'
                WHEN bool_or(j.label_source = 'weakly_labelled') THEN 'weakly_labelled'
                ELSE 'human_reviewed' END AS label_source,
           count(*) AS labeller_count,
           array_agg(DISTINCT j.rubric_version ORDER BY j.rubric_version) AS rubric_versions,
           max(j.updated_at) AS labelled_at
    FROM relevance_pool p
    JOIN relevance_judgments j USING (pool_version, student_id, job_id)
    GROUP BY p.pool_version, p.student_id, p.job_id, p.split
)
SELECT pp.student_id, pp.job_id, pp.relevance_label, pp.label_source, pp.split, pp.pool_version,
       pp.labeller_count, pp.rubric_versions, pp.labelled_at,
       NOT EXISTS (SELECT 1 FROM relevance_pool q
                   WHERE q.pool_version = pp.pool_version AND q.student_id = pp.student_id
                     AND NOT EXISTS (SELECT 1 FROM relevance_judgments r
                                     WHERE (r.pool_version, r.student_id, r.job_id)
                                           = (q.pool_version, q.student_id, q.job_id))) AS student_fully_judged
FROM per_pair pp;

-- ---------------------------------------------------------------------------------------------------------
-- Matching output (Phase 7-9 contracts: MatchResult, EmployabilityScore, SkillGapReport)
-- ---------------------------------------------------------------------------------------------------------
-- Every prediction carries the spec §24 metadata (PredictionVersions): all seven fields are required.
-- experiment_id must name a real experiment, as model_versions does.
CREATE TABLE IF NOT EXISTS match_results (
    id                                  BIGSERIAL PRIMARY KEY,
    request_id                          UUID REFERENCES agent_runs (request_id) ON DELETE SET NULL,
    student_id                          TEXT NOT NULL REFERENCES student_profiles (student_id) ON DELETE CASCADE,
    job_id                              TEXT NOT NULL REFERENCES jobs (job_id) ON DELETE CASCADE,
    rank                                INTEGER NOT NULL CHECK (rank >= 1),
    suitability_score                   DOUBLE PRECISION CHECK (suitability_score BETWEEN 0 AND 1),
    -- EmployabilityScore, 0-100 each. NULL = Not yet evaluated, never a guessed number.
    overall_score                       DOUBLE PRECISION CHECK (overall_score BETWEEN 0 AND 100),
    skill_match_score                   DOUBLE PRECISION CHECK (skill_match_score BETWEEN 0 AND 100),
    experience_match_score              DOUBLE PRECISION CHECK (experience_match_score BETWEEN 0 AND 100),
    education_match_score               DOUBLE PRECISION CHECK (education_match_score BETWEEN 0 AND 100),
    project_relevance_score             DOUBLE PRECISION CHECK (project_relevance_score BETWEEN 0 AND 100),
    certification_relevance_score       DOUBLE PRECISION CHECK (certification_relevance_score BETWEEN 0 AND 100),
    semantic_similarity_score           DOUBLE PRECISION CHECK (semantic_similarity_score BETWEEN 0 AND 100),
    career_interest_alignment_score     DOUBLE PRECISION CHECK (career_interest_alignment_score BETWEEN 0 AND 100),
    transferable_skill_alignment_score  DOUBLE PRECISION CHECK (transferable_skill_alignment_score BETWEEN 0 AND 100),
    trace                               JSONB NOT NULL DEFAULT '{}'::jsonb,   -- MatchTrace, for Phase 11
    embedding_model                     TEXT NOT NULL CHECK (btrim(embedding_model) <> ''),
    embedding_version                   TEXT NOT NULL CHECK (btrim(embedding_version) <> ''),
    matching_model                      TEXT NOT NULL CHECK (btrim(matching_model) <> ''),
    matching_model_version              TEXT NOT NULL CHECK (btrim(matching_model_version) <> ''),
    feature_version                     TEXT NOT NULL CHECK (btrim(feature_version) <> ''),
    dataset_version                     TEXT NOT NULL CHECK (btrim(dataset_version) <> ''),
    experiment_id                       TEXT NOT NULL REFERENCES experiment_runs (experiment_id),
    created_at                          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (request_id, student_id, job_id)
);
CREATE INDEX IF NOT EXISTS match_results_student_idx ON match_results (student_id, created_at DESC);
CREATE INDEX IF NOT EXISTS match_results_job_idx ON match_results (job_id, created_at DESC);

-- One row per missing or partially related job skill. Matched skills are not gaps and are not stored here.
CREATE TABLE IF NOT EXISTS skill_gaps (
    id                      BIGSERIAL PRIMARY KEY,
    match_result_id         BIGINT REFERENCES match_results (id) ON DELETE CASCADE,   -- NULL for a standalone report
    student_id              TEXT NOT NULL REFERENCES student_profiles (student_id) ON DELETE CASCADE,
    job_id                  TEXT NOT NULL REFERENCES jobs (job_id) ON DELETE CASCADE,
    skill                   TEXT NOT NULL,      -- normalised job skill
    gap_type                TEXT NOT NULL CHECK (gap_type IN ('missing_required', 'missing_preferred', 'partially_related')),
    importance_score        DOUBLE PRECISION,   -- counterfactual re-scoring; NULL = Not yet evaluated
    related_student_skill   TEXT,               -- transferable candidate or partially related skill
    similarity              DOUBLE PRECISION CHECK (similarity BETWEEN -1 AND 1),
    importance_model        TEXT NOT NULL,
    similarity_model        TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (match_result_id, skill, gap_type),
    CHECK ((related_student_skill IS NULL) = (similarity IS NULL))
);
CREATE INDEX IF NOT EXISTS skill_gaps_pair_idx ON skill_gaps (student_id, job_id);

CREATE TABLE IF NOT EXISTS career_recommendations (
    id              BIGSERIAL PRIMARY KEY,
    request_id      UUID REFERENCES agent_runs (request_id) ON DELETE SET NULL,
    student_id      TEXT NOT NULL REFERENCES student_profiles (student_id) ON DELETE CASCADE,
    recommended_role TEXT NOT NULL,
    rank            INTEGER NOT NULL CHECK (rank >= 1),
    score           DOUBLE PRECISION CHECK (score BETWEEN 0 AND 100),   -- NULL = Not yet evaluated
    evidence        JSONB NOT NULL,         -- what the recommendation is traceable to (spec §16)
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS career_recommendations_student_idx ON career_recommendations (student_id, created_at DESC);

-- ---------------------------------------------------------------------------------------------------------
-- Learning (Phase 10). Course rows come from a real catalogue, never invented (spec §17). NULL in an
-- optional column means "not available in source data".
-- ---------------------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS learning_resources (
    resource_id         TEXT PRIMARY KEY,
    title               TEXT NOT NULL CHECK (btrim(title) <> ''),
    url                 TEXT NOT NULL UNIQUE CHECK (url ~* '^https?://'),
    provider            TEXT,
    description         TEXT,
    skills_covered      TEXT[] NOT NULL DEFAULT '{}',
    difficulty          TEXT,
    prerequisites       TEXT,
    estimated_effort    TEXT,               -- as written in the source; not converted by guesswork
    source_reference    TEXT NOT NULL CHECK (btrim(source_reference) <> ''),   -- where the row came from
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learning_resources_skills_idx ON learning_resources USING gin (skills_covered);

CREATE TABLE IF NOT EXISTS learning_recommendations (
    id              BIGSERIAL PRIMARY KEY,
    request_id      UUID REFERENCES agent_runs (request_id) ON DELETE SET NULL,
    student_id      TEXT NOT NULL REFERENCES student_profiles (student_id) ON DELETE CASCADE,
    job_id          TEXT REFERENCES jobs (job_id) ON DELETE CASCADE,
    skill_gap_id    BIGINT REFERENCES skill_gaps (id) ON DELETE SET NULL,
    skill           TEXT NOT NULL,
    -- NULL = no resource found for that skill. RESTRICT, so deleting a course never turns into a false "none found".
    resource_id     TEXT REFERENCES learning_resources (resource_id) ON DELETE RESTRICT,
    priority        INTEGER NOT NULL CHECK (priority >= 1),
    reason          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learning_recommendations_student_idx ON learning_recommendations (student_id, created_at DESC);

-- ---------------------------------------------------------------------------------------------------------
-- Feedback and audit
-- ---------------------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feedback_events (
    id              BIGSERIAL PRIMARY KEY,
    source          TEXT NOT NULL CHECK (source IN ('real_deployment', 'simulated_research')),
    event_type      TEXT NOT NULL,          -- e.g. rating, click, apply, dismiss
    user_id         UUID REFERENCES users (user_id) ON DELETE SET NULL,
    student_id      TEXT REFERENCES student_profiles (student_id) ON DELETE CASCADE,
    job_id          TEXT REFERENCES jobs (job_id) ON DELETE CASCADE,
    match_result_id BIGINT REFERENCES match_results (id) ON DELETE SET NULL,
    rating          SMALLINT,
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (NOT contains_email(payload::text) AND NOT has_pii_key(payload))
);
CREATE INDEX IF NOT EXISTS feedback_events_source_idx ON feedback_events (source, created_at DESC);

-- Append-only. actor_user_id has no foreign key: the audit trail must outlive a deleted account, and a
-- SET NULL cascade would be an UPDATE, which the trigger refuses.
CREATE TABLE IF NOT EXISTS audit_logs (
    id              BIGSERIAL PRIMARY KEY,
    actor_type      TEXT NOT NULL CHECK (actor_type IN ('user', 'agent', 'system')),
    actor_user_id   UUID,
    action          TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    entity_id       TEXT,
    request_id      UUID,
    details         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (NOT contains_email(details::text) AND NOT has_pii_key(details))
);
CREATE INDEX IF NOT EXISTS audit_logs_entity_idx ON audit_logs (entity_type, entity_id, created_at DESC);

CREATE OR REPLACE FUNCTION audit_logs_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'audit_logs is append-only (% refused)', TG_OP;
END $$;
DROP TRIGGER IF EXISTS audit_logs_append_only ON audit_logs;
CREATE TRIGGER audit_logs_append_only BEFORE UPDATE OR DELETE ON audit_logs
    FOR EACH ROW EXECUTE FUNCTION audit_logs_append_only();
