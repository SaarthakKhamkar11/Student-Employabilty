-- Phase 13: one user store, and what the dashboards need to persist.
--
-- 1. Auth moves onto migration 0009's `users` table (the only PII table). The auth module's own tables
--    (app_users, student_accounts, recruiter_accounts) are no longer created; if an older database has them,
--    their accounts are copied over once (same email, same bcrypt hash) and the old tables are left untouched.
-- 2. Onboarding state: when a student confirmed the parsed profile, when a company confirmed a parsed job.
-- 3. Cached Gemini explanations (Phase 13 rule: stored explanation by default; the LLM only on request, and a
--    successful answer is never requested twice for the same prediction).

ALTER TABLE users ADD COLUMN IF NOT EXISTS full_name TEXT CHECK (full_name IS NULL OR btrim(full_name) <> '');
-- Signup details that are not part of the matching profile (college/university, recruiter designation).
ALTER TABLE users ADD COLUMN IF NOT EXISTS details JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE users ADD COLUMN IF NOT EXISTS profile_confirmed_at TIMESTAMPTZ;

ALTER TABLE companies ADD COLUMN IF NOT EXISTS company_size TEXT;

-- Jobs a company posted through the portal, and when the company confirmed the parsed structure.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS posted_by UUID REFERENCES users (user_id) ON DELETE SET NULL;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS confirmed_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS jobs_posted_by_idx ON jobs (posted_by);

-- Shortlist decisions are feedback_events rows (event_type shortlist | unshortlist); this index serves the
-- "current shortlist of a job" query.
CREATE INDEX IF NOT EXISTS feedback_events_job_idx ON feedback_events (job_id, student_id, created_at DESC);

CREATE TABLE IF NOT EXISTS llm_explanations (
    id                      BIGSERIAL PRIMARY KEY,
    student_id              TEXT NOT NULL REFERENCES student_profiles (student_id) ON DELETE CASCADE,
    job_id                  TEXT NOT NULL REFERENCES jobs (job_id) ON DELETE CASCADE,
    matching_model_version  TEXT NOT NULL,
    embedding_version       TEXT NOT NULL,
    llm_model               TEXT NOT NULL,
    request_id              UUID,              -- the Supervisor request that produced it (its trace)
    explanation             JSONB NOT NULL,    -- the validated MatchExplanation (source = llm_validated)
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (student_id, job_id, matching_model_version, embedding_version),
    CHECK (NOT contains_email(explanation::text) AND NOT has_pii_key(explanation))
);

-- One-time copy from the pre-Phase-13 auth tables, when they exist.
DO $$
BEGIN
    IF to_regclass('public.app_users') IS NOT NULL THEN
        INSERT INTO users (email, password_hash, role, full_name, is_active, last_login_at, created_at)
        SELECT a.email, a.password_hash, a.role, a.full_name, a.is_active, a.last_login_at, a.created_at
        FROM app_users a
        WHERE NOT EXISTS (SELECT 1 FROM users u WHERE lower(u.email) = lower(a.email));
    END IF;
END $$;
