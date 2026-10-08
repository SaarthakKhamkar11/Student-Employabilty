-- Phase 5: job tables (spec §23) and sparse storage for TF-IDF embeddings.

-- jobs mirrors config/schema.py JOB_SCHEMA 1:1 (python -m app.db.seed_jobs loads data/processed/jobs.csv).
-- Uploaded job descriptions may lack dataset columns: those stay NULL (never guessed).
CREATE TABLE IF NOT EXISTS jobs (
    job_id                  TEXT PRIMARY KEY,
    source_job_id           TEXT,
    experience_required     TEXT,
    qualification_required  TEXT,
    salary_range            TEXT,
    location                TEXT,
    country                 TEXT,
    work_type               TEXT,
    job_title               TEXT,
    role                    TEXT,
    job_portal              TEXT,
    job_description         TEXT,
    skills                  TEXT,
    responsibilities        TEXT,
    company_name            TEXT,
    gender_preference       TEXT,   -- visibility filter only, never a scoring feature (docs/adr/0009)
    source_dataset          TEXT NOT NULL DEFAULT 'kaggle_job_descriptions',
    -- derived by the Job Intelligence Agent
    posting                 JSONB,
    posting_extractor_version TEXT,
    posting_updated_at      TIMESTAMPTZ,
    min_degree              TEXT,
    min_degree_level        SMALLINT CHECK (min_degree_level BETWEEN 1 AND 4),
    min_experience_years    DOUBLE PRECISION CHECK (min_experience_years >= 0),
    max_experience_years    DOUBLE PRECISION CHECK (max_experience_years >= 0),
    near_duplicate_of       TEXT REFERENCES jobs (job_id),   -- docs/adr/0010
    pii_flags               JSONB,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_company_title_idx ON jobs (lower(company_name), lower(job_title));

CREATE TABLE IF NOT EXISTS job_skills (
    id                  BIGSERIAL PRIMARY KEY,
    job_id              TEXT NOT NULL REFERENCES jobs (job_id) ON DELETE CASCADE,
    skill_raw           TEXT NOT NULL,
    skill_normalized    TEXT,
    skill_type          TEXT NOT NULL DEFAULT 'unclassified' CHECK (skill_type IN ('technical', 'soft', 'unclassified')),
    requirement         TEXT NOT NULL CHECK (requirement IN ('required', 'preferred')),   -- INFERRED (docs/adr/0008)
    requirement_basis   TEXT NOT NULL,
    source              TEXT NOT NULL,
    extractor_version   TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (job_id, skill_raw, skill_normalized, extractor_version)
);
CREATE INDEX IF NOT EXISTS job_skills_normalized_idx ON job_skills (skill_normalized);

CREATE TABLE IF NOT EXISTS job_responsibilities (
    id                  BIGSERIAL PRIMARY KEY,
    job_id              TEXT NOT NULL REFERENCES jobs (job_id) ON DELETE CASCADE,
    position            INTEGER NOT NULL,
    description         TEXT NOT NULL,
    source              TEXT NOT NULL,
    extractor_version   TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS job_responsibilities_job_idx ON job_responsibilities (job_id);

-- TF-IDF vectors are ~99% zeros: storing them dense would take ~16 KB per vector (about 480 MB for all
-- job sections). They now go in a sparse column. Dense models (Phase 6) keep using `embedding`.
-- Exactly one of the two columns is set (docs/adr/0011).
ALTER TABLE embeddings ALTER COLUMN embedding DROP NOT NULL;
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS embedding_sparse sparsevec;
ALTER TABLE embeddings ADD CONSTRAINT embeddings_one_representation
    CHECK ((embedding IS NULL) <> (embedding_sparse IS NULL));
