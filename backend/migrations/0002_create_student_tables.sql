-- Phase 4 groundwork: student tables (spec §23).
-- student_profiles mirrors config/schema.py STUDENT_SCHEMA 1:1, so the processed CSV loads
-- without a translation layer (python -m app.db.seed). Blank source fields stay NULL: never guessed.

CREATE TABLE IF NOT EXISTS student_profiles (
    student_id          TEXT PRIMARY KEY,
    degree              TEXT,
    branch              TEXT NOT NULL,
    graduation_year     INTEGER,
    gender              TEXT,               -- protected attribute: fairness audit only, never a matching feature (spec §11)
    avg_cgpa_sem5       DOUBLE PRECISION CHECK (avg_cgpa_sem5 BETWEEN 0 AND 10),
    achievements        TEXT,
    career_interests    TEXT NOT NULL,
    domain              TEXT NOT NULL,
    internship_raw      TEXT,
    technical_projects  TEXT,
    skills_raw          TEXT,
    certifications      TEXT,
    country             TEXT,
    source_dataset      TEXT NOT NULL,
    is_synthetic        BOOLEAN NOT NULL,
    synthetic_fields    TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Structured extractions, filled in by the Profile Agent (Phase 4). Each row records which
-- extractor produced it, so re-running a newer extractor never silently overwrites provenance.
CREATE TABLE IF NOT EXISTS student_skills (
    id                  BIGSERIAL PRIMARY KEY,
    student_id          TEXT NOT NULL REFERENCES student_profiles (student_id) ON DELETE CASCADE,
    skill_raw           TEXT NOT NULL,      -- as written in the source
    skill_normalized    TEXT,               -- canonical name; NULL until normalised
    skill_type          TEXT NOT NULL DEFAULT 'unclassified'
                        CHECK (skill_type IN ('technical', 'soft', 'unclassified')),
    source              TEXT NOT NULL,      -- which field or document it came from
    extractor_version   TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (student_id, skill_raw, source, extractor_version)
);
CREATE INDEX IF NOT EXISTS student_skills_normalized_idx ON student_skills (skill_normalized);

CREATE TABLE IF NOT EXISTS student_projects (
    id                  BIGSERIAL PRIMARY KEY,
    student_id          TEXT NOT NULL REFERENCES student_profiles (student_id) ON DELETE CASCADE,
    description         TEXT NOT NULL,
    source              TEXT NOT NULL,
    extractor_version   TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS student_certifications (
    id                  BIGSERIAL PRIMARY KEY,
    student_id          TEXT NOT NULL REFERENCES student_profiles (student_id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    source              TEXT NOT NULL,
    extractor_version   TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS student_experience (
    id                  BIGSERIAL PRIMARY KEY,
    student_id          TEXT NOT NULL REFERENCES student_profiles (student_id) ON DELETE CASCADE,
    description         TEXT NOT NULL,
    organisation        TEXT,
    role                TEXT,
    duration_text       TEXT,               -- as written; not converted into a number by guesswork
    source              TEXT NOT NULL,
    extractor_version   TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS student_projects_student_idx ON student_projects (student_id);
CREATE INDEX IF NOT EXISTS student_certifications_student_idx ON student_certifications (student_id);
CREATE INDEX IF NOT EXISTS student_experience_student_idx ON student_experience (student_id);
