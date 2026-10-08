-- Phase 4 groundwork: embedding storage and the model registry (spec §23, §24).
-- No model is selected here. Phase 6's Model Selection Engine inserts the evidence-based
-- choice (docs/adr/0002). Until then, embedding results are reported as "Not yet evaluated".

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS model_versions (
    id              BIGSERIAL PRIMARY KEY,
    model_kind      TEXT NOT NULL CHECK (model_kind IN ('embedding', 'matching')),
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate', 'selected', 'retired')),
    experiment_id   TEXT,                   -- the experiment whose evidence justified selection (rule 20)
    selected_at     TIMESTAMPTZ,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_kind, model_name, model_version),
    CHECK (status <> 'selected' OR (experiment_id IS NOT NULL AND selected_at IS NOT NULL))
);
-- At most one selected model per kind at any time.
CREATE UNIQUE INDEX IF NOT EXISTS model_versions_one_selected_idx
    ON model_versions (model_kind) WHERE status = 'selected';

-- Spec §23 embedding record: entity_id, entity_type, model_name, model_version, dimension,
-- embedding, created_at. Plus `section`, because matching is section-wise (skills, experience ...).
-- The vector column has no fixed dimension, because candidate models differ (384/768/1024).
-- Per-model ANN indexes are added once a model is selected.
CREATE TABLE IF NOT EXISTS embeddings (
    id              BIGSERIAL PRIMARY KEY,
    entity_id       TEXT NOT NULL,
    entity_type     TEXT NOT NULL CHECK (entity_type IN ('student', 'job', 'course')),
    section         TEXT NOT NULL DEFAULT 'full',
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    dimension       INTEGER NOT NULL CHECK (dimension > 0),
    embedding       vector NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (vector_dims(embedding) = dimension),
    UNIQUE (entity_type, entity_id, section, model_name, model_version)
);
CREATE INDEX IF NOT EXISTS embeddings_lookup_idx ON embeddings (entity_type, model_name, model_version);
