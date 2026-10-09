-- Phase 6: the human relevance benchmark (labelling page) and experiment tracking (spec §8, §25).

-- The student-job pairs shown on the labelling page (built by src/evaluation/labelling_pool.py).
-- Which model nominated a pair is deliberately not stored here: labellers must not see it.
CREATE TABLE IF NOT EXISTS relevance_pool (
    pool_version    TEXT NOT NULL,
    student_id      TEXT NOT NULL,
    job_id          TEXT NOT NULL,
    split           TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    display_order   INTEGER NOT NULL CHECK (display_order > 0),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (pool_version, student_id, job_id)
);

-- One row per person per pair. The export combines them (median, ties rounded down).
CREATE TABLE IF NOT EXISTS relevance_judgments (
    id              BIGSERIAL PRIMARY KEY,
    pool_version    TEXT NOT NULL,
    student_id      TEXT NOT NULL,
    job_id          TEXT NOT NULL,
    labeller        TEXT NOT NULL CHECK (labeller ~ '^[A-Z]{2,8}$'),  -- initials only
    relevance_label SMALLINT NOT NULL CHECK (relevance_label BETWEEN 0 AND 3),
    label_source    TEXT NOT NULL DEFAULT 'human_reviewed'
                    CHECK (label_source IN ('human_reviewed', 'weakly_labelled', 'synthetic_clearly_marked')),
    rubric_version  TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (pool_version, student_id, job_id, labeller),
    FOREIGN KEY (pool_version, student_id, job_id) REFERENCES relevance_pool ON DELETE CASCADE
);

-- Spec §25: every experiment records these fields.
CREATE TABLE IF NOT EXISTS experiment_runs (
    experiment_id     TEXT PRIMARY KEY,
    kind              TEXT NOT NULL CHECK (kind IN ('embedding_selection', 'matching_selection', 'evaluation')),
    started_at        TIMESTAMPTZ NOT NULL,
    finished_at       TIMESTAMPTZ NOT NULL,
    dataset_version   TEXT NOT NULL,
    dataset_size      JSONB NOT NULL,
    train_split       JSONB NOT NULL,
    validation_split  JSONB NOT NULL,
    test_split        JSONB NOT NULL,
    random_seed       INTEGER NOT NULL,
    feature_version   TEXT NOT NULL,
    embedding_model   TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    prediction_model  TEXT NOT NULL,
    hyperparameters   JSONB NOT NULL,
    metrics           JSONB NOT NULL,
    runtime_seconds   DOUBLE PRECISION NOT NULL,
    memory            JSONB NOT NULL,
    hardware          JSONB NOT NULL,
    labels            JSONB NOT NULL,  -- file, sha256, source breakdown, evaluation-validity line
    notes             TEXT NOT NULL,
    limitations       TEXT NOT NULL,
    report            JSONB NOT NULL,  -- the engine's full output: tables, sensitivity, best-model lines
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiment_metrics (
    experiment_id  TEXT NOT NULL REFERENCES experiment_runs ON DELETE CASCADE,
    model_name     TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    metric         TEXT NOT NULL,
    split          TEXT NOT NULL,             -- 'validation' / 'test' for quality, 'n/a' for cost
    value          DOUBLE PRECISION,          -- NULL unless measured or extrapolated
    status         TEXT NOT NULL CHECK (status IN ('measured', 'extrapolated', 'not_yet_evaluated', 'not_feasible')),
    PRIMARY KEY (experiment_id, model_name, metric, split),
    CHECK ((value IS NULL) = (status IN ('not_yet_evaluated', 'not_feasible')))
);

-- A selected model must point at a real experiment, not just any text.
ALTER TABLE model_versions
    ADD CONSTRAINT model_versions_experiment_fk FOREIGN KEY (experiment_id) REFERENCES experiment_runs (experiment_id);
