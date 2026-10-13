# ADR 0017: Core schema before Phase 10: relevance_labels is a view, PII only in users, the agent-run tree

**Status:** Accepted, 2026-10-13 (migration `0009_core_tables_before_phase_10.sql`).

**Context.**
- The spec §23 table list had 24 names. After Phases 2-9, 14 tables existed. These were missing: `users`, `companies`,
  `relevance_labels`, `match_results`, `skill_gaps`, `career_recommendations`, `learning_resources`,
  `learning_recommendations`, `feedback_events`, `agent_messages` and `audit_logs`. Phase 10 needs `learning_resources`.
- Existing tables must not be dropped or recreated, so every change to them is an additive `ALTER`.
- Phase 6 already stores relevance labels: `relevance_pool` (pairs, split) and `relevance_judgments` (one row per
  labeller per pair). The labelling export combines judgments into one label per pair.

**Decision.**
- **`relevance_labels` is a view, not a table.** It reads `relevance_pool` and `relevance_judgments`, so labels are
  stored once. It returns one row per judged `(student_id, job_id)` and aggregates the same way as the labelling
  export: the median of the labellers' grades, ties rounded down. If labellers' sources differ, the view reports the
  weakest one, so a label never looks like stronger evidence than it is. It also has a `student_fully_judged` column,
  which matches the export's filter.
  - The underlying tables already enforced `relevance_label` 0-3, `label_source` in the three allowed values, and
    `split` in train/validation/test.
  - They were missing three rules, which 0009 adds to `relevance_pool`: foreign keys to `student_profiles` and `jobs`,
    and `UNIQUE (student_id, job_id)`. Judgments reference the pool, so these rules cover them too.
- **PII lives only in `users`** (email, phone, password_hash). `password_hash` must be a bcrypt or argon2 (PHC)
  string, so plaintext is rejected.
  - No other table has a PII column, and `audit_logs` does not even link to `users`.
  - The free-form JSON columns (`agent_runs.agent_results` and `errors`, `agent_messages.content`,
    `audit_logs.details`, `feedback_events.payload`) reject anything shaped like an email address. They also reject a
    top-level PII key that holds a real value. The email check uses the same regex as `redact()`, so it only fails
    when scrubbing was skipped. Phone numbers are not pattern-checked in SQL: the phone regex matches dates and IDs
    too often, so for phones the app's `redact()` is the only safeguard.
- **The agent-run tree reuses `agent_runs`.** New columns: `parent_run_id`, `agent_name` (default `supervisor`),
  `retry_count` and `used_fallback`. A child agent row gets its own UUID in `request_id` and points at its Supervisor
  row. Because the new columns have defaults, the Phase 3 writer (`app/db/store.py`) works unchanged.
- **`match_results` requires all seven spec §24 fields** (`PredictionVersions`). `experiment_id` is a foreign key to
  `experiment_runs`, as it is for `model_versions`. The eight EmployabilityScore dimensions and the overall score are
  checked to be 0-100. NULL means Not yet evaluated.
- **`learning_resources` never invents course data.** `url` must start with `http(s)://`, `source_reference` is
  required, and difficulty, prerequisites and estimated effort are nullable (NULL means "not available in source
  data").
  - `learning_recommendations.resource_id` NULL means no resource was found for that skill.
  - That foreign key uses `ON DELETE RESTRICT`, so deleting a course can never quietly turn into a false "none found".
- **Embedding sections are checked per entity type.** Students use the spec list (skills, experience, education,
  projects, certifications, full). Jobs use the sections the Job Intelligence Agent actually stores (adds
  `required_skills`, `responsibilities` and `description`). Phase 10 decides the course sections.
- **`audit_logs` is append-only.** A trigger rejects UPDATE and DELETE.

**Consequences.**
- A student-job pair can appear in only one pool version. A later re-pooling has to leave out pairs that are already
  pooled. Otherwise `load_pool` fails loudly, and it never silently overwrites a split.
- No matching results can be stored until the registered experiment exists in `experiment_runs`. This database
  starts empty, so `python -m app.matching.evaluate` must run again first.
- `jobs.company_id` and `users.company_id` exist but are not backfilled. Company rows are added only from real data.
- `career_recommendations` is a generic shape (role, rank, score, evidence, model version), because the career agent
  has not been built yet. Its phase may extend it with a new migration.
