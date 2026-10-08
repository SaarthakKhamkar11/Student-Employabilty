# ADR 0003: Plain SQL migrations with psycopg; database failures never break a request

**Status:** Accepted, 2026-10-08

**Context.** Phase 3 needed persistent workflow state (`agent_runs`), and Phase 4 needs the student tables from spec §23. The prompts allowed "alembic or plain SQL".

**Decision.**
- Schema changes are numbered SQL files in `backend/migrations/`, applied once each, in order, by `python -m app.db.migrate`. Applied files are tracked in `schema_migrations`.
- The backend container runs the migrations on start.
- Data access uses psycopg 3 directly; there is no ORM.
- `student_profiles` mirrors `config/schema.py` STUDENT_SCHEMA 1:1. The processed CSV loads with `python -m app.db.seed`, which validates every row through `StudentRecord`.
- Persistence failures are reported (`persisted: false`, error `persistence_failed`) and never crash a request (spec §21).

**Consequences.**
- The SQL is reviewable and has no ORM layer to learn.
- Migrations are forward-only: there are no down-migrations, so a mistake is fixed by a new numbered file.
- Records are PII-scrubbed before storage: emails and phone numbers are masked.
