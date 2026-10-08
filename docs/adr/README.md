# Architecture Decision Records

One file per significant decision (spec §44). The original `docs/ADR.md`, which the phase prompts refer to, was never in this repository (see `PROJECT_AUDIT.md`), so numbering starts here.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-llm-provider-gemini.md) | Runtime LLM provider is Gemini | Accepted (2026-10-08) |
| [0002](0002-no-embedding-model-before-phase-6.md) | No embedding model is used before Phase 6 selects one from evidence | Accepted (2026-10-08) |
| [0003](0003-plain-sql-migrations.md) | Plain SQL migrations with psycopg; database failures never break a request | Accepted (2026-10-08) |
