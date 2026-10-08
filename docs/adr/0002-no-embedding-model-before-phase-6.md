# ADR 0002: No embedding model is used before Phase 6 selects one from evidence

**Status:** Superseded by [ADR 0004](0004-interim-tfidf-embeddings.md) on 2026-10-08 (the owner's Phase 4 instruction)

**Context.** The spec's Profile and Job Intelligence agents (Phases 4 and 5) "generate embeddings using the currently selected embedding model". But no model is selected until Phase 6's Model Selection Engine compares SBERT, E5, BGE, a small model and the baselines on one labelled benchmark (spec §6–§9). The rules forbid choosing a model without evidence (rules 2, 8, 20). The only existing measurements (`results/embedding_model_comparison.json`) are cost-only, and their score means cannot be compared across models (`PROJECT_AUDIT.md` I-3).

**Options considered.**
1. Use a provisional model (for example SBERT) in Phases 4–5 and switch later. Rejected: it is the "blindly choose SBERT" pattern, and provisional vectors would leak into stored data and later comparisons.
2. **Defer embedding generation until a model is selected.** Chosen.

**Decision.**
- In Phases 4–5 the agents build and store *structured* profiles and jobs.
- Their embedding step looks up the selected embedding model in `model_versions` (status `selected`). When there is none, it reports `embedding_model: "Not yet evaluated"` and stores no vector.
- Phase 6 inserts the evidence-based selection, which the schema requires to carry an `experiment_id`. A backfill then embeds every stored profile and job with that model and records model name, version and dimension per vector (`embeddings` table).

**Consequences.**
- Matching (Phase 7) comes after Phase 6, so nothing that needs vectors runs without them.
- Until Phase 6, profile and job responses honestly show "Not yet evaluated" for embeddings.
- Phase 6 needs relevance labels (spec §8). Those remain the critical-path dependency.
