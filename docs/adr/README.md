# Architecture Decision Records

One file per significant decision (spec §44). The original `docs/ADR.md`, which the phase prompts refer to, was never in this repository (see `PROJECT_AUDIT.md`), so numbering starts here.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-llm-provider-gemini.md) | Runtime LLM provider is Gemini | Accepted (2026-10-08) |
| [0002](0002-no-embedding-model-before-phase-6.md) | No embedding model is used before Phase 6 selects one from evidence | Superseded by 0004 |
| [0003](0003-plain-sql-migrations.md) | Plain SQL migrations with psycopg; database failures never break a request | Accepted (2026-10-08) |
| [0004](0004-interim-tfidf-embeddings.md) | The TF-IDF baseline is the interim embedding model until Phase 6 | Accepted (2026-10-08) |
| [0005](0005-pii-detection.md) | PII detection: auditable regex baseline with measured blind spots | Accepted (2026-10-08) |
| [0006](0006-parser-dotnet-fix.md) | Narrow bug fix in the existing resume parser (`.NET` lost its dot) | Accepted (2026-10-08) |
| [0007](0007-missing-value-sentinels.md) | Missing values are sentinels, never defaults: leave blank, never fabricate | Accepted (2026-10-08) |
| [0008](0008-job-skill-phrases-and-required-preferred.md) | Job skill phrases and the INFERRED required/preferred split | Accepted (2026-10-08) |
| [0009](0009-gender-preference-visibility-filter.md) | Employer gender preference: visibility filter only, never a feature (concern recorded) | Accepted (owner decision) |
| [0010](0010-job-near-duplicate-detection.md) | Near-duplicate job detection at ingest time | Accepted (2026-10-08) |
| [0011](0011-sparse-tfidf-embedding-storage.md) | TF-IDF embeddings stored as sparse vectors | Accepted (2026-10-08) |
| [0012](0012-embedding-model-selection.md) | Embedding model selection: the rule is fixed before the human labels; the primary model is chosen from them | Proposed (2026-10-09): selection pending human labels |
| [0013](0013-matching-eligibility-and-location-tiering.md) | Matching Stage 1: hard eligibility rules (clear failures only) and same-country-first tiering | Accepted (2026-10-09) |
| [0014](0014-matching-pipeline-and-weighted-baseline.md) | Matching pipeline, features-v1, calibrated similarity, and the interim weighted baseline (placeholder weights pending Phase 8) | Accepted as interim (2026-10-09) |
| [0015](0015-matching-model-target-training-and-selection.md) | Matching model: binary suitability target (label >= 2), weak-label training, six candidates, selection rule fixed before the human labels; Baseline 3 stays active until then | Accepted (2026-10-11); model choice pending human labels |
