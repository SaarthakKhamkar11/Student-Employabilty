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
| [0012](0012-embedding-model-selection.md) | Embedding model selection: the rule is fixed before the human labels; the primary model is chosen from them | Accepted (2026-10-14): **E5 primary**; TF-IDF baseline and automatic fallback only. Domain-adapted E5 evaluated, not selected |
| [0013](0013-matching-eligibility-and-location-tiering.md) | Matching Stage 1: hard eligibility rules (clear failures only) and same-country-first tiering | Accepted (2026-10-09) |
| [0014](0014-matching-pipeline-and-weighted-baseline.md) | Matching pipeline, features-v1, calibrated similarity, and the interim weighted baseline (placeholder weights pending Phase 8) | Accepted as interim (2026-10-09) |
| [0015](0015-matching-model-target-training-and-selection.md) | Matching model: binary suitability target (label >= 2), weak-label training, six candidates, selection rule fixed before the human labels; Baseline 3 stays active until then | Accepted (2026-10-11); human-label selections 2026-10-14: **LightGBM** on E5 features, Logistic Regression for the TF-IDF fallback |
| [0016](0016-skill-gap-analysis.md) | Skill-gap analysis: shared normaliser, importance by counterfactual re-scoring with the active matching model, relation tiers tuned on real output (TF-IDF: partial 0.40, transferable 0.60), the SkillGapReport contract for Phase 10 | Accepted (2026-10-11) |
| [0017](0017-core-schema-before-phase-10.md) | Core schema before Phase 10: `relevance_labels` is a view over the Phase 6 label tables (constraints added to them), PII only in `users`, the agent-run tree in `agent_runs` | Accepted (2026-10-13) |
| [0018](0018-learning-agent-resources.md) | Learning Agent: the owner-curated Coursera catalogue (fields extracted from each course page, never generated) plus YouTube Data API results for every skill, ranked by relevance under the configured embedding model, with a threshold tuned per model; "No resources found" instead of fillers | Accepted (2026-10-14) |
| [0019](0019-explainability-agent.md) | Explainability Agent: SHAP chosen per model (TreeSHAP for LightGBM, LinearExplainer for the LR fallback), global and per-match explanations as separate paths, LLM prose only as lines citing deterministic evidence items, validated (one strict retry) or replaced by the structured evidence | Accepted (2026-10-15) |
| [0020](0020-agent-tracing-and-cohort-analytics.md) | Agent tracing from real runtime spans (agent executions as `agent_runs` rows, tool attempts / LLM calls / skips as `agent_messages`), the tree rebuilt only from those rows, request-scoped PII redaction fed by the Phase 4 detector; cohort benchmarking with aggregates only (branch / graduation year / target-role filters, top-20 by profile embedding, at least 5 peers) | Accepted (2026-10-15) |
| [0021](0021-evaluation-framework.md) | Evaluation framework: the five baselines on the held-out test split (pre-registered NDCG@5 + paired-CI verdict rule, reusing each phase's own scorer), a leakage report counted on the real data (incl. single-feature AUC vs weak and human labels), RQ9 from persisted traces, performance on the production code in throwaway databases at 286×4,990 real, 1,000×1,000 and 5,000×1,000 (partially synthetic students, seeded, never persisted) | Accepted (2026-10-15) |
