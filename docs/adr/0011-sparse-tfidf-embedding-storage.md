# ADR 0011: TF-IDF embeddings are stored as sparse vectors

**Status:** Accepted, 2026-10-08

**Context.** The interim TF-IDF embeddings (ADR 0004) have 4,096 dimensions but only a handful of non-zero terms each. Stored dense, the job sections alone (4,990 jobs × 6 sections ≈ 30k vectors × 16 KB) would take about **480 MB**.

**Decision.**
- Migration 0007 adds `embeddings.embedding_sparse` (pgvector `sparsevec`), makes the dense `embedding` column nullable, and enforces that exactly one of the two is set.
- The TF-IDF provider returns sparse vectors (0-based indices and values; stored in pgvector's 1-based text format); dense models chosen in Phase 6 keep using `embedding`.
- `dimension`, `model_name` and `model_version` are recorded on every row, as before.

**Measured.** All 29,940 job vectors take **6.8 MB**, and the whole `embeddings` table is 15 MB.

**Consequences.**
- Similarity queries have to use the column that matches the model.
- 17 dense TF-IDF student vectors written during Phase 4 testing remain. They are valid (same model version) and are replaced the next time those students are re-processed.
