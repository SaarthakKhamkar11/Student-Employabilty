# ADR 0004: The TF-IDF baseline is the interim embedding model until Phase 6

**Status:** Accepted, 2026-10-08. Supersedes [ADR 0002](0002-no-embedding-model-before-phase-6.md).

**Context.** ADR 0002 deferred all embeddings until Phase 6. The owner's Phase 4 instruction is different: the Profile Agent must embed profiles with "the currently selected embedding model", which until Phase 6 is the TF-IDF baseline, and must not wire in SBERT early. TF-IDF is the spec's Baseline 2 (§34), not a neural-model choice, so using it does not pre-judge RQ1.

**Decision.**
- `src/embeddings/tfidf_baseline.py` was **re-implemented**: the original file the prompts describe never existed in this repository.
  - It is fitted on the processed corpus (286 students and 4,990 jobs, 5,276 documents), so students and jobs share one vector space.
  - Parameters: word 1–2 grams, `min_df=2` (drops one-off tokens such as names), 4,096 features, sublinear TF, L2 norm.
- The model version is a content hash of the vocabulary, the IDF weights and the parameters (currently `tfidf-3ec097a5876e`). The same data gives the same version on any machine; the host and the Docker container match.
- The model is chosen by configuration (`EMBEDDING_MODEL=tfidf_baseline`), never inside an agent. Phase 6 adds providers and changes the setting; agents and tools stay as they are.
- Every stored vector records `model_name` and `model_version`. `model_versions` lists the baseline as `candidate`, labelled "interim baseline, not selected by experiment". It can't be `selected`: the schema requires an `experiment_id` for that.
- Sections holding a sentinel are never embedded, and text with no in-vocabulary terms (a zero vector) is never stored. This prevents the empty-text artefact behind audit problem I-3.

**Consequences.**
- Profiles get embeddings now: per section plus `full`, 4,096 dimensions each.
- When Phase 6 selects a model, every stored profile and job must be re-embedded (a backfill). Old TF-IDF vectors stay distinguishable by name and version.
- A TF-IDF vector only captures lexical overlap with the corpus vocabulary. It is not a measure of semantic quality, and no quality claim is made for it.
