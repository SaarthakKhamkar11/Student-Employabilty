# ADR 0014: Matching pipeline, features-v1, calibrated similarity, and the interim weighted baseline

**Status:** Accepted as **interim**, 2026-10-09 (Phase 7).
- The pipeline, features and metadata rules are accepted.
- The prediction weights are **placeholders pending Phase 8**, which compares trained matching models (spec §10, §13).

**Context.**
- Spec §5.4, §11, §14, §15 and §24, and the Phase 7 prompt, require:
  - separate tools per stage;
  - exactly the 20 spec §11 features, versioned;
  - an honest interim prediction model (no XGBoost before Phase 8);
  - a multi-dimensional employability score that is never raw cosine × 100;
  - seven version fields on every prediction.
- Phase 6 selected no embedding model ([ADR 0012](0012-embedding-model-selection.md)). The configured model is
  the interim TF-IDF baseline ([ADR 0004](0004-interim-tfidf-embeddings.md)).

**Decision.**

1. **Pipeline.** `app/agents/matching_agent.py` runs one tool per stage (`app/tools/matching_tools.py`, logic in
   `app/matching/`):
   - Stage 1, `filter_eligible`: [ADR 0013](0013-matching-eligibility-and-location-tiering.md).
   - Stage 2, `retrieve_jobs` / `retrieve_candidates`: top `top_n` = 50.
   - `compute_similarity`.
   - Stage 3, `generate_features`.
   - Stage 4, `predict_suitability`.
   - Stage 5, `rank_candidates`.
   - Stage 6, `identify_skill_gaps`.
   - `evaluate_model` and `benchmark_matching` serve the research path (`python -m app.matching.evaluate`).
2. **Retrieval** uses Phase 6's section-weighted protocol over stored section embeddings:
   - skills vs skills (0.35), experience vs responsibilities (0.25), projects vs description (0.15);
   - weights renormalised over the sections both sides have;
   - same-country jobs first (ADR 0013).
   - Education and certifications are matched as structured features instead. The job side has no meaningful text
     for them: `qualification_required` is random in this dataset.
3. **The model is never named in code.** It comes from `EMBEDDING_MODEL` through the provider. Stored vectors are
   compared only with query vectors of the same model name and version, and a mismatch fails loudly.
4. **Calibrated similarity** (`app/matching/calibration.py`).
   - Raw cosine is model-specific (Phase 6), so each similarity feature is used as a **percentile**: where the
     pair falls among the same similarity over every processed student × every canonical job.
   - That reference is 292 × 4,941 pairs for `tfidf-3ec097a5876e`, stored in
     `data/benchmark/matching_calibration/`. Ties use the mid-rank.
   - It is rebuilt after any re-embedding. Without it, the similarity stage fails rather than guess.
   - Example: a raw TF-IDF similarity of 0.023 is the 87th percentile.
5. **Features `features-v1`** (definitions in `app/matching/feature_builder.py`):
   - **Skill scores** give a related ("transferable") skill half credit (`partial_credit` 0.5).
     - `skill_match_score` counts a required skill twice.
     - `matched_skill_ratio` and `preferred_skill_ratio` are exact, with no partial credit.
     - The three counts are kept for Phase 8.
   - **`transferable_skill_score`** is the share of the required skills the profile doesn't name that a related
     student skill covers. Two skills are **related** when the embedding similarity of their names is at least the
     calibrated threshold: the 98th percentile of all distinct skill pairs, with a floor of 0.3. For TF-IDF that
     resolved to 0.3, marking 7,746 of 1.69 million skill pairs as related.
     - With TF-IDF, "related" is lexical, for example "Security analysis" ~ "Security best practices".
     - Words outside the TF-IDF vocabulary are dropped, so "AI Chatbots" and "Generative AI" score 1.0 because
       only "AI" is known. A semantic model fixes this when one is selected.
   - **Structured features:**
     - `education_match`: 1 if the student meets the degree level, 0.5 if one level below, else 0.
     - `degree_match`: same degree, where B.E = B.Tech.
     - `branch_match`: the branch fits the job's role family (`branch_rules`).
     - `experience_match` and `experience_gap`: computed from the longest stated duration.
   - **Similarity features** (percentiles):
     - `semantic_similarity`: retrieval's section-weighted score.
     - `project_relevance`: projects vs the job description.
     - `certification_relevance`: certifications vs the job's skills.
     - `career_interest_alignment`: the student's stated interest vs the job title and role.
     - `role_similarity`: past experience vs the job title and role.
     - `domain_similarity`: branch and domain vs the full job text.
   - **No evidence means `None`**, never a default. Every vector is checked against the feature contract, which
     rejects protected attributes and PII (`tests/test_matching_agent.py`). The student view the features read
     has no gender field at all.
6. **`weighted_baseline_v1`** (`app/matching/baseline.py`, weights in `config/matching_baseline_v1.yaml`, each with
   its reason):
   - suitability = Σ wᵢ · nᵢ / Σ wᵢ over the features present, where nᵢ is the feature normalised to [0, 1].
   - Placeholder weights: skill match 0.20, semantic similarity 0.15, required-skill match 0.10, career-interest
     alignment 0.10, experience match 0.08, education match 0.06, branch match 0.06, project relevance 0.06,
     eight smaller weights (0.02–0.03), and 0 for the counts and `preferred_skill_ratio`.
   - Every output says it is an interim, untrained baseline, and carries every ingredient Phase 11 needs: raw
     similarities, features, normalised values, weights, contributions, weight covered, and missing features.
7. **Employability score (spec §15).**
   - Eight dimensions, each 0–100: skill match, education, experience, project relevance, certification
     relevance, semantic similarity, career-interest alignment, and transferable skills.
   - Each is the weighted mean of its own features. The overall score is suitability × 100.
   - A dimension without evidence reads "Insufficient evidence".
8. **Ranking.** By suitability, with ties broken by id. For jobs, at most `ranking.max_per_role` (1) posting per
   role: every posting of a role has identical text in this data, and without the cap one role filled a whole
   top 5. The other postings are counted, not shown.
9. **Metadata (spec §24).**
   - Every prediction carries all seven fields: `embedding_model`, `embedding_version`, `matching_model`,
     `matching_model_version`, `feature_version`, `dataset_version` and `experiment_id`.
   - They are validated strictly by `PredictionVersions`: a missing, empty or sentinel value fails, and no
     prediction is returned.
   - `matching_model_version` is `weighted_baseline_v1+<config hash>`.
   - `experiment_id` is the recorded experiment that registered that exact config. If the config is edited, the
     hash changes and predictions are refused until it is registered again.

**Evidence** (experiment `match-20261009T155859Z-weakly`, recorded in Postgres and in `results/matching_evaluation/`):

- **Weak labels**, validation split only, 57 students and 21,432 pairs; validity LIMITED. The circularity check
  passes: mean per-student Kendall tau is 0.26 and 0.21, where ≥ 0.9 would mean the labels look model-generated.

  | Scorer | NDCG@5 (95% CI) | MRR | P@5 | R@10 |
  |---|---|---|---|---|
  | `weighted_baseline_v1` | 0.518 (0.435–0.602) | 0.714 | 0.452 | 0.382 |
  | Semantic similarity alone | 0.483 (0.389–0.573) | 0.693 | 0.426 | 0.324 |

  The difference is +0.035 NDCG@5 (95% CI −0.047 to +0.124): **not distinguishable**.
- **Latency** on this laptop, full Stage 1–6 pipeline over 4,992 postings: median 1.0 s per student
  (95th percentile 1.2 s). Per stage: eligibility 254 ms, retrieval 295 ms, similarity 253 ms,
  features 106 ms, prediction 6 ms, skill gaps 83 ms.

**Consequences and caveats.**
- **Leakage.** The weak labels share the role families with `branch_match`, and reflect the student's stated
  domain as `career_interest_alignment` does. **Phase 8 must not select a model on weak labels alone**: it needs
  the human labels (ADR 0012's labelling page).
- **Skill match is low across the board.** 92% of job skill phrases aren't in the alias table (Phase 5), so exact
  matches are rare and the TF-IDF relatedness is lexical. Similarity features carry much of the score. A semantic
  embedding model and a richer skill vocabulary are the levers.
- **Lexical artefacts of TF-IDF are visible in real results.** A data student's list includes "Financial Analyst"
  through the shared word "analyst", and a web student's "UI Developer" scores low on interest alignment because
  it doesn't share the word "web".
- **Phase 8 replaces the placeholder.** It replaces `predict_suitability`'s model by evidence and keeps the
  feature contract (bump `feature_version` if it changes).
