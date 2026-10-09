# ADR 0012: Embedding model selection: the rule is fixed before the human labels; the primary model is chosen from them

**Status:** Proposed, 2026-10-09. It becomes Accepted when the engine has run on the team's human labels and the
result is recorded here with its `experiment_id`. **Until then no embedding model is selected.** The deployed model
stays the interim TF-IDF baseline ([ADR 0004](0004-interim-tfidf-embeddings.md)), and every `model_versions` row
stays `candidate`.

**Context.**
- Spec §6 and §9 require the primary embedding model to be chosen from evidence. The choice must use transparent,
  configurable weights and a sensitivity analysis, and TF-IDF must appear in every table as the floor. Spec §8 forbids
  circular evaluation. The Phase 6 prompt makes "never fabricate a metric" and an honest `label_source` hard rules.
- Evidence on 2026-10-09. The reports are `emb-20261009T074529Z-nolabels` and `emb-20261009T074653Z-weakly` in
  `results/embedding_evaluation/`, also recorded in Postgres and served at `GET /experiments`:
  - **Cost:** measured for all five models. Each ran in its own process with network access off, using 30 warm
    trials per latency figure (see the table below).
  - **Human labels:** collection is under way on the labelling page. The pool is `pool-5c6a5972b32d`:
    30 validation/test students × 12 jobs = 360 pairs. When this ADR was written, 0 pairs had been graded.
  - **Weak labels:** rule-based, from the student's stated domain × the job's role family
    (`config/weak_label_rules.yaml`). Evaluation validity is LIMITED, and they cover 57 validation students.

**Decision.**

1. **Models compared (spec §6).**
   - A. SBERT `all-mpnet-base-v2`.
   - B. E5 `e5-base-v2`.
   - C. BGE `bge-base-en-v1.5`.
   - D. `all-MiniLM-L6-v2`, the small model. It is the most widely used small sentence-embedding model and comes from
     the same library and training recipe as A, so A vs D isolates the effect of model size (6 layers, 384
     dimensions, 87 MB).
   - E. Custom/domain model: **"Not feasible — insufficient labeled data"**. There are 286 students and no labelled
     training pairs; the human benchmark is evaluation-only.
   - F. Hybrid: not an embedding model. Phase 7 builds it and Phase 14 evaluates it.
   - Floor: the TF-IDF baseline as deployed (`tfidf-3ec097a5876e`).
2. **One benchmark for every model.**
   - **Text view v2:**
     - Audit I-3 is fixed: empty sections are never embedded, and section weights are renormalised.
     - Audit I-4 is fixed: each job field is used once.
     - `experience_required` and `qualification_required` are left out of the job text. They are randomly
       assigned per posting (one role carries a median of 7 degrees and 10 experience ranges).
   - **Jobs:** the 376 role representatives.
   - **Splits:** fixed per student by SHA-256: train 180, validation 57, test 49 students.
   - **What is evaluated:** selection uses the validation split only. Train is never evaluated, and test is held
     back for Phase 14.
   - **Metrics:** computed on judged jobs only (condensed lists). The primary metric is NDCG@5, and "relevant"
     means a grade of 2 or more.
3. **The selection rule.** It was fixed now, before any human label exists, so the weights can't be tuned towards a
   preferred result. It is implemented and tested as `recommend()` in `src/embeddings/model_evaluation_engine.py`.
   - Run the engine on the `human_reviewed` labels with `config/model_selection_weights.yaml`
     (`weights-v1`, sha256 `24b4fadb2770…`).
   - **Selection requires:** evaluation validity OK, meaning human labels whose independence from every evaluated
     model is confirmed by the manifest and by the score-correlation check.
   - **Eligibility:** a candidate is eligible only if its NDCG@5 beats the TF-IDF floor's.
   - **The winner:** the eligible candidate with the highest Overall Score under the Balanced weights. It is
     selected unless the floor scores higher overall *and* the winner's quality gain over the floor is not
     statistically distinguishable (paired bootstrap 95% CI). In that case the decision is **keep TF-IDF**, which is
     a valid, evidence-based outcome.
   - The sensitivity analysis is reported with the decision: 3 weight scenarios, 1,000 random weightings and the
     alternative scoring protocol. If the winner is not stable, this ADR says so.
4. **Primary model today: none (Insufficient evidence).** No human labels exist yet. Weak labels can never select a
   model.

**Evidence so far.**

Cost (CPU, this M2 laptop with Docker running; absolute numbers are machine-specific):

| Model | CPU latency per pair (median) | Throughput, CPU / MPS (texts/s) | Peak memory | Size | Dimension | Runtime dependencies |
|---|---|---|---|---|---|---|
| TF-IDF (floor) | 1.9 ms | 3,817 / n/a | 97 MB | 0.1 MB | 4,096 sparse (~66 non-zero) | 6 packages, 121 MB |
| A. SBERT mpnet | 686 ms | 3.9 / 49.5 | 741 MB | 418 MB | 768 | 31 packages, 556 MB |
| B. E5 | 654 ms | 4.0 / 61.9 | 635 MB | 418 MB | 768 | 31 packages, 556 MB |
| C. BGE | 643 ms | 4.3 / 61.9 | 654 MB | 418 MB | 768 | 31 packages, 556 MB |
| D. MiniLM | 125 ms | 25.0 / 249 | 210 MB | 87 MB | 384 | 31 packages, 556 MB |

Quality on **weak labels** (labelled "WEAK LABELS — RULE-BASED, NOT HUMAN GROUND TRUTH"; 54 validation students
evaluated):

| Model | NDCG@5 | MRR | vs TF-IDF floor, NDCG@5 (95% CI) |
|---|---|---|---|
| C. BGE | 0.633 | 0.823 | +0.053 (−0.011 to +0.119), not distinguishable |
| B. E5 | 0.603 | 0.834 | +0.023 (−0.046 to +0.090), not distinguishable |
| TF-IDF | 0.580 | 0.781 | — |
| D. MiniLM | 0.577 | 0.769 | −0.003 (−0.083 to +0.076), not distinguishable |
| A. SBERT | 0.524 | 0.737 | −0.056 (−0.136 to +0.021), not distinguishable |

- On weak labels, BGE leads the candidates in all three weight scenarios and in 100% of random weightings among
  eligible models.
- The leads are fragile in three ways:
  - BGE is not distinguishable from E5 (+0.030 NDCG@5).
  - MiniLM is excluded from eligibility by 0.003 NDCG@5, a difference that is not distinguishable.
  - **The decision changes with the scoring protocol.** Under the section-weighted protocol (skills, experience
    and projects scored separately), MiniLM beats the floor (NDCG@5 0.592 vs 0.525). It then becomes eligible
    and wins the Balanced decision on cost. The quality order there is E5 0.628 ≈ BGE 0.627 > MiniLM 0.592.
- The TF-IDF floor's Balanced Overall Score (0.755) is higher than BGE's (0.567).
- Applied to human labels, this pattern would mean **keep TF-IDF**.
- A transformer earns its cost only if the human labels show a real quality gain.

**Consequences.**
- Phase 7 does not wait. It reads embeddings through the provider interface (`app/embeddings/provider.py`) and must
  not hard-code a model.
- Finishing this ADR takes the following steps:
  1. The team finishes grading.
  2. Export the labels: `scripts/export_labels.py`.
  3. Run `python -m src.embeddings.model_evaluation_engine --labels data/benchmark/human_labels.csv --record-db`.
  4. Copy the Decision line and the sensitivity result into this ADR, and set the status to Accepted.
  5. If the decision is "select":
     - Insert the model into `model_versions` as `selected` with that `experiment_id`. The schema enforces both,
       and migration 0008 adds the foreign key.
     - Add its provider and set `EMBEDDING_MODEL`.
     - Re-embed the 286 students and 4,990 jobs.
- Selecting a base-size transformer changes the deployment:
  - It adds about 556 MB of runtime dependencies to the backend image.
  - It raises CPU latency from about 2 ms to about 650 ms per pair.
  - It adds about 650 to 740 MB of memory.
  - Re-embedding the 5,276 full texts takes about 20 minutes on CPU at the measured 4.3 texts/s, or about
    1.5 minutes on the M2 GPU. The backend also stores per-section vectors (up to 6 per student or job), so a
    complete backfill takes several times that.
  - Dense 768-dimensional vectors take about 3 KB each.
- The weak labels remain a sanity benchmark and possible training signal for Phase 8. They are never reported as
  ground truth.
