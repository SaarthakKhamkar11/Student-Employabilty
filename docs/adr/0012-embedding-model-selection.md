# ADR 0012: Embedding model selection: the rule is fixed before the human labels; the primary model is chosen from them

**Status:** Accepted, 2026-10-14. **E5 `e5-base-v2` is the primary embedding model**, selected by the rule below
(`emb-20261014T100323Z-human`, confirmed by `emb-20261014T145548Z-human`). **TF-IDF is the baseline and the
fallback only** (owner decision). Proposed 2026-10-09.

**Owner decision, 2026-10-14: E5 primary, TF-IDF strictly baseline and fallback.**
- **Why:** matching needs semantic similarity. TF-IDF compares shared words only. It has the highest Balanced
  Overall Score (0.710) only because it is ~300× cheaper, and the rule never treated it as a candidate.
  - The same day TF-IDF was briefly deployed on a "highest Overall Score" instruction; this decision reverses that.
- **TF-IDF stays in two roles:**
  - **the floor** every model must beat (unchanged);
  - **the automatic fallback.** If E5 can't be loaded (weights or runtime missing),
    `build_embedding_provider(verify=True)` returns TF-IDF:
    - `/health` reports `embedding_fallback`, and every response's model versions say TF-IDF;
    - TF-IDF vectors are stored next to E5's for every profile and job (`fallback_provider`, `generate_embedding`);
    - TF-IDF has its own calibration, skill-gap tiers, learning threshold (0.15), and matching model
      (`config/matching_model.yaml` `fallback.tfidf_baseline`: Logistic Regression, selected on TF-IDF features).
  - **Verified live:** with E5's weights unreachable, a request for S00001 ran on TF-IDF + Logistic Regression and
    got learning resources for all 6 gaps.

**Improving E5 (2026-10-14).** The owner asked for E5's Overall Score to be raised by improving the model, with the
fixed weights unchanged.
- **Cost levers, measured in the backend image (x86 CPU) on 60 students × 120 jobs:**

  | Variant | Latency per pair | Size | Ranking agreement with E5 (median τ / top-5 overlap) |
  |---|---|---|---|
  | E5 as is | 233 ms | 418 MB | 1.00 / 100% |
  | int8 dynamic quantisation | 161 ms | 173 MB | 0.65 / 78% |
  | int8, per channel | 158 ms | 174 MB | 0.72 / 85% |
  | int8, feed-forward layers only | 190 ms | 256 MB | 0.75 / 88% |
  | max 256 or 384 tokens | no faster | 418 MB | ~1.00 / 100% |

  - **Quantisation was rejected.** E5's cosines sit in a narrow band (0.68–0.75), so small numeric errors reorder
    jobs: it would give up the quality that is E5's reason to exist.
  - **Shorter input limits change nothing:** texts are short (median 128 tokens, longest 415).
  - **Cost can't move the score much anyway:** cost is compared on log10 against TF-IDF's ~2 ms, so even a 3× speedup
    adds about +0.03.
- **Quality lever: domain adaptation (slot E).**
  - `src/embeddings/domain_adapt.py` fine-tunes E5 on the train split's weak labels.
    - 1,830 strong pairs from 173 students, each with one hard negative.
    - Contrastive loss; one fixed recipe, set before evaluation.
    - 1 epoch, 71 min on CPU.
  - The human-labelled students are never seen in training, and the test split stays held back.
  - Evaluated with `PHASE6_DOMAIN_MODEL=models/embeddings/e5-base-v2-domain-81e7ec962511`. Its cost is E5's measured
    cost, since the network is identical; complexity gets +1 point for in-house retraining.

  | Model | MRR | NDCG@5 (95% CI) | Recall@5 | Precision@5 | Section-weighted NDCG@5 | Overall (Balanced) |
  |---|---|---|---|---|---|---|
  | B. E5 | **0.910** | **0.756** (0.646–0.856) | 0.691 | 0.453 | 0.679 | **0.512** |
  | E. E5, domain-adapted | 0.765 | 0.745 (0.627–0.844) | **0.741** | **0.507** | **0.747** | 0.466 |

  - **Gains:** domain adaptation raised recall, precision, agreement with the graders (Kendall τ 0.48 → 0.57) and the
    section-weighted score.
  - **Loss:** it lowered MRR. The first good job appears later.
  - **Net:** it is not distinguishable from E5 on NDCG@5 (+0.011, CI −0.085 to +0.112), and **the rule keeps plain E5**
    (wins 75.6% of random weightings; the adapted model 15.3%).
  - E5's score reads 0.512 here instead of 0.532 only because a sixth model entered the min-max normalisation.
  - **Not retried with other settings:** tuning the recipe until it wins on these 180 validation pairs would overfit
    them. A fair next attempt needs new training signal (more human-labelled train pairs), not another pass over
    the same validation labels.
  - The adapted weights are kept locally (`models/embeddings/`, gitignored and dockerignored), not deployed.
- **Platform caveat (fallback):** the TF-IDF version this Windows/x86 setup fits is `tfidf-953a75d166d8`; a
  Mac/arm64 fit gives `tfidf-3ec097a5876e` (ADR 0015). The fallback's matching model was trained on the former.

**Human-label result (2026-10-14).**
- **Labels:** 360 `human_reviewed` pairs (30 students, pool `pool-5c6a5972b32d`).
  - **One labeller**, so agreement couldn't be measured.
  - Evaluated on the validation split: 180 pairs, 15 students.
  - Grades: 0 × 145, 1 × 114, 2 × 75, 3 × 26.
  - Evaluation validity **OK**. Mean per-student Kendall tau vs the labels is 0.31 to 0.48, far from the 0.9
    leakage line.
- **Full-text protocol** (the rule's primary protocol), NDCG@5 (95% CI):

  | Model | NDCG@5 (95% CI) | MRR |
  |---|---|---|
  | TF-IDF (floor) | 0.602 (0.466–0.725) | 0.719 |
  | SBERT | 0.588 | 0.689 |
  | **E5** | **0.756 (0.646–0.856)** | **0.910** |
  | BGE | 0.697 | 0.764 |
  | MiniLM | 0.619 | 0.601 |

  - E5 vs the floor: +0.154, 95% CI **+0.005 to +0.314**, distinguishable, but only just.
  - E5 vs BGE: +0.059, 95% CI −0.008 to +0.134, not distinguishable.
- **Decision line:** "Select B. E5 e5-base-v2 as the primary embedding model (Balanced winner, Overall Score 0.532)."
  - The floor scores higher overall (0.710), because it is far cheaper. But E5's quality gain is distinguishable,
    so the fixed rule selects E5.
  - E5 wins 2 of 3 weight scenarios and 87.2% of random weightings (MiniLM wins 12.8%).
- **⚠ Protocol caveat (recorded when activated).** The Matching Agent retrieves with the **section-weighted**
  protocol (ADR 0014). Under it, the report's robustness check gives NDCG@5:
  - BGE 0.748, MiniLM 0.725, SBERT 0.724, **TF-IDF 0.683, E5 0.679**;
  - E5 sits below the floor there, and the Balanced winner would be MiniLM.

  The rule was fixed on the full-text protocol, so it is applied as written, and the owner chose to activate E5
  knowing this. Phase 14 must compare the deployed (section-weighted) retrieval with E5 against TF-IDF on the test
  split, and a future selection should pre-register the protocol the system actually uses.
- **Consequences carried out on 2026-10-14:**
  - `model_versions` row `selected`;
  - `E5Provider` (`app/embeddings/provider.py`): "query: " for student texts and skill names, "passage: " for job
    and course texts, as in the benchmark;
  - `EMBEDDING_MODEL` defaults to `e5_base_v2`;
  - the backend image carries CPU torch and the pinned weights;
  - re-embedding, recalibration, a rebuilt feature matrix, and Phase 8 retrained and selected on it (ADR 0015).

**Original status (2026-10-09):** Proposed. It was to become Accepted when the engine ran on the team's human
labels. Until then no embedding model was selected, the interim TF-IDF baseline stayed deployed
([ADR 0004](0004-interim-tfidf-embeddings.md)), and every `model_versions` row stayed `candidate`.

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
