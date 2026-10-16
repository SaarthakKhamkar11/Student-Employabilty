# Model evaluation: every comparison, and why each model was chosen

This is the single summary of the project's model experiments. Every number below is copied from a generated
report in `results/` or from the decision record (ADR) that cites it. Nothing is estimated here.

| Decision | Chosen | Runner-up | Decided by | Source |
|---|---|---|---|---|
| Embedding model | **E5 (`intfloat/e5-base-v2`)** | E5 domain-adapted / BGE | Human labels, rule fixed beforehand | `results/embedding_model_evaluation_human_reviewed.md`, [ADR 0012](adr/0012-embedding-model-selection.md) |
| Embedding fallback | **TF-IDF** (automatic, if E5 can't load) | — | Owner decision | [ADR 0012](adr/0012-embedding-model-selection.md) |
| Matching model (on E5) | **Logistic Regression** (features-v2, re-selected 2026-10-15) | LightGBM | Human labels, rule fixed beforehand | `results/matching_model_evaluation/mms-20261015T124013Z-human.md`, [ADR 0015](adr/0015-matching-model-target-training-and-selection.md) |
| Matching model (on the TF-IDF fallback) | **Baseline 3** (features-v2: the rule kept it) | Logistic Regression | Human labels | `results/matching_model_evaluation/mms-20261015T124901Z-human.md` |
| Explainer | **LinearExplainer** (Logistic Regression); exact TreeSHAP for tree models | — | Model type | [ADR 0019](adr/0019-explainability-agent.md) |

> **Updated 2026-10-15 (Phase 14).** Section 3 below records the features-v1 runs of 2026-10-14, when LightGBM was
> selected. After the features-v2 change (Phase 13 review) both paths were retrained and re-selected, giving the
> rows above. The held-out test-split comparison of all five spec §34 baselines, and the answers to RQ1–RQ9, are in
> [`docs/RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md).

---

## 1. How the models were judged

- **Human relevance labels.** The team graded 360 student × job pairs on a 0–3 rubric (rubric-v1). Grades ≥ 2 count as
  relevant/suitable. The pairs come from 30 students, 15 for validation and 15 for test, assigned per student and fixed
  in `data/benchmark/student_splits.csv`.
  - **Validation (180 pairs) is the only split used for selection.** Test (180 pairs) is held back, untouched, for the
    final unbiased estimate (Phase 14).
  - There was one labeller, so inter-annotator agreement could not be measured.
  - The pairs were pooled TREC-style: the five embedding models chose which jobs were judged.
- **The selection rules were written down before any human label existed:** the weights, the gate, three weighting
  scenarios and a sensitivity analysis. They live in `config/model_selection_weights.yaml` and
  `config/matching_model_selection_weights.yaml`, so no weight could be tuned toward a preferred answer.
- **Confidence intervals** use a paired bootstrap over students (2,000 resamples). With 15 validation students many
  differences are **not statistically distinguishable**, and this document says so wherever that is the case.
- **Circularity check.** For each model, the mean per-student Kendall τ between its scores and the human labels is
  0.31–0.57. A τ of 0.9 or more would mean the labels look model-generated.

---

## 2. Embedding models (Phase 6)

Experiment `emb-20261014T145548Z-human` · validation split, 180 pairs, 15 students · grades {0: 145, 1: 114, 2: 75, 3: 26}
across all 360 pairs.

### Quality (K = 5; NDCG@5 with 95% CI)

| Model | MRR | NDCG@5 (95% CI) | Recall@5 | Precision@5 | NDCG@10 | Kendall τ | vs TF-IDF floor, NDCG@5 |
|---|---|---|---|---|---|---|---|
| TF-IDF baseline (floor) | 0.719 | 0.602 (0.466–0.725) | 0.766 | 0.440 | 0.749 | 0.311 | — |
| A. SBERT all-mpnet-base-v2 | 0.689 | 0.588 (0.488–0.674) | 0.486 | 0.360 | 0.759 | 0.359 | −0.014, not distinguishable |
| **B. E5 e5-base-v2** | **0.910** | **0.756 (0.646–0.856)** | 0.691 | 0.453 | **0.858** | 0.481 | **+0.154 (+0.005 to +0.314)** |
| C. BGE bge-base-en-v1.5 | 0.764 | 0.697 (0.572–0.816) | 0.683 | 0.427 | 0.803 | 0.471 | +0.095, not distinguishable |
| D. all-MiniLM-L6-v2 | 0.601 | 0.619 (0.506–0.722) | 0.653 | 0.440 | 0.763 | 0.363 | +0.017, not distinguishable |
| E. E5, domain-adapted (in house) | 0.765 | 0.745 (0.627–0.844) | **0.741** | **0.507** | 0.831 | **0.566** | +0.144 (+0.022 to +0.273) |

### Cost (CPU, one student–job pair, median of 30 warm runs, each model in its own offline process)

| Model | Latency | Peak memory | Size | Dimension | Throughput (texts/s, CPU) |
|---|---|---|---|---|---|
| TF-IDF | 1.9 ms | 97 MB | 0.1 MB | 4096 sparse (~66 non-zero) | 3,817 |
| SBERT | 685.9 ms | 741 MB | 417.7 MB | 768 | 3.9 |
| **E5** | 654.1 ms | 635 MB | 417.6 MB | 768 | 4.0 |
| BGE | 642.9 ms | 654 MB | 417.6 MB | 768 | 4.3 |
| MiniLM | 124.5 ms | 210 MB | 86.6 MB | 384 | 25.0 |

These were measured on the developer's laptop (arm64). Only the relative order transfers to other machines. In the
backend image (x86) E5 takes 233 ms per pair.

### Overall Score (multi-criteria; gate: an embedding model must beat the TF-IDF floor on NDCG@5)

| Scenario | Weights (quality / cost) | Winner among eligible models |
|---|---|---|
| **Balanced (the decision scenario)** | 0.55 / 0.45 | **E5: 0.512** (E5-domain 0.466, BGE 0.366, MiniLM 0.348; SBERT below the floor) |
| Quality prioritised | 0.80 / 0.20 | E5: 0.718 |
| Low-resource deployment | 0.35 / 0.65 | MiniLM: 0.368 (E5 0.352) |

- **Sensitivity:** across 1,000 random weightings centred on Balanced, E5 wins 75.6%, E5-domain 15.3%, MiniLM 9.1% and
  BGE 0%.
- The TF-IDF floor scores 0.678 overall, because its cost advantage outweighs the quality gap. It is a floor, not a
  candidate.
- **Protocol caveat:** under the section-weighted protocol (skills, experience and projects scored separately) the
  quality order changes (BGE ahead), and MiniLM would win Balanced. The scoring protocol matters.

### Why E5

1. It is the **Balanced winner (0.512)** under the rule fixed before the labels. It also wins quality-prioritised and
   75.6% of random weightings.
2. It is the **only off-the-shelf model whose gain over the TF-IDF floor is statistically distinguishable** (+0.154
   NDCG@5, CI +0.005 to +0.314). It also has the best MRR (0.910) and NDCG@10 (0.858).
3. It runs fully offline on CPU (pinned revision `f52bf8ec8c71`) and fits the backend container.

### What was tried to improve E5, and rejected

| Attempt | Result | Decision |
|---|---|---|
| int8 dynamic quantisation | 233 → 161 ms, 418 → 173 MB, but only a 78% top-5 overlap with E5's rankings (τ 0.65) | **Rejected:** E5's cosines sit in a narrow band (0.68–0.75), so small numeric errors reorder jobs |
| int8 per channel / feed-forward only | 85% / 88% top-5 overlap | Rejected for the same reason |
| Shorter max length (256/384 tokens) | No faster (median text 128 tokens) | No effect |
| Domain adaptation (E5 fine-tuned on the train split's weak labels) | Recall@5 0.741, P@5 0.507, τ 0.566 all improve, but MRR falls 0.910 → 0.765; Overall 0.466 vs 0.512 | **Not selected.** Not retried, to avoid fitting the validation labels |

**Fallback:** if E5 can't load, the backend runs on TF-IDF automatically and says so in `/health` and in every
response's model versions. TF-IDF vectors are stored beside E5's for every profile and job.

---

## 3. Matching models (Phase 8)

- **Target:** binary suitability (human grade ≥ 2).
- **Training data:** 67,680 train-split pairs labelled by rules ("weak labels", 1,830 suitable), with class-balanced
  weights, seed 42 and one thread. All six candidates use the same 20 features (features-v1: similarity percentiles,
  skill-match counts, branch/education/experience fit). A guard rejects protected attributes; gender is never a
  feature (ADR 0009).
- **Evaluation:** human labels, validation split: 180 pairs, 15 students, 50 suitable.
- **Gate:** a trained model must beat Baseline 3 (the hand-weighted interim score) on NDCG@10.

### On E5 features (the primary path): experiment `mms-20261014T151058Z-human`

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | MRR | NDCG@5 | NDCG@10 (95% CI) | vs Baseline 3, NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.572 | 0.390 | 0.960 | 0.555 | 0.754 | 0.558 | 0.743 | 0.743 | 0.817 (0.737–0.879) | +0.039, n.d. |
| Decision Tree | 0.578 | 0.392 | 0.940 | 0.553 | 0.773 | 0.494 | 0.710 | 0.681 | 0.783 (0.706–0.855) | +0.005, n.d. |
| Random Forest | 0.644 | 0.434 | 0.920 | 0.590 | 0.845 | 0.647 | **0.819** | 0.755 | 0.838 (0.752–0.906) | +0.060, n.d. |
| Gradient Boosting | 0.694 | 0.473 | 0.860 | 0.610 | 0.822 | 0.624 | 0.714 | 0.718 | 0.815 (0.731–0.878) | +0.037, n.d. |
| XGBoost | 0.711 | 0.488 | 0.820 | 0.612 | 0.839 | **0.661** | 0.712 | 0.722 | 0.817 (0.733–0.883) | +0.039, n.d. |
| **LightGBM** | **0.739** | **0.519** | 0.800 | **0.630** | **0.847** | 0.631 | 0.739 | **0.774** | **0.844 (0.762–0.908)** | **+0.066 (−0.015 to +0.156), n.d.** |
| Baseline 3 (weighted sum) | 0.656 | 0.440 | 0.880 | 0.587 | 0.773 | 0.509 | 0.654 | 0.643 | 0.778 (0.703–0.850) | — |

n.d. = not statistically distinguishable at 15 students.

| Model | Latency (50 jobs) | Training time | Memory added | Size on disk | Complexity (points) | Explainability (/5) | Overall (Balanced) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 1.77 ms | 4.10 s | 60.8 MB | 2.3 KB | 4 | 5 | 0.518 |
| Decision Tree | 1.98 ms | 0.27 s | 44.2 MB | 3.4 KB | 2 | 5 | 0.428 |
| Random Forest | 10.78 ms | 7.88 s | 45.3 MB | 2,715.5 KB | 3 | 2 | 0.460 |
| Gradient Boosting | 1.69 ms | 19.64 s | 43.9 MB | 91.4 KB | 3 | 3 | 0.474 |
| XGBoost | 6.59 ms | 1.23 s | 8.1 MB | 327.7 KB | 5 | 2 | 0.353 |
| **LightGBM** | **0.95 ms** | 1.35 s | 22.9 MB | 345.9 KB | 6 | 3 | **0.615** |
| Baseline 3 | 2.68 ms | — | — | 7.4 KB | 0 | 5 | 0.523 (baseline) |

- **Scenarios:** LightGBM wins Balanced (0.615), quality-prioritised (0.714) and, among the trained models,
  low-resource (0.535). The winner does not change across scenarios.
- **Sensitivity:** over 1,000 random weightings, LightGBM wins 76.2%, Logistic Regression 17.2%, Random Forest 4.1%,
  Decision Tree 2.3%, Gradient Boosting 0.2% and XGBoost 0%.
- **Simpler-model check:** LightGBM vs Logistic Regression is +0.028 NDCG@10 (CI −0.015 to +0.079, not
  distinguishable) and +0.075 F1, at half the latency, but 2 more complexity points and 2 fewer explainability points.
  The engine flags this as "close enough".

### Why LightGBM

1. It is the **Balanced winner (0.615)** under the rule fixed before the labels (ADR 0015). It wins all three
   scenarios and 76% of random weightings.
2. It is best or joint-best on the ranking and classification metrics that matter here: NDCG@10 0.844, NDCG@5 0.774,
   F1 0.630, accuracy 0.739, ROC-AUC 0.847. It is also the fastest trained model (0.95 ms for 50 jobs).
3. Its explanations are exact: TreeSHAP contributions add up to the model's output (max error 2.1e-14 on 200 pairs;
   5.2e-14 over all 21,432 validation pairs in Phase 11).
4. **The honest caveat.** Its lead over Baseline 3 (+0.066) and over Logistic Regression (+0.028) is not
   statistically distinguishable with 15 validation students. The choice follows the pre-registered rule; the held-out
   test split (Phase 14) and more labels are what will confirm it or not.

### On TF-IDF features (the fallback path): experiment `mms-20261014T132323Z-human`

| Model | F1 | NDCG@10 | MRR | Latency | Explainability | Overall (Balanced) |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.606 | 0.813 | **0.776** | 1.28 ms | 5/5 | **0.699** |
| LightGBM | **0.683** | **0.821** | 0.758 | 0.45 ms | 3/5 | 0.655 |
| Decision Tree | 0.621 | 0.792 | 0.747 | 1.27 ms | 5/5 | 0.627 |
| Gradient Boosting | 0.637 | 0.819 | 0.766 | 1.44 ms | 3/5 | 0.615 |
| XGBoost | 0.646 | 0.800 | 0.732 | 2.35 ms | 3/5 | 0.457 |
| Random Forest | 0.642 | 0.798 | 0.758 | 9.34 ms | 2/5 | 0.391 |
| Baseline 3 | 0.522 | 0.774 | 0.690 | 2.13 ms | 5/5 | 0.449 (baseline) |

**Why Logistic Regression for the fallback:** it is the Balanced winner (0.699). It stays within 0.008 NDCG@10 of
LightGBM, with the best MRR and full explainability. A trained model refuses features from another embedding model, so
each embedding path has its own matching model (`config/matching_model.yaml`).

### What the model relies on (Phase 11, global SHAP over all 21,432 validation pairs, log-odds)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | branch_match | 1.976 |
| 2 | career_interest_alignment | 1.125 |
| 3 | domain_similarity | 0.464 |
| 4 | missing_skill_count | 0.382 |
| 5 | semantic_similarity | 0.351 |
| 6 | project_relevance | 0.302 |
| 7 | role_similarity | 0.292 |
| 8 | required_skill_count | 0.245 |
| 9 | certification_relevance | 0.222 |
| 10 | education_match | 0.155 |

`required_skill_match_score` ranks only 17th (0.010). The model leans on field fit (branch, interest, domain) more than
on skills, which reflects its rule-based training labels. This is a known limitation (see §6).

---

## 4. Other measured components

| Component | Metric | Result | Source |
|---|---|---|---|
| PII detector (Phase 4) | Recall on a labelled synthetic set | url 100%, phone 83%, email 75%, address 75%, dob 67%, **name 62%**; 0/12 false positives | `backend/tests/test_pii.py`, ADR 0005 |
| PII in logs and traces (Phase 12) | Literal name/email/phone strings found after a full five-agent run | **0** in log lines, 0 in persisted `agent_runs` / `agent_messages`, 0 in the trace tree | `backend/tests/test_tracing.py` |
| Explanations (Phase 11) | LLM lines that pass the evidence validator, live | 3 of 3 pairs `llm_validated` (gemini-3.5-flash-lite, 2026-10-15); otherwise the validated structured fallback | `scripts/demo.py 16` |
| Learning resources (Phase 10) | Invented resources | 0: every recommendation traces to a curated catalogue row (20 courses) or a YouTube API result (124 videos) | ADR 0018 |
| Cohort benchmark (Phase 12) | Other students' ids in a response | 0 of 285, for students in two different branches | `backend/tests/test_cohort_analytics.py` |
| End-to-end latency (Phase 12 traces) | Median full request (19 traced live requests) | 3.1 s; matching median 1.2 s; first-time course embedding up to ~15 s | `agent_runs` |

---

## 5. Reproducing these numbers

```bash
python -m src.embeddings.model_evaluation_engine --labels data/benchmark/human_labels.csv --reuse-cost --record-db
python -m app.matching.training_data
python -m src.scoring.matching_model_selection_engine --labels data/benchmark/human_labels.csv \
    --reuse-models mms-20261014T150744Z-nolabels --record-db
python -m src.scoring.shap_explainer global
```

Every run writes `results/<area>/<experiment_id>.{md,json}` and, with `--record-db`, the `experiment_runs` and
`experiment_metrics` tables (`GET /experiments`). The dataset hashes are pinned in PROGRESS.md.

## 6. Limitations (read before quoting a number)

- **Small validation set:** 15 students, 180 pairs, one labeller. Most model-to-model differences are not
  statistically distinguishable, and the test split hasn't been used yet.
- **The job corpus is synthetic and templated:** 4,990 jobs, 376 distinct role texts, with qualifications and work
  types assigned near-uniformly. Results describe this data, not real job boards.
- **Matching models learn rule-based labels.** That is why `branch_match` and `career_interest_alignment` dominate. Only
  more human-labelled training data can show whether this generalises.
- **Hyperparameters are fixed, not tuned.** Suitability is a class-balanced ranking score, not a calibrated
  probability; LightGBM scores cluster near 1 for strong jobs.
- **Costs come from one machine each** (laptop for embeddings, WSL2 for matching). Only the relative order transfers.
