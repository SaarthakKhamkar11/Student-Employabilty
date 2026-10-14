# ADR 0015: Matching model: binary suitability target, weak-label training, and a selection rule fixed before the human labels

**Status:** Accepted, 2026-10-11 (Phase 8): the target formulation, the training design, the candidates and the
selection rule. **Models selected 2026-10-14 on the human labels, one per embedding model** (a trained model refuses
other embeddings; `config/matching_model.yaml` holds both):
- **E5, the primary model: LightGBM.** Selection `mms-20261014T151058Z-human`, artifacts
  `mms-20261014T150744Z-nolabels`, features `features-v1-intfloat_e5-base-v2@f52bf8ec8c71`.
- **TF-IDF, the fallback: Logistic Regression.** Selection `mms-20261014T132323Z-human`, artifacts
  `mms-20261011T104041Z-nolabels`.

Both replaced Baseline 3 (`weighted_baseline_v1`, [ADR 0014](0014-matching-pipeline-and-weighted-baseline.md)).

**Human-label result on E5 features (2026-10-14): LightGBM.**
- **Labels:** the same 180 validation pairs (15 students, 50 suitable).
- **Decision line:** "Select LightGBM as the matching model (Balanced winner, Overall Score 0.615)."

  | Model | NDCG@10 | F1 | MRR | Overall (Balanced) |
  |---|---|---|---|---|
  | **LightGBM** | **0.844 (0.762–0.908)** | **0.630** | 0.739 | **0.615** |
  | Random Forest | 0.838 | 0.590 | 0.819 | 0.460 |
  | Logistic Regression | 0.817 | 0.555 | 0.743 | 0.518 |
  | XGBoost | 0.817 | 0.612 | 0.712 | 0.353 |
  | Gradient Boosting | 0.815 | 0.610 | 0.714 | 0.474 |
  | Decision Tree | 0.783 | 0.553 | 0.710 | 0.428 |
  | Baseline 3 | 0.778 | 0.587 | 0.654 | 0.523 |

- **Stability:** LightGBM wins all three weight scenarios and 76.2% of random weightings (Logistic Regression 17.2%).
- **Caveat:** LightGBM vs Baseline 3 is +0.066 NDCG@10, 95% CI −0.015 to +0.156, **not distinguishable** at 15
  students.
- **Effect on Phase 9:** a tree model's score moves in steps, so counterfactual skill importances are tiny
  (around 0.0001–0.0004) and some are negative. Phase 11 must explain this rather than present the importances as
  precise.

**Human-label result on TF-IDF features (2026-10-14), now the fallback's model.**
- **Labels:** 180 validation pairs, 15 students, 50 suitable (grade ≥ 2). One labeller.
- **Decision line:** "Select Logistic Regression as the matching model (Balanced winner, Overall Score 0.699)."
- **Quality, NDCG@10 (95% CI):**

  | Model | NDCG@10 | F1 |
  |---|---|---|
  | **Logistic Regression** | **0.813 (0.729–0.881)** | 0.606 |
  | LightGBM | 0.821 | 0.683 |
  | Gradient Boosting | 0.819 | 0.637 |
  | XGBoost | 0.800 | 0.646 |
  | Random Forest | 0.798 | 0.642 |
  | Decision Tree | 0.792 | 0.621 |
  | Baseline 3 | 0.774 | 0.522 |

  - **No candidate is distinguishable from Baseline 3** on NDCG@10. Logistic Regression: +0.039, CI −0.001 to
    +0.096.
  - **No candidate is distinguishable from Logistic Regression** either (LightGBM: +0.008, CI −0.038 to +0.052).
    The simpler-model check prefers it.
- **The winner changes with the weights:** Quality-prioritised and Low-resource pick LightGBM. Over random
  weightings, Logistic Regression wins 58.5% and LightGBM 33.6%.
- **Known effect on Phase 9:** under Logistic Regression, some counterfactual importances are **negative**.
  Detecting the skill lowers the score slightly; for example −0.003 for "SQL and database querying". The model's
  coefficients on the correlated skill-count features cause this, not the skill itself. Phase 10 sorts these
  gaps last.

**Context.**
- Spec §10 and §13 and the Phase 8 prompt require:
  - the prediction target to be decided from the labels that actually exist;
  - six candidate models, compared on the same features, split and benchmark;
  - the full metric suite and a second selection engine with transparent weights and a sensitivity analysis;
  - **never choosing XGBoost automatically**.
- **The label distribution, 2026-10-11** (reported before deciding, as the prompt asks):
  - **Human labels: none.** The labelling pool is `pool-5c6a5972b32d`: 30 students × 12 jobs = 360 pairs. All of
    them are **validation (15 students) or test (15 students)**, because Phase 6 trained nothing. Graded so far: 0.
    The human benchmark therefore has **no training split**.
  - **Weak labels** (`data/benchmark/weak_labels.csv`, rule-based, validity LIMITED): 107,536 pairs, 286 students ×
    376 jobs. Grades used are only 0, 1 and 3; **grade 2 never occurs**.

    | Split | 0 | 1 | 2 | 3 | Share 3 |
    |---|---|---|---|---|---|
    | train | 63,489 | 2,361 | 0 | 1,830 | 2.7% |
    | validation | 20,120 | 736 | 0 | 576 | 2.7% |
    | test | 17,239 | 702 | 0 | 483 | 2.6% |
- ADR 0014: the weak labels share the role families with `branch_match`, and reflect the student's stated
  domain, as `career_interest_alignment` does. **They can't select a model.**

**Decision.**

1. **Target: binary job suitability, `relevance_label >= 2` ("suitable": a moderate or strong match).**
   Chosen by the project owner on 2026-10-11, from three options.
   - **Why not (b), regression on 0–3?** The only training labels have no grade 2: a 0/1/3 target with 94% zeros
     is a binary target in disguise, and a regressor would mostly learn the 0-vs-3 split anyway.
   - **Why not (c), learning to rank?** Only XGBoost and LightGBM have native rankers. Logistic Regression, the
     trees, the forest and scikit-learn's Gradient Boosting would need a pointwise stand-in, so the six-way
     comparison would not be like for like. A pointwise probability already ranks each student's jobs.
   - **Why (a) fits:**
     - It is the same threshold every ranking metric already uses (relevant = grade >= 2, Phase 6).
     - All six candidates support it in the same way.
     - It is robust to how the human labels split between 2 and 3.
   - **Revisit when:** the human grades show that 2 vs 3 matters AND graded training-split pairs exist.
2. **Training data: weak labels, train split only. Selection: human labels, validation split. Test: held back for
   Phase 14.** Chosen by the project owner on 2026-10-11 over two alternatives:
   - grouped cross-validation on the 15 human validation students: too few to train on;
   - grading a new train-split pool first: blocks the phase.

   Consequences:
   - Every model learns the weak-label rules, including their leakage. The human labels are what show whether that
     transfers to real fit.
   - The models are trained **once**. The human-label run only evaluates the same artifacts
     (`--reuse-models <id>`), so no model can be refitted after its score is seen.
3. **Training is the same for every model** (`config/matching_model_candidates.yaml`):
   - **Features:** the 20 `features-v1` contract features. The protected-attribute guard runs on the actual
     training columns.
   - **Rows, seed and threads:** the same rows, seed 42 and a single thread for every model.
   - **Class balance:** balanced class weights (a sample weight for scikit-learn's Gradient Boosting).
   - **Decision threshold:** 0.5, used for Accuracy, Precision, Recall and F1.
   - **Hyperparameters are fixed, each with a written reason, and not tuned.** Tuning on weak labels would make them
     a selection signal; tuning needs human-labelled training data.
   - **Missing features** are NaN:
     - XGBoost and LightGBM handle NaN natively.
     - The four scikit-learn models get median imputation plus a missing-indicator per feature inside their
       pipeline.
4. **Candidates:**
   - Logistic Regression, Decision Tree (depth 6), Random Forest, scikit-learn Gradient Boosting, XGBoost and
     LightGBM, all on the same budget (200 trees for the ensembles).
   - **Baseline 3** (structured feature matching, `weighted_baseline_v1`) is measured in the same table. Every
     trained model must beat it on NDCG@10 to be eligible.
   - **Saved formats:**
     - scikit-learn: joblib;
     - XGBoost and LightGBM: their own formats.
   - Every file's SHA-256 is in the run's manifest and is checked **before** loading, because a joblib file can run
     code.
5. **The selection engine** (`src/scoring/matching_model_selection_engine.py`, weights in
   `config/matching_model_selection_weights.yaml`, fixed on 2026-10-11 before any human label):
   - **Columns:** F1, NDCG@10, MRR, Latency, Training Time, Memory, Complexity, Explainability, Overall Score.
   - **Normalisation:** min-max across the 7 models; latency, training time and memory on log10.
   - **Scenarios:** Balanced, Quality-prioritized and Low-resource deployment, each table shown in full. Plus 1,000
     random weightings.
   - **Simpler-model check:** the report states how close the best simple model (LR or DT) is to the winner.
   - **Explainability and complexity are measured, not asserted.**
     - SHAP compatibility is checked per model with the explainer actually used:
       - LinearExplainer for Logistic Regression;
       - TreeExplainer for the trees and ensembles;
       - for every model, the contributions must add up to the model's output on 200 validation pairs.
     - Points rubrics are in the candidates file.
6. **Decision rule** (`recommend()`), the same shape as ADR 0012:
   - Only `human_reviewed` labels with evaluation validity OK can select.
   - The Balanced winner among eligible models is selected.
   - **Exception:** keep Baseline 3 when it scores higher overall AND the winner's NDCG@10 gain over it is not
     statistically distinguishable (paired bootstrap over students).
   - No model is named in code. If XGBoost wins, the numbers show why.
7. **Wiring (Stage 4).**
   - `config/matching_model.yaml` `active` is the flag. `MATCHING_MODEL` overrides it for one process (demos,
     tests).
   - `app/matching/predictor.py` loads either Baseline 3 or a trained artifact from
     `models/matching/<artifacts_experiment>/`.
   - **Metadata:**
     - `matching_model` is the model key.
     - `matching_model_version` is `<key>+<artifact sha256[:12]>`.
     - `experiment_id` is the selection run, or the training run when the model is forced by the override.
   - **Refusals:** a trained model refuses to predict with another feature or embedding version.
   - **`--activate`** writes a `select` decision into the flag. It refuses any other decision.
   - **Baseline 3 is kept**, not deleted: it is the default until a human-label selection, and Phase 14's
     "Baseline 3: structured feature matching".

**Evidence** (2026-10-11, recorded in Postgres and in `results/matching_model_evaluation/`). All runs used
`features-v1` on `tfidf-953a75d166d8`: 107,536 pairs, 0 unscored, with the protected-attribute guard passing on the
training columns.

- **Training run `mms-20261011T104041Z-nolabels`**:
  - all 6 candidates and Baseline 3, each measured in its own process: Linux container on WSL2, x86_64, Python
    3.11.17;
  - artifacts in `models/matching/mms-20261011T104041Z-nolabels/`;
  - every refit was bit-identical;
  - every model's SHAP contributions add up to its output (max error 1e-5 for XGBoost, at or below 1e-12 for the
    others);
  - quality: **Not yet evaluated**; decision: **Insufficient evidence**.

  | Model | Latency, 50 pairs | Training (median of 3) | Memory added | Size | Complexity | Explainability |
  |---|---|---|---|---|---|---|
  | Logistic Regression | 1.28 ms | 4.03 s | 61.8 MB | 2.3 KB | 4 | 5 |
  | Decision Tree | 1.27 ms | 0.20 s | 42.1 MB | 3.9 KB | 2 | 5 |
  | Random Forest | 9.34 ms | 5.33 s | 44.6 MB | 2.4 MB | 3 | 2 |
  | Gradient Boosting (sklearn) | 1.44 ms | 11.12 s | 44.0 MB | 90 KB | 3 | 3 |
  | XGBoost | 2.35 ms | 1.12 s | 8.0 MB | 352 KB | 5 | 3 |
  | LightGBM | 0.45 ms | 1.03 s | 22.8 MB | 346 KB | 6 | 3 |
  | Baseline 3 | 2.13 ms | 0 s | 0 MB | 7.4 KB | 0 | 5 |

  Random Forest's explainability (2) sits on the rubric's 10 ms edge: TreeSHAP took 9.8 ms per pair in a first run
  and just over 10 ms in this one.
- **Weak-label diagnostic `mms-20261011T104404Z-weakly`** (same artifacts; weak validation split, 21,432 pairs,
  57 students; **NOT A SELECTION**, validity LIMITED):

  | Model | F1 | NDCG@10 | MRR | Overall (Balanced) |
  |---|---|---|---|---|
  | Logistic Regression | 0.375 | 0.813 | 0.948 | 0.571 |
  | Decision Tree | 0.541 | 0.823 | 1.000 | 0.711 |
  | Random Forest | 0.619 | 0.910 | 0.981 | 0.466 |
  | Gradient Boosting (sklearn) | 0.626 | 0.929 | 0.991 | 0.606 |
  | XGBoost | 0.720 | 0.954 | 1.000 | 0.629 |
  | LightGBM | 0.804 | 0.954 | 1.000 | 0.687 |
  | Baseline 3 | 0.349 | 0.515 | 0.714 | 0.449 |

  - **Winners by scenario:**
    - Balanced: Decision Tree;
    - Quality-prioritized: LightGBM;
    - Low-resource: Decision Tree.

    **The winner changes by scenario.** Over 1,000 random weightings, Decision Tree wins 54.9% and LightGBM 44.3%.
  - **XGBoost** is never a scenario winner. It is +0.131 NDCG@10 above the best simple model (Decision Tree), a
    distinguishable gap, at 1.9x its latency, +3 complexity and −2 explainability points.
  - **Read with care:** the models were trained on the same rules these labels come from. Every trained model's top SHAP
    feature is `branch_match`, the leakage ADR 0014 predicted.
- **Live wiring check** (backend restarted with `MATCHING_MODEL=decision_tree` for one request, then restored):
  - For S00002 (Data Science branch), the agent returned Data Analyst and Business Analyst jobs first, with
    `matching_model=decision_tree` and `matching_model_version=decision_tree+8a11032b8829`. Experiment:
    `mms-20261011T104041Z-nolabels`.
  - For an uploaded résumé with no branch and no stated interest, the same tree scored every job 0.0. With
    `branch_match` and `career_interest_alignment` missing, a weak-label tree has no other signal: the ranking
    fell back to the id tie-break. This is a concrete reason not to select on weak labels.
- **Platform note:**
  - On this machine (Linux x86_64 container) the TF-IDF fit is `tfidf-953a75d166d8`, not the M2's
    `tfidf-3ec097a5876e`, on byte-identical data. The version hashes IDF values rounded to 6 decimals, and 1,806
    of them sit within 1e-9 of a rounding boundary, so a last-bit difference in `log` across platforms is the
    likely cause.
  - The calibration was rebuilt for `953a…` as ADR 0014 prescribes after a re-embedding.
  - `test_the_deployed_tfidf_version_is_unchanged_by_the_benchmark_rewrite` fails on this platform for the same
    reason.

**Consequences.**
- When the team has graded the pool:
  1. Export the labels.
  2. Run `python -m src.scoring.matching_model_selection_engine --labels data/benchmark/human_labels.csv
     --reuse-models <artifacts experiment> --record-db --activate`.
  3. Record the decision here and set this ADR's model choice to Accepted.
- If the embedding model changes (ADR 0012), the features change: rebuild the feature matrix, retrain and select
  again. A trained model refuses to run on other embeddings.
