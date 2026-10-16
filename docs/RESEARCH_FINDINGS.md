# Research findings: answers to RQ1–RQ9 (spec §33)

Phase 14, 2026-10-15. Every answer below names the phase, file and table it comes from. Numbers are copied from
generated reports, never estimated. Where a question has no automatic metric, the answer says so and labels the
judgement as subjective. Method and decisions: [ADR 0021](adr/0021-evaluation-framework.md).

## How to read these answers

- **Labels.** The only ground truth is the human benchmark: 360 student × job pairs graded 0–3 by one labeller
  (rubric-v1), from 30 students. Grade ≥ 2 counts as relevant.
  - **Validation split (15 students, 180 pairs):** Phase 6 selected E5 here and Phase 8 selected the matching model
    here. Numbers from it favour the selected models by construction.
  - **Test split (15 students, 180 pairs; grades 0/1/2/3 = 71/58/37/14):** held back until Phase 14 and used once, in
    `results/baseline_comparison.md`. This is the unbiased estimate.
- **"Not statistically distinguishable"** means the paired 95% bootstrap CI over students contains 0. With 15 students,
  most differences here are of that kind. That says the sample is small; it does not prove the models are equal.
- **The job corpus is synthetic and templated** (4,990 postings, 376 distinct role texts; Phase 1 audit I-7). Every
  answer describes this data, not real job boards.

| RQ | Short answer | Main evidence |
|---|---|---|
| RQ1 | **E5 (`intfloat/e5-base-v2`)**, selected by the pre-registered rule; the only off-the-shelf model distinguishably better than TF-IDF on validation | Phase 6, `results/embedding_evaluation/emb-20261014T145548Z-human.md` |
| RQ2 | **Not shown on held-out data.** Semantic (E5) scores higher than keyword and TF-IDF on test NDCG@5, but neither difference is distinguishable | Part A, `results/baseline_comparison.md` (test) |
| RQ3 | **Not shown on held-out data.** Hybrid scores higher than semantic-only on test NDCG@5 (+0.018), not distinguishable; semantic-only has the higher MRR | Part A, `results/baseline_comparison.md` (test) |
| RQ4 | **Logistic Regression** (features-v2), selected by the pre-registered rule; LightGBM within 0.003 NDCG@10 | Phase 8, `results/matching_model_evaluation/mms-20261015T124013Z-human.md` |
| RQ5 | **Diminishing returns.** Each step up in complexity bought little or no distinguishable accuracy | Phase 8 Overall Score components; Part A test ladder |
| RQ6 | **Depends on the weighting:** E5 (embeddings) and Logistic Regression (matching) under Balanced; MiniLM for low-resource embeddings; TF-IDF beats all on cost-weighted Overall Score | Phase 6/8 sensitivity analyses; Part D measurements |
| RQ7 | **Yes for faithfulness to the model** (exact SHAP additivity, validated evidence lines, tests pass). **Partly for the match itself:** the model leans on features that leak the weak training labels | Phase 11 tests and global SHAP; live traces; Part C |
| RQ8 | **Subjective judgement: useful context, with clear limits** | Phase 12 cohort outputs (examples below) |
| RQ9 | **It routes correctly:** 67/67 traced requests ran exactly their task's agents, for 3–5 ms of orchestration overhead per request. The real cost is code size and concepts | `results/trace_route_audit.md`; Part D |

---

## RQ1: Which embedding model provides the best semantic job/student matching?

**Answer: E5 (`intfloat/e5-base-v2`, pinned `f52bf8ec8c71`).** This is Phase 6's Model Evaluation Engine decision,
quoted, not re-derived. The selection rule was fixed in ADR 0012 before any human label existed.

Source: `results/embedding_evaluation/emb-20261014T145548Z-human.md` (validation split, 15 students, 180 pairs,
full-text protocol, K = 5):

| Model | MRR | NDCG@5 (95% CI) | vs TF-IDF floor, NDCG@5 | Overall Score (Balanced) |
|---|---|---|---|---|
| TF-IDF (floor) | 0.719 | 0.602 (0.466–0.725) | — | 0.678 (floor, not a candidate) |
| A. SBERT all-mpnet-base-v2 | 0.689 | 0.588 | −0.014, not distinguishable | 0.093 (below floor) |
| **B. E5 e5-base-v2** | **0.910** | **0.756 (0.646–0.856)** | **+0.154 (+0.005 to +0.314), distinguishable** | **0.512** |
| C. BGE bge-base-en-v1.5 | 0.764 | 0.697 | +0.095, not distinguishable | 0.366 |
| D. MiniLM-L6-v2 | 0.601 | 0.619 | +0.017, not distinguishable | 0.348 |
| E. E5 domain-adapted | 0.765 | 0.745 | +0.144 (+0.022 to +0.273) | 0.466 |

- **E5 wins** Balanced and Quality-prioritised, and 75.6% of 1,000 random weightings.
- **E5 vs the runner-up (E5-domain) is not distinguishable:** +0.011, CI −0.085 to +0.112. The "best" model is the
  pre-registered rule's winner, not a statistically separated one.
- **The winner depends on the protocol.** Under the section-weighted scoring protocol, MiniLM would win Balanced
  (same report, "Does the winner change with the scoring protocol? Yes").
- **Held-out check** (Part A, test split; only E5 and TF-IDF were re-scored on test). E5's NDCG@5 drops from 0.756 to
  0.720, and its lead over TF-IDF shrinks from +0.154 to +0.038, no longer distinguishable. This is the expected
  winner's-curse effect of selecting on validation; RQ2 covers it.

## RQ2: Does semantic matching outperform keyword matching?

**Answer: not demonstrated on held-out data.** On the test split, semantic matching (Baseline 4, E5 alone) has a higher
NDCG@5 than both keyword baselines, but neither difference is statistically distinguishable at 15 students. On validation
(the selection split), E5 did beat TF-IDF distinguishably. That gap did not survive on unseen students.

Source: `results/baseline_comparison.md`, run `base-20261015T141402Z-human`, test split. All five baselines rank the
identical 180 pairs.

| Comparison (a − b) | Δ NDCG@5 (primary) | Paired 95% CI | Verdict | Δ NDCG@10 | Δ MRR |
|---|---|---|---|---|---|
| B4 semantic − B1 normalised keyword | +0.029 | −0.120 to +0.188 | higher, not distinguishable | +0.062 (−0.020 to +0.148) | +0.092 (−0.062 to +0.262) |
| B4 semantic − B2 TF-IDF | +0.038 | −0.050 to +0.140 | higher, not distinguishable | +0.028 (−0.021 to +0.084) | **+0.095 (+0.003 to +0.216)** |

- **The only CI that excludes 0 is a secondary metric:** MRR, E5 vs TF-IDF. Secondary metrics were added after the
  first run and are never decisive. With 2 metrics × 6 comparisons, one marginal hit is within chance. Read it as
  "E5 puts a relevant job first more often" — suggestive, not established.
- **Keyword matching is a stronger baseline than expected.** B1 (0.691 test NDCG@5) ties TF-IDF (0.682). Part C shows
  why: `skill_match_score` alone has AUC 0.705 against the test labels. Skills named on both sides are real evidence
  of fit.
- **Validation, for comparison** (favours E5, which was selected there): B4 − B2 = +0.154 (+0.005 to +0.314),
  distinguishable. B4 − B1 = +0.096 (−0.021 to +0.227), not distinguishable.
- **What would settle it:** more labelled test students. The CI half-widths (≈ 0.1–0.15 NDCG@5) are larger than the
  observed effects.

## RQ3: Does hybrid matching outperform semantic-only matching?

**Answer: not demonstrated.** On test, the hybrid (Baseline 5: E5 similarity features + structured features + the
trained Logistic Regression, i.e. the production pipeline's scorer) scores +0.018 NDCG@5 over semantic-only E5. The CI
is −0.050 to +0.098, not distinguishable. Semantic-only has the higher MRR (0.864 vs 0.833).

Source: `results/baseline_comparison.md`, test split.

| Baseline | P@5 | R@5 | MRR | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| B1 normalised keyword | 0.440 | 0.653 | 0.772 | 0.691 | 0.782 |
| B2 TF-IDF + cosine | 0.453 | 0.709 | 0.769 | 0.682 | 0.816 |
| B3 structured feature matching (Phase 7 weights) | 0.440 | 0.660 | 0.728 | 0.693 | 0.819 |
| B4 semantic (E5 alone) | **0.480** | 0.686 | **0.864** | 0.720 | 0.843 |
| B5 hybrid (production) | 0.467 | **0.729** | 0.833 | **0.738** | **0.862** |

| Comparison | Δ NDCG@5 (95% CI) | Δ NDCG@10 | Δ MRR |
|---|---|---|---|
| B5 hybrid − B4 semantic | +0.018 (−0.050 to +0.098) | +0.019 (−0.017 to +0.059) | −0.031 (−0.150 to +0.083) |
| B5 hybrid − B3 structured | +0.045 (−0.085 to +0.171) | — | — |
| B3 structured − B4 semantic | −0.027 (−0.134 to +0.086) | — | — |

- **Point estimates order the baselines by complexity** (B5 > B4 > B3 ≈ B1 ≈ B2 on NDCG@5), but no step is
  distinguishable.
- **Why the hybrid can't pull ahead:** it is trained on rule-based weak labels, and its two strongest features leak
  those rules (Part C: `career_interest_alignment` AUC 0.954 on the weak labels vs 0.794 on test). It partly learns
  the labelling rule instead of fit. Only human-labelled training data can test whether a hybrid trained on real fit
  beats E5.
- **What this comparison does not test:** Stage 1 eligibility and Stage 2 top-50 retrieval. All judged jobs are ranked.
  The hybrid adds value there too (explanations, skill gaps, hard filters), and none of that is in these metrics.

## RQ4: Which prediction model provides the best suitability prediction?

**Answer: Logistic Regression** (on features-v2, E5 similarity features), chosen by Phase 8's Matching Model Selection
Engine under the rule fixed in ADR 0015. This is quoted, not re-derived.

Source: `results/matching_model_evaluation/mms-20261015T124013Z-human.md` (validation split, 180 pairs, 50 suitable;
models trained on train-split weak labels):

| Model | F1 | ROC-AUC | MRR | NDCG@10 (95% CI) | vs Baseline 3, NDCG@10 | Overall (Balanced) |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.555 | 0.766 | 0.810 | 0.840 (0.755–0.909) | +0.039, not distinguishable | **0.669** |
| LightGBM | **0.641** | 0.839 | 0.772 | **0.843** (0.759–0.910) | +0.042, n.d. | 0.546 |
| Random Forest | 0.586 | 0.839 | **0.819** | 0.838 | +0.037, n.d. | 0.444 |
| Gradient Boosting | 0.629 | 0.829 | 0.751 | 0.822 | +0.021, n.d. | 0.530 |
| XGBoost | 0.607 | 0.834 | 0.748 | 0.820 | +0.019, n.d. | 0.317 |
| Decision Tree | 0.563 | 0.776 | 0.708 | 0.784 | −0.017, n.d. (below the gate) | 0.405 |
| Baseline 3 (weighted sum) | 0.582 | 0.795 | 0.761 | 0.801 | — | 0.652 (baseline) |

- **Logistic Regression wins all three weight scenarios and 81.1% of random weightings.**
- **On ranking quality alone, no model is distinguishable from Baseline 3.** LightGBM is the best classifier (F1,
  ROC-AUC), and LR wins on explainability (5/5) and latency.
- **Held-out check** (Part A, test, not a re-selection). The selected model scores NDCG@10 0.862 vs Baseline 3's 0.819
  (B5 − B3 NDCG@5 +0.045, CI −0.085 to +0.171): the same direction as on validation, still not distinguishable.
- **History:** on features-v1 (2026-10-14) the same rule chose LightGBM (`mms-20261014T151058Z-human`). The choice
  flipped when the features changed. The two models' quality is that close.

## RQ5: What is the trade-off between accuracy and computational complexity?

**Answer: accuracy rises much more slowly than complexity, and on held-out data no complexity step bought a
statistically distinguishable gain.** This draws on Phase 8's Overall Score components (quality vs complexity /
explainability / cost) and on Part A's test-split ladder.

**Within the matching models** (Phase 8, `mms-20261015T124013Z-human`, criteria and normalised values in its JSON).
Complexity points come from the rubric in `config/matching_model_candidates.yaml`:

| Model | Complexity (points) | Explainability (/5) | NDCG@10 | F1 | Latency, 50 pairs |
|---|---|---|---|---|---|
| Baseline 3 | 0 | 5 | 0.801 | 0.582 | 2.34 ms |
| Decision Tree | 2 | 5 | 0.784 | 0.563 | 1.44 ms |
| Random Forest | 3 | 2 | 0.838 | 0.586 | 9.79 ms |
| Gradient Boosting | 3 | 3 | 0.822 | 0.629 | 1.68 ms |
| **Logistic Regression** | 4 | 5 | 0.840 | 0.555 | 1.49 ms |
| XGBoost | 5 | 2 | 0.820 | 0.607 | 9.34 ms |
| LightGBM | 6 | 3 | 0.843 | 0.641 | 6.54 ms |

- **Ranking gains per step:** 0 → 4 points (Baseline 3 → LR) bought +0.039 NDCG@10. 4 → 6 (LR → LightGBM) bought
  +0.003. Neither is distinguishable.
- **Classification is where complexity helps:** LightGBM's F1 is +0.086 over LR's. The Balanced weights (quality 0.50:
  F1 0.15, NDCG 0.20, MRR 0.15; complexity 0.10; explainability 0.20) value LR's explainability and simplicity more
  than that.
- **Under Quality-prioritised weights the two nearly tie** (LR 0.691 vs LightGBM 0.688). Under Low-resource weights,
  untrained Baseline 3 scores highest (0.735).

**Across the whole pipeline** (Part A, test NDCG@5, with each baseline's cost measured in Phases 6 and 8):

| Step up | What it adds | Test NDCG@5 | Δ (CI) |
|---|---|---|---|
| B1 keyword | normaliser only | 0.691 | — |
| B2 TF-IDF | a fitted vocabulary, 1.9 ms / pair | 0.682 | −0.009 vs B1 (−0.137 to +0.132) |
| B4 E5 | a 418 MB transformer, 654 ms / pair on the laptop CPU (Phase 6) | 0.720 | +0.038 vs B2 (−0.050 to +0.140) |
| B5 hybrid | 20 features, calibration, a trained model, 6 similarity signals | 0.738 | +0.018 vs B4 (−0.050 to +0.098) |

- **Cost of B2 → B4:** about 350× the per-pair latency and 6.6× the memory (Phase 6: 1.9 → 654 ms, 97 → 635 MB) for
  +0.038 NDCG@5 on held-out students.
- **Phase 6 records the same trade-off:** under its Balanced weights the TF-IDF floor's Overall Score (0.678) is above
  E5's (0.512). E5 was selected only because the rule also requires a statistically distinguishable quality gain on
  validation, which it had there. That gain shrank on test.

## RQ6: Which model provides the best accuracy–latency–memory trade-off?

**Answer: it depends on the weighting, and the reports show where it flips.** The three-way view comes from the
sensitivity analyses of Phases 6 and 8; Part D adds the deployed system's measured costs.

**Embeddings** (Phase 6, `emb-20261014T145548Z-human`; latency = CPU per pair, memory = peak added):

| Model | NDCG@5 (val) | Latency | Memory | Balanced | Quality-prioritised | Low-resource |
|---|---|---|---|---|---|---|
| TF-IDF (floor) | 0.602 | 1.9 ms | 97 MB | 0.678 | 0.533 | 0.778 |
| E5 | 0.756 | 654 ms | 635 MB | **0.512** | **0.718** | 0.352 |
| MiniLM | 0.619 | 124 ms | 210 MB | 0.348 | 0.317 | **0.368** |
| BGE | 0.697 | 643 ms | 654 MB | 0.366 | 0.494 | 0.256 |

- **Winners among eligible candidates:** E5 for Balanced and Quality-prioritised; **MiniLM for Low-resource.**
- **Random weightings:** E5 75.6%, E5-domain 15.3%, MiniLM 9.1%.
- **TF-IDF has the best three-way trade-off of all** if it is allowed to compete. It is the floor, not a candidate.

**Matching models** (Phase 8, `mms-20261015T124013Z-human`): Logistic Regression wins Balanced (0.669),
Quality-prioritised (0.691) and Low-resource among trained models (0.548), and 81.1% of random weightings. It is
among the fastest (1.49 ms per 50 pairs) but has the largest training-memory footprint (60.5 MB added during the fit).
XGBoost is the lightest in memory (7.9 MB) and the slowest to explain.

**In the deployed system** (Part D, `results/performance_benchmark.md`, measured on this machine, no LLM): the E5 choice shows up mainly at
ingestion. Embedding a student's sections takes a median 148 ms (`5k`) to 336 ms (`real`), and a job's
93 ms (`real`) to 134 ms (`1k`). Everything else in ingestion is under 10 ms per entity: parsing 0.2–1.2 ms, skill extraction 6–7 ms.
A full student request takes a median 1.3–1.9 s end to end, of which the Matching Agent is 1.2–1.8 s. The
production Logistic Regression's prediction stage, which includes feature building for the top 50, takes
0.2–0.6 s. Its model call alone is 1.5 ms per 50 pairs (Phase 8), so feature building, not the model, is the
cost. The process peaks at 2.3–2.5 GB, mostly E5 plus its runtime. The trade-off that matters in deployment is
therefore E5's ingestion cost and memory against its quality gain, not the matching model's.

## RQ7: Can SHAP and evidence-based explanations explain individual matches?

**Answer: yes, they explain what the model did, faithfully and verifiably. Whether that is the real reason a student
fits a job is limited by the model itself.**

**Faithfulness to the model** (measured):
- **SHAP is exact for the production model.**
  - Phase 8: LR's LinearExplainer contributions add up to its log-odds output, max error 5.3e-15 on 200 validation
    pairs (`mms-20261015T124013Z-human.md`, Explainability table).
  - Phase 11's global run on the production model (`results/explainability/global-logistic_regression+a13440424c1f.json`):
    21,432 validation pairs, additivity error 7.1e-15.
- **The Phase 11 validation tests pass reliably:** `backend/tests/test_explainability_agent.py` passed **20/20 in each of 5 consecutive
  runs** (2026-10-16, backend image). The full suite: 397 passed, 21 skipped, 1 known platform-hash failure
  unrelated to explanations. The tests include:
  - `test_every_bullet_corresponds_to_a_real_contributing_feature_or_input`: every bullet greps back to a real
    feature or input.
  - `test_a_deliberately_broken_explanation_is_caught_retried_once_and_never_returned`: a draft with 13 planted errors
    is caught, retried once, and never returned.
  - `test_real_models_get_the_explainer_their_type_needs_and_values_add_up`: real LR and LightGBM artifacts.
  - `test_shap_that_does_not_reproduce_the_predicted_score_is_refused`.
- **Live runs** (the explanations stored in `agent_runs`, queried 2026-10-15): **41 of 41 explanations that were
  returned passed `validate_explanation`.** By 2026-10-16, 5 further live requests returned no explanation at all,
  because of the bug below.
  - 3 are Gemini prose (`llm_validated`). 38 are the validated structured fallback.
  - The fallbacks happened because the LLM was unavailable: free-tier quota 429 ×16, no key configured ×14, a pre-fix
    400 schema error ×6, a 504 timeout ×1, other client error ×1.
  - No stored run shows a live LLM draft rejected by the validator.

**Whether that explains the match** (measured, Part C):
- **The production model's two largest global SHAP features are the two that leak the weak training labels:**
  `career_interest_alignment` (mean |SHAP| 2.23) and `branch_match` (2.16). Next is `project_relevance` (0.61).
- **On human labels those features are much weaker** than on the weak labels they were trained on
  (`results/leakage_report.md` §5). So a faithful explanation can still say "your branch fits this role family" where
  a person grading fit would weigh skills.
- **Some evidence lines read oddly because the data is odd:** "B.TECH meets the job's minimum B.COM" (Kaggle
  qualifications are assigned near-uniformly, Phase 1 I-7).
- **Bug found by Part D: a pair with more than 12 missing job skills gets no explanation at all.**
  - The structured fallback writes one gap line per missing skill, but `ProseDraft.potential_gaps` allows at most 12
    (the Gemini-safe limit).
  - **It happens in real use:** 5 live requests (10 failed executions with retries, 2026-10-15/16), mostly
    company-side candidate ranking, and once in the `1k` benchmark run.
  - In the real data, 1 of the 376 role texts lists 13 skills, so 244 of the 107,536 labelled pairs (0.23%) can hit
    it.
  - So the Phase 11 claim "the fallback always passes its own validation" holds only up to 12 gaps. Not fixed in
    Phase 14: the fix needs a decision on how the validator's coverage rule treats the omitted gaps.
- **Not measured:** whether students find the explanations understandable or useful. That needs a user study.

## RQ8: Does cohort benchmarking provide useful additional context?

**Answer (subjective judgement, not a measured result): yes, as context, with clear limits.** No automatic metric
exists for "useful": there is no user study and no outcome data. This answer reads Phase 12's actual outputs.

Source: `src/scoring/cohort_analytics.py --student <id> --top-job` on the live database, run 2026-10-16 with the
production model (Logistic Regression, features-v2). This re-runs Phase 12's two report students; their top jobs
changed with the model.

| | S00001 (placement cell, CSE Data Science) | S00262 (Kaggle, IT) |
|---|---|---|
| Filters | branch, graduation year 2027, target role (data_ml) | branch, target role (software_dev); graduation year not stated, not applied |
| Cohort → comparable peers | 69 → 20 (mean cosine 0.887) | 30 → 20 (mean cosine 0.849) |
| Top job | Research Analyst | Java Developer |
| Student / cohort average / median score | 0.9886 / 0.9824 / 0.9885 | 0.9935 / 0.9413 / 0.9863 |
| Percentile | 50 | 90 |
| Skill coverage (top 10 cohort skills) | 0.3: Python, SQL, Power BI | 0.2: Java, JavaScript |
| Missing job skills vs cohort | 3 vs 4.8 (fewer than 67.5% of peers) | 1 vs 2.05 (72.5%) |
| Distinctive gap (most peers have it) | Critical Thinking (85% of peers) | none |
| Advantages (student has, few peers do) | Data collection and interpretation, Research methodologies (45%), Data visualization tools (40%), Research report writing (30%) | none |

**What it adds that a match score alone doesn't** (judgement):
- **"Missing 3 skills" becomes comparable:** fewer than 67.5% of similar peers. A student can see whether a gap is
  theirs alone (S00001's "Critical Thinking", which 85% of peers list) or shared by almost everyone, so less urgent
  ("Attention to Detail", 0%).
- **The advantages list names concrete differentiators** for an application.
- **Privacy holds:** aggregates only, at least 5 peers, a skill only if 3 or more peers hold it (Phase 12 test: no
  other student's id in 285 checks).

**Limits** (observed in these same outputs):
- **Scores are compressed near 1** (0.94–0.99), so the percentile separates differences in the third decimal. S00001's
  "50th percentile" means "about average", not a meaningful rank.
- **The cohort is small:** 20 comparable peers. One student's change moves the percentile by 5 points.
- **"Missing" means "not detected in the profile text,"** as the output's own note says. Phase 5 skill-phrase
  artefacts reach the output ("Problem-solving Communication skills" as one skill).
- **Kaggle students lack a graduation year**, so their cohorts are broader.

**Not measured:** whether students act differently with this context. That needs a user study, so RQ8 stays a
subjective answer.

## RQ9: Does the agentic architecture improve workflow orchestration and personalisation without making the system unnecessarily complex?

**Answer: orchestration, yes, measurably, at a runtime cost of 3–5 ms per request. Personalisation is real but not
separately measured. The complexity cost is in code and concepts, not in runtime.**

**Orchestration: does the Supervisor run only what a request needs?** Source: `results/trace_route_audit.md`, built
by `src/evaluation/trace_route_audit.py` from the live `agent_runs` / `agent_messages` rows.
- **Coverage (re-run 2026-10-16):** 97 persisted requests, 67 with Phase 12 execution traces, 1,432 tool calls.
- **67 of 67 traced requests ran exactly their task's route.** Every agent in `FALLBACK_ROUTES[task]` either ran or was
  skipped with a recorded reason, and **0** requests ran an agent their task doesn't need.

| Task | Requests | Agents run | Agents never run for this task | Skips seen (reason) |
|---|---|---|---|---|
| `cohort_benchmark` | 8 | profile | job intelligence, matching, learning, explainability | — |
| `explain_match` | 19 | profile, job intelligence, matching, explainability | learning | — |
| `learning_recommendations` | 12 | profile, matching, learning (8 of 12) | job intelligence, explainability | learning ×4: "not needed: no skill gaps were identified" |
| `recommend_jobs_for_student` | 11 | profile, matching, learning, explainability | job intelligence | — |
| `rank_candidates_for_job` | 5 | job intelligence, matching, explainability | profile, learning | — |
| `ingest_student_profile` | 10 | profile; then matching, learning, explainability (8 of 10) | job intelligence | 2 failed resumes: matching skipped ("upstream profile_agent failed"), then learning and explainability |
| `ingest_job` | 2 | job intelligence, matching, explainability | profile, learning | — |

- **Failure isolation works as designed (Phase 3):**
  - A failed profile stops its dependants with a reason instead of passing bad data on.
  - A degraded agent keeps the request's matches. Explainability fell back on 14 requests (status `partial`), and
    their matches were still returned.
  - Retries: 10 tool retries and 5 agent retries in 1,432 tool calls. All 15 are the explanation bug described
    under RQ7, which a retry can't fix.
- **LLM routing adds cost and has no measured benefit here.** The routing LLM calls took a median 875 ms per request
  that tried them (most failed with 429 before the Phase 12 quota fix). Since then, requests with an explicit task skip
  the LLM, with no loss: the deterministic table routed 67/67 correctly.

**Complexity cost** (measured):
- **Latency is concentrated in one agent:** the Matching Agent (Stages 1–6 + cohort benchmark) takes a median
  1.2–1.8 s of a 1.3–1.9 s student request at every size, about 90% of it (`results/performance_benchmark.md`).
  - Explainability takes a median 24–32 ms without an LLM.
  - Learning takes 77–109 ms once its course embeddings are cached, and 1.1–1.5 s (p95) on the first request of a
    process. The live traces' 11 s learning median is that first-request cost after restarts.
- **Orchestration overhead** (Supervisor time not spent inside an agent): **3–5 ms per request at every size**: request time minus the agents' critical
  path, where agents in one Supervisor wave run concurrently (Part D). The agent architecture's runtime cost is
  negligible; its cost is in code and concepts.
- **Code size:** 6 agents and 47 distinct registered tools (profile 11, job 11, matching 11, explainability 8,
  learning 6; `backend/app/tools/`), over 12 SQL migrations (`backend/migrations/`). A single pipeline function would be smaller. In return, every stage has a typed contract, a
  trace span, a retry/degrade policy and a test.

**Personalisation:** every response is per student and per job: their skill gaps ranked by the model's own counterfactual
importance (Phase 9), resources matched to those gaps (Phase 10), and an explanation of their score (Phase 11). This is
what the agent split organises. Whether it helps students more than a single ranked list is **not measured**: there
is no user study and no outcome data.

**Is it unnecessarily complex?** Judged against the measurements above:
- **Earns its place:** the split into agents with typed hand-offs, per-task routing, degrade-on-failure and traces.
  The traces are what made this audit possible.
- **Doesn't, here:** LLM routing for explicit tasks (already disabled). The first-request course embedding in the
  Learning Agent (up to ~15 s live) could be precomputed at load time instead.

---

## Limits that apply to every answer

- 15 test students and one labeller. Most differences are not statistically distinguishable, and inter-annotator
  agreement is unmeasured.
- **The test split has now been used.** A new unbiased estimate needs newly labelled students.
- **The job corpus is synthetic and templated.** Qualifications and work types are near-uniform at random.
- **Matching models learn rule-based weak labels** that share inputs with their two strongest features
  (`results/leakage_report.md` §5).
- **Performance was measured on one machine** with no LLM and no YouTube calls. Sizes above 286 students are partially
  synthetic, and the tested ceiling is 5,000 students × 1,000 jobs (`results/performance_benchmark.md`).
