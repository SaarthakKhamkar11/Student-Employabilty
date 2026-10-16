# ADR 0021: Evaluation framework: five baselines on the held-out test split, a measured leakage report, and performance at real and scaled sizes

**Status:** Accepted, 2026-10-15 (Phase 14).

**Context.**
- **Spec §33–35, §26:** answer RQ1–RQ9 with evidence; implement five baselines and compare them under one framework;
  detect leakage and write a report; measure latency, throughput and memory at several dataset sizes, without
  claiming scalability beyond what was run.
- **Already there:** Phase 6 selected E5 and Phase 8 selected Logistic Regression on the human labels' *validation*
  split. Both held the *test* split back "for Phase 14" (ADR 0012, ADR 0015). Each phase has its own scorer and metric
  code (`src/evaluation/retrieval_metrics.py`), plus the Phase 6 circular-evaluation guard and the Phase 12 trace spans.
- **Missing:** no keyword baseline, no leakage report, no synthetic-generation script. The prompt assumed Phase 1 had
  one, but the Phase 1 audit found only the `is_synthetic` columns. There was also no measurement above 286 students.

**Decision.**

1. **One benchmark for all five baselines** (`src/evaluation/baseline_comparison.py`, `config/baseline_comparison.yaml`).
   - Every baseline scores the identical 180 test pairs (15 students × 12 judged jobs) and goes through the same metric
     code: rank each student's judged jobs, ties by job_id. Validation is reported too, but only to show that the
     module reproduces the Phase 6 and Phase 8 numbers (8 of 8 match).
   - **Nothing is re-implemented:**

     | Baseline | Scored by |
     |---|---|
     | B2 TF-IDF, B4 E5 | Phase 6's `score_matrix` |
     | B3 structured | Phase 7's `weighted_baseline_v1` on Phase 8's feature matrix |
     | B5 hybrid | The production Stage 4 model named by `config/matching_model.yaml`, hash-checked |
     | B1 keyword (new) | The agents' own skill extraction plus `normalize_skills` |

   - **Fixed before the first test run:** the primary metric (NDCG@5), the comparisons for each RQ, and the verdict
     rule. "a outperforms b" needs a paired 95% bootstrap CI over students that excludes 0. Anything else is reported
     as "higher/lower, not statistically distinguishable".
   - **Stated in the report:**
     - B3 is not purely structured: 0.38 of its weight is on E5 similarity percentiles.
     - The pool was nominated by TF-IDF, SBERT, E5, BGE and MiniLM, not by B1, B3 or B5.
     - Ranking judged jobs exercises Stages 3–4, not eligibility or top-50 retrieval.
2. **Leakage report from the real data** (`src/pipeline/leakage_report.py` → `results/leakage_report.md`). It counts
   duplicates and near-duplicates among students and jobs, split integrity in every label file and feature matrix, and
   synthetic rows. Target leakage is measured, not argued: each of the 20 features gets a single-feature ROC-AUC against
   the weak training labels and against the human labels, next to a code-level note of what the feature reads. Label
   provenance is checked, along with the circularity guard's Kendall τ and whether any stored model output is read back
   as a label.
3. **RQ9 evidence from the persisted traces** (`src/evaluation/trace_route_audit.py`). For every traced request, the
   agents that ran plus those skipped (with a reason) are compared with `FALLBACK_ROUTES[task]`.
4. **Performance on the production code, in throwaway databases** (`src/evaluation/performance_benchmark.py`).
   - Each size gets its own Postgres + pgvector database on the project's server (`phase14_bench_*`, guarded by name),
     built with the project's migrations.
   - The Profile and Job Intelligence agents ingest every entity with one worker. Sampled requests then go through the
     Supervisor as the portal sends them.
   - Stage latencies are read from the Phase 12 spans. No timers are added to the code under test.
   - **No LLM and no YouTube.** No external quota is spent, and an external service's latency would swamp the
     system's own. The report says what this leaves unmeasured.
5. **Scale-up data** (`src/pipeline/synthetic_scaleup.py`).
   - **Students:** synthetic students recombine real fields within the same source and branch. They are seeded,
     flagged `is_synthetic=True`, and list every recombined field. A larger size extends a smaller one.
   - **Jobs:** a seeded sample of real postings. No jobs are generated.
   - **Where it lives:** in memory and in the throwaway databases only. It is never in `data/`, never labelled, never
     split, and never used for quality.
   - **Sizes:**

     | Size | Students | Jobs |
     |---|---|---|
     | `real` | 286 (all real) | 4,990 |
     | `1k` | 1,000 (714 synthetic) | 1,000 |
     | `5k` | 5,000 (4,714 synthetic) | 1,000 |

   - The report names the largest size run as the tested ceiling.
6. **Phase 13 finding fixed: the Phase 6 recorder.** `record_experiment` now writes rows for the report's own models
   (`metric_rows`), not this process's `RUNNABLE`. It writes slot E's "custom: not feasible" row only when slot E did not
   run. The live database still holds one such contradictory row for `emb-20261014T145548Z-human`. It was left as is;
   deleting it is the owner's call.

**Consequences.**
- The RQ answers in `docs/RESEARCH_FINDINGS.md` cite a file, table or test for every claim, and quote test-split
  numbers as the unbiased estimate.
- The test split has now been used once. Any later change chosen by looking at it would turn it into a selection split.
  A fresh held-out set (more labelled students) is needed before quoting a new unbiased number.
- Performance numbers are specific to this machine and to "no LLM, no YouTube". The spec's "5,000–10,000 students"
  (§28) is covered only up to 5,000. 10,000 was not run and is not claimed.
