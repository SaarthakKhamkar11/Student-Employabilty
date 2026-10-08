# PROJECT_AUDIT.md — Phase 1 Repository Audit

Audit date: 2026-10-08 · Commit audited: `a5f758c` ("PHASE 1", identical to `origin/main`) · Machine: Apple M2, 16 GB, macOS (Darwin 25.6.0), arm64

Everything below was either read directly or executed in this session. Nothing was changed except creating this file.

> **Snapshot note:** this audit describes the repo as of `a5f758c`. Several issues have since been fixed: Python 3.11 venv plus `requirements.txt`, repo-root-relative paths, and the NaN missing-count bug (I-6). For current status, see [`PROGRESS.md`](PROGRESS.md).

---

## 0. Read this first: the repo is not what the Phase 1 prompt describes

The Phase 1 prompt lists about 14 files as "verified, not a guess". Only 2 source files, 2 data files and 3 result files exist. I searched the working tree, git history (one commit), every branch, the stash, `origin/main`, and `~/Desktop`, `~/Downloads` and `~/Documents` (4 levels deep).

| Described in prompt | Actual status |
|---|---|
| `preprocessing.py` | Exists as **`preprocess.py`** (different name) |
| `config/schema.py` (`JOB_SCHEMA` / `STUDENT_SCHEMA`) | **Not a file.** The two lists live inline at `preprocess.py:23-34` |
| `src/pipeline/ingest_placement_cell.py` | **Not a file.** Function `ingest_placement_cell()` at `preprocess.py:109` |
| `src/pipeline/ingest_kaggle_resumes.py` | **Not a file.** Function at `preprocess.py:154` |
| `src/pipeline/ingest_real_jobs.py` | **Not a file.** `ingest_jobs()` at `preprocess.py:193` |
| `src/pipeline/build_unified_dataset.py` | **Not a file.** The `__main__` block at `preprocess.py:242-254` |
| `src/pipeline/validate_and_report.py` | **Not a file.** `write_quality_report()` at `preprocess.py:215` |
| `src/parsing/section_parser.py`, `resume_text_parser.py` | **Not files.** `_slice_section` / `extract_skills` / `parse_resume` / `guess_degree` at `preprocess.py:39-92`. They extract **only skills, education and experience**. Projects and certifications are used only as section boundaries and are never extracted. |
| `src/embeddings/tfidf_baseline.py` | **Does not exist anywhere.** No TF-IDF results file in `results/` either. |
| `src/embeddings/embedding_model_benchmark.py` | Exists at the **repo root**: `./embedding_model_benchmark.py` |
| `docs/ADR.md` | **Does not exist anywhere.** The only rationale in the repo is the "Known limitations" comment at `preprocess.py:256-263` plus inline comments. |
| `README.md`, `docs/claude_code_prompts/README.md` | **Do not exist anywhere.** |
| "Master spec" (Sections 2, 5, 9, 13) | **Not in this session's conversation history.** Sections L, M and N below are based on `~/Desktop/Major Projec/Architecture.png` (outside the repo) and the Phase 1 prompt text. They are not based on the spec itself. |

Raw inputs are **outside the repo and unversioned**, at `~/Desktop/dataset/` (`Student Data.xlsx`, `UpdatedResumeDataSet.csv`, `job_descriptions.csv` at 695 MB).

**Consequence:** most of the ingestion, parsing and schema logic does exist, consolidated in one script. **The TF-IDF baseline and ADR.md must be recovered from wherever they were authored, or recreated.** Neither can be "confirmed working".

---

## A. Current architecture (what is really there)

```
~/Desktop/dataset/            (outside repo, not versioned, read via CWD-relative paths)
 ├─ Student Data.xlsx          120 rows × 12 cols  (placement-cell form export)
 ├─ UpdatedResumeDataSet.csv   962 rows → 166 after exact-duplicate drop
 └─ job_descriptions.csv       1,048,575 rows, 695 MB
            │
            ▼  preprocess.py   (single script; constants at top, no CLI, no config file)
            │    ├─ JOB_SCHEMA / STUDENT_SCHEMA            canonical column lists
            │    ├─ parse_resume / extract_skills          regex section slicer
            │    ├─ ingest_placement_cell                  rename + light cleaning
            │    ├─ ingest_kaggle_resumes + CATEGORY_MAP   25 categories → 6 branches / domains
            │    ├─ ingest_jobs                            dedupe, re-ID, stratified sample by country
            │    └─ write_quality_report
            ▼
data/processed/students_unified.csv   286 × 17
data/processed/jobs.csv               4,990 × 16
results/data_quality_report.md
            │
            ▼  embedding_model_benchmark.py   (argparse CLI)
            │    5 sections × weighted cosine, SBERT / E5 / BGE via sentence-transformers
            ▼
results/embedding_model_comparison.json
results/embedding_model_comparison_summary.csv
```

- Batch, file-based, two scripts. No package structure, no `requirements.txt`, no `.gitignore`. `.DS_Store` files are committed (and currently modified in the working tree).
- **No backend, database, frontend, agents, tests, CI, Docker files or config.**

## B. Current technology stack (exact versions found)

**Interpreter actually used:** `/usr/bin/python3` = **Python 3.9.6** (Apple Command Line Tools build, linked against **LibreSSL 2.8.3**; urllib3 warns about this on every run), **pip 21.2.4**. Packages live in the user site `~/Library/Python/3.9/`.

| Purpose | Package | Version |
|---|---|---|
| Data | pandas | 2.3.3 |
| | numpy | 2.0.2 |
| | openpyxl | 3.1.5 |
| | scipy | 1.13.1 |
| ML (installed, **not imported by any repo code**) | scikit-learn | 1.6.1 |
| Embeddings | sentence-transformers | 5.1.2 |
| | torch | 2.8.0 (MPS backend available) |
| | transformers | 4.57.6 |
| | tokenizers | 0.22.2 |
| | huggingface_hub | 0.36.2 |
| | safetensors | 0.7.0 |
| Profiling | psutil | 7.2.2 |

**Not installed:** `xgboost`, `lightgbm`, `shap`, `fastapi`, `uvicorn`, `sqlalchemy`, `psycopg`/`psycopg2`, `pgvector`, `redis`, `anthropic`, `pytest`.

**Other tooling on this machine:** Docker 29.8.2; Node v26.0.0 and npm 11.12.1; Homebrew Python 3.11.17 (`/opt/homebrew/bin/python3.11`); uv 0.11.7. Postgres (`psql`/`postgres`) and Redis (`redis-server`) are **not installed** locally.

**HF model cache** (`~/.cache/huggingface/hub`): `all-mpnet-base-v2` (418 MB), `e5-base-v2` (419 MB), `bge-base-en-v1.5` (419 MB). All three were already downloaded before this session.

The full `pip freeze` is in the Appendix.

## C. Existing functionality (verified by running it in this session)

### C1. Dependency install (as documented)
```
$ /usr/bin/python3 -m pip install pandas numpy openpyxl scikit-learn
Requirement already satisfied: ... (all satisfied — nothing installed or changed)
```

### C2. `preprocess.py` — WORKS, and is byte-for-byte reproducible
The input paths are CWD-relative (`preprocess.py:12-14`), and running from the repo root would overwrite the committed outputs. So I ran it from a scratch directory with symlinks to `~/Desktop/dataset/*`:
```
$ cd <scratch>/run_pre && /usr/bin/time -l python3 ".../preprocess.py"
students_unified.csv: 286 rows | jobs.csv: 4990 rows
# Data Quality Report
## Jobs (sampled from real job_descriptions.csv)
- Row count: 4990
- Duplicate job_id: 0
- Missing fields: {... all 0 ...}
- gender_preference distribution: {'both': 1685, 'male': 1663, 'female': 1642}
## Students (real, unified)
- Row count: 286
- Duplicate student_id: 0
- Rows by source_dataset: {'kaggle_resume_dataset': 166, 'placement_cell': 120}
- Rows by branch: {'Computer Science and Engineering (Data Science)': 130, 'Information Technology': 90,
  'Non-Engineering / Management': 50, 'Civil Engineering': 6, 'Mechanical Engineering': 5, 'Electrical Engineering': 5}
- Missing fields (blank, never fabricated): {'degree': 65, 'graduation_year': 166, 'gender': 166, 'avg_cgpa_sem5': 166,
  'achievements': 166, 'internship_raw': 0, 'technical_projects': 166, 'skills_raw': 33, 'certifications': 166,
  'country': 166, 'synthetic_fields': 286, (others 0)}
        6.14 real    5.60 user    0.31 sys
  1269581000  peak memory footprint
```
`cmp` against the committed files gives **IDENTICAL** for all three: `students_unified.csv`, `jobs.csv` and `data_quality_report.md`.
SHA-256: `students_unified.csv` = `a123ed6f…51520`, `jobs.csv` = `42c7d78b…2acb1`.
Raw row counts checked: xlsx 120; resumes 962 → 166 after dedupe; jobs 1,048,575 across 216 countries.
(The "Missing fields" line is **wrong**. See problem I-6.)

### C3. TF-IDF baseline — CANNOT RUN, the file does not exist
```
$ python3 src/embeddings/tfidf_baseline.py
can't open file '.../src/embeddings/tfidf_baseline.py': [Errno 2] No such file or directory
```

### C4. Embedding benchmark dependencies
```
$ python3 -m pip install sentence-transformers torch psutil
Requirement already satisfied: sentence-transformers (5.1.2) / torch (2.8.0) / psutil (7.2.2)
```

### C5. `embedding_model_benchmark.py` — RUNS FOR REAL on this machine
```
$ /usr/bin/time -l python3 embedding_model_benchmark.py --out_dir <scratch>/bench_run1
No --eval_labels provided: precision/recall/F1/Recall@K will be reported as null, not guessed.
model  model_size_mb  total_embed_seconds  single_item_latency_ms  overall_score_mean recall_at_k
sbert          417.7               11.021                    71.6              0.1686        None
   e5          417.6               10.610                    38.8              0.7351        None
  bge          417.6                9.603                    12.7              0.4177        None
       50.84 real    9.29 user    3.04 sys
  3129591248  peak memory footprint
```
(30 students × 500 jobs, sampled with `random_state=42`. 20 of the 30 sampled students come from the Kaggle source.)

## D. Existing ML pipeline

| Component | Status |
|---|---|
| **TF-IDF baseline** | **NOT CONFIRMED — the code is missing.** The prompt says "confirmed working", but no TF-IDF code or output exists in this repo or its history, and scikit-learn is not imported anywhere. Treat it as **Not yet evaluated**. It must be recovered or re-implemented, and it is required as the Phase 6 baseline. |
| **Embedding benchmark (SBERT / E5 / BGE)** | **Executed for real this session.** The committed `results/embedding_model_comparison.json` (dated Sep 25) also contains real numbers that match today's scores exactly, which contradicts "never run with real models". **It measures cost only, not quality.** There are no eval labels, so `recall_at_k` is null, and precision and F1 are never computed at all (I-2). `overall_score_mean` is not a quality metric and must not be used to rank models (I-3). |
| Section weighting | Hard-coded `skills 0.35 / experience 0.25 / education 0.15 / projects 0.15 / certifications 0.10` (`embedding_model_benchmark.py:17-20`). These are untuned assumptions, and the ADR that reportedly justifies them is missing. |
| Matching / suitability model | None. No labels, no features, no training code. |

Benchmark results as reported (deterministic scores, non-deterministic cost):

| | SBERT | E5 | BGE |
|---|---|---|---|
| overall_score_mean (all 3 runs) | 0.1686 | 0.7351 | 0.4177 |
| overall_score_std | 0.0592 | 0.0129 | 0.0286 |
| model_size_mb | 417.7 | 417.6 | 417.6 |
| single_item_latency_ms — committed / run 1 / run 2 (reversed order) | 1611.2 / 71.6 / 38.4 | 152.3 / 38.8 / 18.3 | 12.8 / 12.7 / 34.5 |
| total_embed_seconds — committed / run 1 / run 2 | 14.88 / 11.02 / 11.22 | 11.36 / 10.61 / 9.88 | 10.23 / 9.60 / 9.67 |
| memory_delta_embed_mb — committed / run 1 / run 2 | 234.1 / 256.5 / 175.9 | 206.3 / **−75.5** / 147.2 | **−148.2** / 121.8 / 242.8 |

Run 2 was a scratch probe that imported `benchmark_model()` and called it in the order BGE → E5 → SBERT. No source file was touched.

## E. Existing database schema
**None.** No database, ORM, migrations or SQL anywhere. The only "schema" is the two Python lists `JOB_SCHEMA` (16 cols) and `STUDENT_SCHEMA` (17 cols) in `preprocess.py:23-34`.

## F. Existing frontend
**None.**

## G. Existing backend
**None.** No API, server or service code.

## H. Existing tests
**None.** No test files, no pytest, no CI configuration. The only regression check available today is the byte-identical rerun of `preprocess.py` (C2).

## I. Existing problems

Severity: 🔴 blocks trustworthy results · 🟠 materially distorts results · 🟡 hygiene / latent.

1. 🔴 **Documented files are missing** (Section 0): ADR.md, the TF-IDF baseline, the modular `config/` / `src/` layout and both READMEs. The design rationale the prompt says to "read first" is unavailable. Later phases that assume these paths will fail.
2. 🔴 **The benchmark cannot measure quality.**
   - No eval labels exist anywhere in the repo.
   - `compute_recall_at_k` (`embedding_model_benchmark.py:86-116`) has a docstring promising "precision/recall/F1 @k" but returns only `recall_at_k` and `hit_rate_at_k`. Precision and F1 are not implemented.
   - Even with labels, scoring a random 30-student × 500-job sample means most labelled jobs (500 / 4,990 ≈ 10% inclusion chance) are absent from the candidate pool. That artificially caps recall.
3. 🟠 **Cross-model scores are not comparable, and E5's high mean is mostly an artifact of empty text.**
   - 21 of the 30 sampled students have **empty** projects and certifications sections, which carry 25% of the weight.
   - An empty string embedded and compared to a real passage scores: **SBERT 0.073, BGE 0.478, E5 0.762** (measured).
   - Combined with E5/BGE's compressed, prefix-dependent score bands (E5 std 0.0129 vs SBERT 0.0592), this confirms the ADR's comparability caveat and shows it is worse than stated. Empty sections actively inflate E5 scores.
4. 🟠 **The job side of two sections uses the same text.** Both `projects` and `certifications` map to `job_description` (`embedding_model_benchmark.py:57-58`), so 25% of the weight is two comparisons against one text. The job data has no certification field at all.
5. 🟠 **Latency and memory numbers are unreliable.**
   - Each figure comes from a single, un-warmed call, and model order affects it: BGE measured 12.7 ms when last and 34.5 ms when first, and SBERT's committed 1611 ms is a first-call artifact.
   - RSS memory deltas go negative (−148.2, −75.5, −5.1 MB).
   - The device (MPS vs CPU) is not recorded.
   - Committed `load_seconds` of 18–28 s vs about 4 s today indicates a cold cache or download.
6. 🟠 **`data_quality_report.md` undercounts missing values.** `missing()` (`preprocess.py:217`) does `astype(str).eq("")`, so NaN from the Excel export becomes `"nan"` and is counted as present. The placement-cell blanks are hidden, despite the claim "every number here is measured". Actual blank counts in `students_unified.csv`:

   | field | report says | actual | hidden placement-cell blanks |
   |---|---|---|---|
   | achievements | 166 | **216** | 50 |
   | internship_raw | 0 | **57** | 57 |
   | certifications | 166 | **173** | 7 |
   | skills_raw | 33 | **35** | 2 |
   | technical_projects | 166 | **167** | 1 |
7. 🟠 **Job data provenance.**
   - (a) The raw file has exactly **1,048,575 rows = 2²⁰ − 1**. That is Excel's per-sheet row limit (plus a header), so the file very likely passed through Excel and was **truncated**. Verify against the original Kaggle download.
   - (b) Its distributions are near-perfectly uniform: 216 countries with 22–24 sampled jobs each (including North Korea and San Marino), `work_type` 5 × ~1,000, `qualification` 10 × ~450–530, `gender_preference` ⅓ / ⅓ / ⅓, `job_portal` 16 × ~400. That is a fingerprint of **programmatically generated data**, not real postings. Check the dataset's Kaggle description before calling it "real" in any report.
   - (c) Only **23 India jobs**, against a student cohort that is India-based. Salaries are in USD.
   - (d) The `skills` field is space-joined phrases with no delimiter (`"Tax law knowledge Individual tax planning Tax return preparation…"`), so it cannot be reliably split into a skill list.
   - (e) 18 duplicate `source_job_id`s (documented float-precision issue, handled by re-IDing).
8. 🟠 **Resume parsing quality.**
   - Skills-section miss rate measured: **19.9% (33/166)** on deduped rows and 16.6% on all 962. The documented "~21%" is close but not reproduced exactly.
   - Degree missing for **65/166 (39.2%)**.
   - Extracted skills carry noise suffixes (`"Python- Exprience - Less than 1 year months"`), averaging 4.7 skills per resume.
   - Degree strings are not normalized: `B.E/ B.Tech`, `B.E`, `BE`, `B.TECH`, `BTECH`, `B.SC`/`BSC`, `M.SC`/`MSC` are all distinct values.
   - Placement-cell skills use a different format (multi-line, `"Programming Languages: C, C++…"`).
   - There is no skill normalization or skills master anywhere.
9. 🟠 **Cohort imbalance.**
   - All 120 placement-cell students are CSE (Data Science), graduating in 2027, in India. Only they have CGPA, gender and graduation year.
   - The 166 Kaggle rows have none of those fields. Civil, Mechanical and Electrical have 5–6 students each.
   - Cohort analytics (percentiles, K-means) is only meaningful on the 120-row subset.
10. 🟡 **Hygiene.**
    - Input paths are hard-coded and CWD-relative (`preprocess.py:12-14`), and the raw data is unversioned and outside the repo.
    - No dependency manifest.
    - System Python 3.9.6 is end-of-life (Oct 2025) and linked against LibreSSL.
    - `JOB_SAMPLE_N=5000` yields 4,990, because per-country rounding is never topped up.
    - `start_id=200` for Kaggle IDs (`preprocess.py:154`) would collide if the placement export ever exceeds 199 rows.
    - The benchmark docstring says "Phase 2" while the prompt files this work under Phase 6.
    - `.DS_Store` files are committed.
11. 🟡 **Spec vs diagram conflicts to resolve.** `Architecture.png` names SBERT and XGBoost as *the* models, which conflicts with the non-negotiable that model choice is decided experimentally. It also says the shared LLM key is "e.g., Gemini", while the Phase 1 prompt says an Anthropic key is needed from Phase 3.
12. 🟡 **Data sensitivity.** `students_unified.csv` holds 120 real students' CGPA, gender, internships and achievements. These are quasi-identifiers, and the file is pushed to `github.com/SaarthakKhamkar11/Student-Employabilty`. I could not check the repo's visibility (`gh` is not installed). An NDA PDF sits in the project folder. Confirm the repo is private and that consent covers this. The job data also has a `gender_preference` column, which should never be a matching feature.

## J. What can be reused (the foundation, built on top of, not around)

- **`preprocess.py` as a whole:** the schemas, `CATEGORY_MAP` (25 categories → 6 branches plus domains), the ingestion functions, the resume section slicer and the stratified job sampler. Its output is deterministic and verified byte-identical, so it can serve as a regression fixture.
- **`data/processed/*.csv`:** 286 students and 4,990 jobs, with SHA-256s recorded above. These form the seed data for the database load.
- **`embedding_model_benchmark.py`:**
  - `MODEL_CONFIGS` (including E5/BGE prefixes)
  - `student_section_texts` / `job_section_texts`
  - the section-wise weighted-cosine structure and the Recall@K scaffold
  - the "null, not guessed" handling of missing labels

  It needs the targeted fixes in K, not a rewrite.
- **Local HF cache** of all three base models (about 1.25 GB). No re-download is needed.
- **Outside the repo, a candidate (weak) label source:** `~/Desktop/Major Projec/Dataset/job_resume_fit.csv` (about 30 MB) has `resume_text`, `job_text`, `ai_match_score`, `skill_string_match_score` and `fuzzy_match_score`. Caveats: its scores are model-generated, not human ground truth, and its IDs do not join to our `student_id` / `job_id`. It could pretrain or sanity-check, but not certify.
- **`Architecture.png`** is the most complete design artifact available (Section L).

## K. What must be changed

Nothing needs to be rewritten. New capability must be **added**: database, agents, API, frontend and tests. Before Phase 6 and 8 results can be trusted, these **targeted, minimal fixes** are needed. All are proposed here and none is applied:

1. **Recover or recreate** `docs/ADR.md` and the TF-IDF baseline. Add a `README.md` and a pinned dependency manifest.
2. **Decide on layout:** keep `preprocess.py` as is, or split it into the `config/schema.py` + `src/pipeline/*` + `src/parsing/*` layout the prompt assumes. A split would only move code, with no logic changes, and needs your decision.
3. **Eval labels:** create a labelled student–job set, which Phases 6 and 8 cannot proceed without. Score each labelled student against the full job pool, not a 500-job sample, and add the missing precision@K, F1, nDCG and MRR.
4. **Benchmark methodology:**
   - skip or zero-weight empty sections and renormalize the remaining weights
   - stop double-counting `job_description`
   - compare models on rank-based metrics, not raw cosine
   - add warm-up, repeated trials (p50/p95) and fixed or recorded device and order
5. **Quality report:** count NaN as missing (one-line fix at `preprocess.py:217`) and regenerate.
6. **Normalization:** degree strings, skill-string noise, a skills master, and splitting the delimiter-less job `skills` field.
7. **Paths:** make input paths configurable (CLI or env) instead of CWD-relative.

## L. Proposed agent architecture (my understanding, from `Architecture.png`)

> Caveat: the master spec's Sections 2 and 5 are not in this session. This is based on `~/Desktop/Major Projec/Architecture.png` ("6 agents" = Supervisor + 5). Please confirm it matches the spec.

**Users.**
- Students upload a resume or profile, set target roles, and see matches with scores, skill gaps with courses, their cohort benchmark, and explanations.
- Companies post jobs, view matched candidates with suitability scores, and shortlist.

The app layer is Next.js plus FastAPI. Both call the Supervisor.

1. **Supervisor Agent.** The only agent that talks to the user layer. It classifies intent ("find me jobs", "why didn't I match X", "candidates for this posting"), breaks the request into sub-tasks, routes them to the right specialists, and merges their outputs into one response. It holds no domain logic itself.
2. **Profile Agent.** Turns a raw resume or form into a structured student profile (education, skills, projects, certifications, experience), normalizes skills against a skills master, and produces section embeddings. It grows directly out of `parse_resume` / `ingest_*`. An LLM fallback for extraction is the natural fix for the ~20% skills / 39% degree misses.
3. **Job Intelligence Agent.** Same idea for jobs: ingest from APIs or the DB, extract requirements, normalize skills, dedupe, and embed. It grows out of `ingest_jobs`.
4. **Matching Agent (core ML).** Computes section-wise semantic similarity, engineers features (skill overlap, experience, education, projects, certifications, location), and predicts a suitability score with a supervised model. It also outputs skill gaps and the feature vector. The diagram names SBERT and XGBoost, but per the non-negotiables both are placeholders until Phases 6 and 8 pick winners experimentally.
5. **Learning Agent.** Takes skill gaps, searches a course DB or APIs, ranks courses by course–skill similarity, and labels them free or paid. **No course data exists yet.**
6. **Explanation Agent.** Turns model internals (SHAP feature impacts, positive and negative) plus gaps and courses into human-readable reasons ("why this job", "why this gap").

**Shared tool layer, not agents:**
- NLP and skill extraction, embeddings and similarity, the prediction model, SHAP
- **Cohort analytics:** filtering by department, target role and graduation year; K-means with the elbow method; percentiles, ahead/behind and skill coverage
- **Search and integration tools:** job and course search APIs, pgvector search, SQL query engine, Redis cache, feedback processor

**Data layer:**
- PostgreSQL tables: students, jobs, skills master, courses, applications, outcomes and feedback, cohorts and clusters, embeddings, model registry
- pgvector, Redis, and object storage for resumes

**Feedback loop:** recommendation → application → shortlist / interview → offer or reject → outcome data → evaluation → retrain → model registry update, with adaptive re-weighting.

**LLM usage:** one shared API key, and only agents that need reasoning or generation call it (Supervisor and Explanation for sure, and possibly a Profile fallback). Deterministic steps (parsing, embedding, scoring, SHAP) stay plain Python tools, so results stay reproducible and testable.

## M. Proposed model experimentation architecture (Phases 6 and 8)

> Caveat: master spec Sections 9 and 13 (the two Model Selection Engines) are not in this session. This is my understanding from the Phase 1 prompt and must be checked against the spec.

**Phase 6: representation and retrieval models.**
- **Candidates:**
  - TF-IDF (recovered or re-implemented; the floor every model must beat)
  - SBERT `all-mpnet-base-v2`
  - E5 `e5-base-v2`
  - BGE `bge-base-en-v1.5`
  - **a small local model** (exact model is an open decision, e.g. a MiniLM or `bge-small`-class encoder), for the cost/quality trade-off
- **Prerequisites:** eval labels plus the benchmark fixes in K.4.
- **Metrics:**
  - Quality: Recall@K, Precision@K, nDCG@K, MRR, against the full job pool.
  - Cost: warm p50/p95 latency, throughput, peak memory, model size, deployment complexity.
- **Comparisons:** use ranks or per-model normalized scores, never raw cosine, because of the E5/BGE score-distribution problem.
- **Model Selection Engine #1 (embeddings):** a scripted, reproducible ranker. It takes the per-model metrics table, applies declared constraints (e.g. a p95 latency budget) and declared criterion weights, and emits a decision record plus the selected model ID into config or the model registry. Nothing is hard-coded, and rerunning the engine on new results can change the winner.

**Phase 8: matching / suitability models.**
- **Inputs:** section similarities from the Phase 6 winner, plus structured features (skill overlap and gap counts, experience fit, education / qualification match, certifications, location). Gender is excluded.
- **Candidates:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (sklearn), XGBoost, and optionally LightGBM.
- **Labels:** the same eval set, ideally extended with real placement outcomes. Weak labels (e.g. `job_resume_fit.csv`) only with explicit caveats.
- **Protocol:**
  - cross-validation grouped by student, to avoid leakage
  - metrics: ROC-AUC, PR-AUC, F1, calibration (Brier), ranking metrics, inference latency
  - SHAP compatibility, which the Explanation Agent needs
- **Model Selection Engine #2 (matching):** same pattern as #1. It writes the chosen model and its metrics to the model registry, which the feedback loop later re-evaluates and retrains against.

## N. Proposed implementation plan (Phases 2–18)

> Anchors taken from the Phase 1 prompt: Phase 2 builds the agent architecture, Phase 3 needs the Anthropic key, Phase 6 is embedding experiments, Phase 8 is matching-model comparison. All other phase numbers and contents are my proposal and must be reconciled with the master spec.

| Phase | Proposed scope | Depends on |
|---|---|---|
| 2 | Foundations plus agent skeleton: Python 3.11 env with pinned deps, `.gitignore`, configurable paths, ADR.md restored or recreated, Supervisor + 5 agent interfaces with deterministic stub tools wrapping the existing `preprocess.py` functions | Phase 1 audit; `preprocess.py` |
| 3 | LLM integration: Supervisor intent routing and aggregation, Explanation narration, Profile extraction fallback; key via env var only | 2; API key |
| 4 | PostgreSQL + pgvector schema derived from `STUDENT_SCHEMA` / `JOB_SCHEMA` (plus skills master, courses, applications, outcomes, cohorts, embeddings, model registry); loaders from `data/processed/*.csv` | 2 |
| 5 | Data quality and labels: NaN-count fix, degree and skill normalization, skills master, job-skill splitting, labelled student–job eval set | 1 (problems I-2, I-6–I-8), 4 |
| 6 | Embedding experiments: TF-IDF baseline + SBERT / E5 / BGE + small local model with fixed methodology; Model Selection Engine #1 | 5 (labels), existing benchmark |
| 7 | Matching features and skill-gap computation using the Phase 6 winner | 5, 6 |
| 8 | Matching-model comparison (LogReg / DT / RF / GBM / XGBoost / optionally LightGBM); Model Selection Engine #2 | 7 |
| 9 | Explainability: SHAP for the selected model, wired into the Explanation Agent | 3, 8 |
| 10 | Learning Agent: course data source, course–skill similarity, ranking, free/paid | 4, 6, 7 |
| 11 | Cohort analytics: filters, K-means / elbow, percentiles, ahead/behind, skill coverage | 4, 6 |
| 12 | FastAPI backend: student and company endpoints over the Supervisor | 3, 4, 8–11 |
| 13 | Next.js frontend: student and company portals | 12 |
| 14 | Performance: Redis caching, pgvector indexes, latency budgets | 4, 12 |
| 15 | Feedback loop and model registry: outcome capture, evaluation, retraining, adaptive re-weighting | 4, 8, 12 |
| 16 | QA consolidation: unit, integration and e2e coverage plus CI, including the byte-identical `preprocess.py` regression and model-metric regressions. Tests are added incrementally from Phase 2. | 2–15 |
| 17 | MLOps and deployment: Docker Compose (Postgres + pgvector, Redis, API, web) with pinned images; CI/CD | 12–16 |
| 18 | Final evaluation, fairness audit (gender, branch imbalance), documentation and report | all |

## O. Risks and dependencies

1. **Master spec and ADR are not available in this session.** Sections L, M and N are reconstructions. *Mitigation:* paste or commit the spec and ADR before Phase 2.
2. **No-internet sandbox history.**
   - SBERT, E5 and BGE have now **actually been executed** on this machine (this session, and evidently on Sep 25).
   - **XGBoost, LightGBM, SHAP, FastAPI, SQLAlchemy/psycopg, pgvector, Redis and the Anthropic SDK are not installed and have never been executed here.**
   - The TF-IDF baseline has no code present.
   - All of these remain **Not yet evaluated** and must not be assumed to work.
3. **No eval labels.** This is the single biggest blocker. Phase 6 quality metrics and all of Phase 8 depend on it. Without labels, any "best model" claim would be fabricated.
4. **Job data is likely synthetic and Excel-truncated (I-7).** Matching results may not transfer to real Indian campus hiring, since only 23 India jobs are in the sample.
5. **Versions must be pinned.**
   - Python: 3.9.6 is end-of-life; 3.11.17 is available via Homebrew or uv.
   - Docker 29.8.2 is installed. Postgres, pgvector and Redis are not installed locally, so they will need pinned Docker images (exact tags to be chosen in Phase 2 or 4).
   - Node v26.0.0 must be pinned for Next.js compatibility.
   - Python deps need a lockfile.
6. **No API key is configured.** `ANTHROPIC_API_KEY` is not set in the environment, and there is no `.env` or key reference anywhere in the repo. It is required from Phase 3. The diagram's "e.g., Gemini" conflicts with this, so the provider must be decided.
7. **Data sensitivity (I-12).** Real student records are in a GitHub-pushed repo. Confirm visibility and consent before adding more.
8. **Measurement noise on Apple Silicon (I-5).** MPS vs CPU and warm-up effects will dominate latency comparisons unless they are controlled.
9. **Small, imbalanced cohort (I-9).** 286 students, 120 with full fields, and 5–6 per core-engineering branch. This limits Phase 8 model capacity and cohort-analytics validity.

---

## Appendix — full `pip freeze` (`/usr/bin/python3`, Python 3.9.6)

```
altgraph @ file:///AppleInternal/.../altgraph-0.17.2-py2.py3-none-any.whl
appnope==0.1.4
asttokens==3.0.1
certifi==2026.7.22
charset-normalizer==3.5.1
comm==0.2.3
debugpy==1.8.20
decorator==5.2.1
et_xmlfile==2.0.0
exceptiongroup==1.3.1
executing==2.2.1
filelock==3.19.1
fsspec==2025.10.0
future @ file:///AppleInternal/.../future-0.18.2-py3-none-any.whl
hf-xet==1.6.0
huggingface_hub==0.36.2
idna==3.20
importlib_metadata==8.7.1
ipykernel==6.31.0
ipython==8.18.1
jedi==0.19.2
Jinja2==3.1.6
joblib==1.5.3
jupyter_client==8.6.3
jupyter_core==5.8.1
macholib @ file:///AppleInternal/.../macholib-1.15.2-py2.py3-none-any.whl
MarkupSafe==3.0.3
matplotlib-inline==0.2.1
mpmath==1.3.0
nest-asyncio==1.6.0
networkx==3.2.1
numpy==2.0.2
openpyxl==3.1.5
packaging==26.1
pandas==2.3.3
parso==0.8.6
pexpect==4.9.0
pillow==11.3.0
platformdirs==4.4.0
prompt_toolkit==3.0.52
psutil==7.2.2
ptyprocess==0.7.0
pure_eval==0.2.3
Pygments==2.20.0
python-dateutil==2.9.0.post0
pytz==2026.4
PyYAML==6.0.3
pyzmq==27.1.0
regex==2026.1.15
requests==2.32.5
safetensors==0.7.0
scikit-learn==1.6.1
scipy==1.13.1
sentence-transformers==5.1.2
six @ file:///AppleInternal/.../six-1.15.0-py2.py3-none-any.whl
stack-data==0.6.3
sympy==1.14.0
threadpoolctl==3.7.0
tokenizers==0.22.2
torch==2.8.0
tornado==6.5.5
tqdm==4.70.1
traitlets==5.14.3
transformers==4.57.6
typing_extensions==4.15.0
tzdata==2026.4
urllib3==2.6.3
wcwidth==0.6.0
zipp==3.23.1
```
