# PROGRESS.md — Project hand-off log

**Rule:** update this file at the end of every phase. That means the status table, the "Pick up here" block, and a new entry in the Phase log. **From Phase 6 on, also extend `scripts/demo.py`** with a live demo of the phase's new capability, and run it once against the backend, and refresh the owner's progress dashboard (a private claude.ai artifact: aggregates only, real measured numbers, with the same caveats as the reports). Anyone taking over should be able to read only this file and know exactly where to start.

Last updated: **2026-10-11** · Last completed phase: **Phase 9 (Skill-gap analysis)** · Next phase: **Phase 10 (Learning Agent)**. The team's labelling (below) is still open and now finishes TWO choices: the embedding model (ADR 0012) and the matching model (ADR 0015).

**Latest code: `main`** holds Phases 1–8 (Phase 8 committed directly on `main` as `7e680b6`). Phase 9 is also committed. Phases 6–7 were committed on `phase-6-7-embeddings-matching` and fast-forwarded into `main` on 2026-10-10. The per-phase branches `phase-1-audit`, `phase-2-3-supervisor`, `phase-4-5-profile-job-agents` and `phase-6-7-embeddings-matching` are kept for history.

Source of truth for scope: [`docs/MASTER_SPEC.md`](docs/MASTER_SPEC.md). Phase 1 audit: [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md).

---

## ▶ Pick up here: Phase 10 (Learning Agent), plus one team task that finishes Phases 6 AND 8

**Team task (still open).** Grade the relevance pairs on the labelling page:
1. Start the stack (`docker compose up -d --wait`) and open **http://127.0.0.1:8000/labelling**. Enter your initials, or open `/labelling?labeller=AB`.
2. There are 30 students × 12 jobs = **360 pairs**, about 2 hours in total if split between people. The rubric is on the page. Grades save as you click, and you can come back later.
   - 4 students are deliberately given to a second person, to measure agreement.
   - Grade the fit between the student and the job. The page hides salary, company, location, gender and CGPA, plus the posting's randomly assigned degree and experience fields.
3. When everyone is done:
   ```bash
   .venv/bin/python scripts/export_labels.py      # -> data/benchmark/human_labels.csv + .manifest.json
   export EXPERIMENTS_DATABASE_URL=postgresql://<user>:<password>@127.0.0.1:55433/<db>
   # Phase 6: the embedding model
   .venv/bin/python -m src.embeddings.model_evaluation_engine --labels data/benchmark/human_labels.csv --reuse-cost --record-db
   # Phase 8: the matching model. It evaluates the SAME trained artifacts (no retraining), applies the ADR 0015
   # rule and, only on a "select" decision, writes the winner into config/matching_model.yaml
   python -m src.scoring.matching_model_selection_engine --labels data/benchmark/human_labels.csv \
       --reuse-models mms-20261011T104041Z-nolabels --record-db --activate
   ```
4. Copy each report's **Decision** line into its ADR ([0012](docs/adr/0012-embedding-model-selection.md) embeddings,
   [0015](docs/adr/0015-matching-model-target-training-and-selection.md) matching). The rules were fixed in advance,
   so don't change the weights after seeing the results.
   - If ADR 0012 selects a new embedding model, the features change: re-embed, rebuild the calibration
     (`python -m app.matching.calibration`) and the feature matrix (`python -m app.matching.training_data`), then
     retrain with the Phase 8 engine (no `--reuse-models`) and select again. A trained matching model refuses to
     run on other embeddings.
   - Rebuild the backend image after `--activate` (the image copies `config/` and `models/`).

**Phase 10 needs:**
1. The Phase 10 prompt.
2. **The input contract is fixed:** `SkillGapReport` (`app/schemas/agent_io.py`, ADR
   [0016](docs/adr/0016-skill-gap-analysis.md)). It holds:
   - `matched_skills`;
   - `missing_required` and `missing_preferred`: lists of `SkillGap` (`skill`, `importance_score`,
     `transferable_candidate`, `transferable_similarity`), ordered by importance;
   - `partially_related`;
   - provenance: `importance_model`, `similarity_model`, `thresholds`, `method`, `note`.

   The Supervisor already passes it as `LearningAgentInput.skill_gap_report`, plus `skill_gaps`: the names, most
   important first. Don't change the shape without updating ADR 0016.
3. **What the report can and can't say:**
   - **Importance ties within a requirement class.** `features-v1` sees skills as counts, so every model scores
     every required gap of a job the same; ties are broken alphabetically. The Learning Agent must not present a tie
     as a ranking.
   - **A missing skill is "not detected"**, never "lacking" (keep the report's `note`).
   - **Transferable candidates are lexical with TF-IDF** (ADR 0016's evidence).
4. **Spec §17:** learning resources must come from a course catalogue, never generated. No catalogue exists in the
   repo yet; that is Phase 10's first data question.
5. **Owner decisions pending:**
   - the hard degree/experience filters (ADR 0013: all 18 diploma students get no eligible job on this dataset);
   - the gender visibility filter (ADR 0009);
   - whether to make the TF-IDF version hash platform-stable (ADR 0015).

Decisions already made (do not re-ask):
- **LLM provider: Gemini.** Wherever the spec or a prompt says "Claude / Anthropic", use Gemini (`GEMINI_API_KEY`).
- **Code layout:** the pipeline was split into `config/` + `src/` in Phase 2. `preprocess.py` is now a thin entry point.
- **Interim embeddings:** the TF-IDF baseline until Phase 6 selects a model ([ADR 0004](docs/adr/0004-interim-tfidf-embeddings.md); supersedes 0002).
- **Missing values:** sentinels only ([ADR 0007](docs/adr/0007-missing-value-sentinels.md)).
- **Employer gender preference:** a visibility filter only, never a feature ([ADR 0009](docs/adr/0009-gender-preference-visibility-filter.md)). A concern and recommendation are recorded there for the owner to review.
- **Required/preferred job skills:** INFERRED by heuristic v2, never ground truth ([ADR 0008](docs/adr/0008-job-skill-phrases-and-required-preferred.md)).

Open items owned by the project owner:
- **The GitHub repo is public** (checked 2026-10-08) and contains 120 real students' records in `data/processed/students_unified.csv`. Switch it to private.
- **Relevance labels**: the team grades 360 pairs on `/labelling` (see Pick up here). This finalises the embedding choice (ADR 0012) and the matching-model choice (ADR 0015).
- **Gender visibility filter**: decide whether to keep it on (ADR 0009 recommends off).
- 🔴 **Real PII in the committed processed data.** `data/processed/students_unified.csv` (public since the first commit, produced by the original preprocessing, which never masked PII) holds 3 Kaggle rows whose experience text contains an email, a phone number, a LinkedIn URL, an address and at least one personal name. Options: make the repo private; mask PII inside `src/pipeline` and regenerate the data (changes the data hash); and, if needed, rewrite git history. History rewrite is destructive and is the owner's call.
- The original `docs/ADR.md` was never found. ADRs now live in `docs/adr/`. The TF-IDF baseline was **re-implemented** in Phase 4.

---

## Phase status (spec §43)

| Phase | Scope | Status |
|---|---|---|
| 1 | Repository audit | ✅ Done (2026-10-08) |
| 2 | Agent architecture | ✅ Done (2026-10-08) |
| 3 | Supervisor Agent | ✅ Done (2026-10-08). LLM path verified with a fake client only; no real Gemini call yet |
| 4 | Profile Agent | ✅ Done (2026-10-08) |
| 5 | Job Intelligence Agent | ✅ Done (2026-10-08) |
| 6 | Embedding experimentation framework | ✅ Done (2026-10-09): framework, labelling page, cost and weak-label results. The model choice waits for the team's human labels (ADR 0012) |
| 7 | Matching Agent | ✅ Done (2026-10-09): Stages 1-6 as separate tools, interim weighted baseline (placeholder weights pending Phase 8) |
| 8 | Matching model experimentation | ✅ Done (2026-10-11): 6 candidates trained, measured and wired behind a config flag; quality "Not yet evaluated" and Baseline 3 stays active until the human labels (ADR 0015) |
| 9 | Skill-gap analysis | ✅ Done (2026-10-11): `src/scoring/skill_gap_analysis.py`, counterfactual importance, tuned relation tiers, the `SkillGapReport` contract (ADR 0016) |
| 10 | Learning Agent | ⏭ Next |
| 11 | Explainability Agent + SHAP | Not started |
| 12 | Agent state/tracing | Not started |
| 13 | Research dashboard | Not started |
| 14 | Evaluation framework | Not started |
| 15 | Feedback loop | Not started |
| 16 | Security hardening | Not started |
| 17 | Testing | Not started |
| 18 | Documentation | Not started |

**Gaps in the spec's phase plan to resolve later:**
- The database tables in spec §23 are being added by the phases that need them (student tables: Phase 4 groundwork; job tables: Phase 5).
- No phase owns the student and company dashboards in spec §30. Only the research dashboard has a phase (13).

---

## Repository layout (after Phase 9)

```
preprocess.py                 thin entry point → src/pipeline/build_unified_dataset.py
config/schema.py              JOB_SCHEMA / STUDENT_SCHEMA (single source of truth)
config/model_selection_weights.yaml   Phase 6 selection weights: 3 scenarios, a reason per weight, gate, sensitivity
config/weak_label_rules.yaml  Phase 6 weak-label rules (role families x student domains; Phase 7 reuses the role families)
config/matching_baseline_v1.yaml  Phase 7 matching config: top_n, hard rules, ranking cap, feature settings, weights
                              (each with a reason), employability dimensions, branch rules, registration
config/matching_model.yaml    Phase 8 THE FLAG: which model Stage 4 uses (active), whose artifacts, which selection run
config/matching_model_candidates.yaml  Phase 8 candidates: fixed hyperparameters (a reason each), training settings, rubrics
config/matching_model_selection_weights.yaml  Phase 8 selection weights: 3 scenarios, gate, sensitivity, "close enough"
config/skill_gap.yaml         Phase 9 relation-tier thresholds per embedding model family, with the tuning evidence
src/parsing/                  section_parser.py (+ parse_job_sections, P5), resume_text_parser.py
src/pipeline/                 ingest_placement_cell.py, ingest_kaggle_resumes.py, ingest_real_jobs.py,
                              validate_and_report.py, build_unified_dataset.py
src/embeddings/               embedding_model_benchmark.py (5 models: texts, scoring, isolated cost measurement),
                              model_evaluation_engine.py (Phase 6 table, weights, sensitivity, best-model lines),
                              tfidf_baseline.py (re-implemented in Phase 4; corpus texts frozen in Phase 6)
src/evaluation/               splits.py (fixed per-student splits), relevance_labels.py (loader + hard rules),
                              retrieval_metrics.py (P/R/MRR/NDCG/Hit@K, bootstrap CIs), labelling_pool.py, weak_labels.py
src/scoring/                  Phase 8: matching_models.py (build/train/save/load/score/SHAP per candidate; isolated
                              measurement worker), matching_model_selection_engine.py (the Phase 8 table, scenarios,
                              sensitivity, simpler-model check, ADR 0015 decision, --activate, --record-db),
                              skill_gap_analysis.py (P9: the SkillGapReport; Stage 6 wraps it; CLI per pair)
backend/app/agents/           base.py, supervisor.py + routing.py (P3), profile_agent.py (P4), job_intelligence_agent.py (P5), 3 shells
backend/app/schemas/          agent_state.py, agent_io.py, student.py, job.py
backend/app/tools/            registry.py, profile_tools.py (11 tools), job_tools.py (12 tools, incl. check_job_visibility),
                              matching_tools.py (10 tools: Stages 1-6 + evaluate_model + benchmark_matching)
backend/app/profile/          text_extraction.py, pii.py, sections.py (wraps src/parsing), skills.py,
                              data/skill_aliases.csv (+ build_skill_aliases.py that regenerates it with evidence counts)
backend/app/jobs/             skills_split.py, requirements.py (inferred required/preferred), dedup.py, visibility.py
backend/app/matching/         features.py (spec §11 contract + protected-attribute guard), config.py, entities.py (views),
                              eligibility.py (Stage 1), similarity.py + vectors.py (Stage 2, signals), calibration.py,
                              skill_gap.py, feature_builder.py (features-v1), baseline.py (weighted_baseline_v1),
                              pipeline.py (all stages), evaluate.py (evaluate + benchmark + register Baseline 3),
                              predictor.py (P8: Stage 4 model from the config flag), training_data.py (P8 feature matrix)
models/matching/<experiment>/ Phase 8 trained artifacts (joblib / XGBoost JSON / LightGBM text) + manifest.json (SHA-256 of
                              each, checked before loading) + measurements.json
backend/app/embeddings/       provider.py (configured embedding model; interim TF-IDF, sparse vectors)
backend/app/core/             config.py (.env loading), logging.py (JSON, PII-redacting), sanitize.py (prompt-injection defence)
backend/app/llm/client.py     Gemini client (structured JSON output, validated locally)
backend/app/db/               migrate.py, store.py, seed.py (students), seed_jobs.py (jobs + dedup + --process),
                              profile_repository.py, job_repository.py, matching_repository.py
backend/app/labelling/        Phase 6 labelling page (page.html), API (routes.py), logic (service.py), storage, load_pool.py
backend/app/experiments.py    GET /experiments, GET /experiments/{id} (recorded model-selection runs)
backend/migrations/           0001 agent_runs · 0002 student tables · 0003 embeddings + model_versions · 0004 profile columns ·
                              0005/0006 student_skills unique-key fix · 0007 jobs tables + sparse embeddings ·
                              0008 relevance_pool, relevance_judgments, experiment_runs, experiment_metrics
docs/adr/                     architecture decision records 0001–0016 (index: docs/adr/README.md)
backend/app/main.py           FastAPI: GET /health, GET /agents, POST /requests, POST /requests/resume-file, GET /requests/{id},
                              /labelling (page + API), /experiments
backend/tests/                327 tests (310 run by default; 17 skipped: 14 need TEST_DATABASE_URL, 1 live Gemini, 2 others)
backend/db/init/              01_enable_pgvector.sql (runs on first postgres start)
docker-compose.yml            postgres+pgvector, redis, backend
.env.example                  template; copy to .env (gitignored)
docs/MASTER_SPEC.md           the project spec
data/processed/               286 students, 4,990 jobs (committed)
data/benchmark/               student_splits.csv, labelling_pool*.csv/.json, weak_labels.csv/.json, human_labels.csv/.json (once exported),
                              matching_calibration/<model>-<version>.json (Phase 7),
                              matching_features/<feature_version>-<embedding_version>.csv.gz + .manifest.json (Phase 8)
scripts/                      demo.py (live demo, extended every phase), export_labels.py (labelling page -> data/benchmark)
data/raw/                     raw sources (gitignored; see setup)
results/                      data-quality report; embedding_evaluation/<experiment_id>.{md,json} (every engine run),
                              embedding_model_evaluation[_<label source>].{md,json} (latest), embedding_cost_measurements.json,
                              matching_evaluation/<experiment_id>.json (Phase 7 registration runs),
                              matching_model_evaluation/<experiment_id>.{md,json} (Phase 8 engine runs),
                              matching_model_evaluation[_<label source>].{md,json} (latest)
.gitattributes                data/** and models/** are never line-ending converted (their bytes are hashed)
```

---

## Setup from scratch (verified on 2026-10-08, Apple M2 / macOS, Docker 29.8.2)

```bash
# 1. Python 3.11 virtual environment
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt -r backend/requirements-dev.txt
# VS Code picks up .venv via .vscode/settings.json. If not: Cmd+Shift+P → "Python: Select Interpreter" → .venv

# 2. Raw data → data/raw/ (gitignored; never commit these)
#    Student Data.xlsx          placement-cell export, real student data (get it from the project owner)
#    UpdatedResumeDataSet.csv   Kaggle resume dataset, 962 rows
#    job_descriptions.csv       Kaggle job dataset, 695 MB, 1,048,575 rows
#    Use the team's exact copies: the job file is a truncated version, and a fresh
#    download will NOT reproduce the processed data. Or set RAW_DATA_DIR=/path/to/folder.

# 3. Data pipeline (about 6 s)
.venv/bin/python preprocess.py

# 4. Embedding cost benchmark: all 5 models, each in its own offline process (about 15 min on an M2;
#    downloads about 1.35 GB of models on first run). The evaluation engine (Phase 6 section below) runs this too.
.venv/bin/python -m src.embeddings.embedding_model_benchmark

# 5. Tests (about 35 s)
.venv/bin/python -m pytest
#    Optional: include the 14 real-database tests (start postgres first, step 6):
#    TEST_DATABASE_URL=postgresql://<user>:<password>@127.0.0.1:55433/<db> .venv/bin/python -m pytest

# 6. Docker stack. The backend applies SQL migrations on start.
cp .env.example .env    # then set POSTGRES_PASSWORD, the same password inside DATABASE_URL, and JWT_SECRET
docker compose up --build -d --wait
docker compose exec backend python -m app.db.seed    # loads the 286 students (idempotent)
docker compose exec backend python -m app.db.seed_jobs --process   # 4,990 jobs + dedup + full pipeline (~50 s)
curl http://127.0.0.1:8000/health    # expect postgres and redis "reachable"
curl -X POST http://127.0.0.1:8000/requests -H 'Content-Type: application/json' \
  -d '{"actor":"student","request_kind":"resume_upload","resume_text":"Skills: Python, SQL"}'
curl http://127.0.0.1:8000/requests/<request_id from the response>
docker compose exec backend python -m app.labelling.load_pool   # Phase 6: the labelling pool (360 pairs)
docker compose exec backend python -m app.db.seed --process     # Phase 7: profiles + embeddings for all 286 students (~25 s)
docker compose down                  # add -v to also delete the database volume
```

**Phase 6 evaluation** (host-side, `.venv`; the regenerated files are committed):
```bash
.venv/bin/python -m src.evaluation.splits            # data/benchmark/student_splits.csv (fixed per student)
.venv/bin/python -m src.evaluation.labelling_pool --device mps   # the pool (already built: pool-5c6a5972b32d)
.venv/bin/python -m src.evaluation.weak_labels       # data/benchmark/weak_labels.csv (rule-based)
.venv/bin/python -m src.embeddings.model_evaluation_engine       # no labels: measures cost (~13 min), quality "Not yet evaluated"
.venv/bin/python -m src.embeddings.model_evaluation_engine --labels data/benchmark/weak_labels.csv --reuse-cost
#   add --record-db with EXPERIMENTS_DATABASE_URL=postgresql://<user>:<password>@127.0.0.1:55433/<db> to store the run
.venv/bin/python scripts/demo.py 9                   # labelling page status;  scripts/demo.py 10: recorded experiments
```

**Phase 7 matching** (host-side, after `seed --process`; `DATABASE_URL` pointing at 127.0.0.1):
```bash
DATABASE_URL=postgresql://<user>:<password>@127.0.0.1:55433/<db> PYTHONPATH=backend:. \
  .venv/bin/python -m app.matching.calibration       # similarity percentiles for the configured model (~20 s)
DATABASE_URL=... PYTHONPATH=backend:. .venv/bin/python -m app.matching.evaluate --labels data/benchmark/weak_labels.csv --register
#   records an experiment and writes its id into config/matching_baseline_v1.yaml; rebuild the backend image afterwards
.venv/bin/python scripts/demo.py 11                  # jobs for a student;  scripts/demo.py 12: candidates for a job
```

**Phase 8 matching models** (verified 2026-10-11 on Windows 11 + Docker Desktop/WSL2, inside the pinned
Python 3.11 backend image; host-side `.venv` works the same with `PYTHONPATH=backend:.`):
```bash
python -m app.matching.training_data                 # features-v1 for all 107,536 labelled pairs (~3 min; needs DATABASE_URL)
python -m src.scoring.matching_model_selection_engine --record-db    # trains 6 candidates, each in its own process (~2 min);
#   quality "Not yet evaluated" without labels; registers the artifacts in config/matching_model.yaml (active unchanged)
python -m src.scoring.matching_model_selection_engine --labels data/benchmark/weak_labels.csv \
    --reuse-models mms-20261011T104041Z-nolabels --record-db    # stamped weak-label diagnostic, never a selection
.venv/bin/python scripts/demo.py 13                  # recorded Phase 8 runs (GET /experiments?kind=matching_selection)
MATCHING_MODEL=decision_tree ...                     # force a trained model for one process (demo/tests only)
```

**Phase 9 skill gaps** (needs `DATABASE_URL`):
```bash
python -m src.scoring.skill_gap_analysis --student S00001 --job J0974289 [--model lightgbm]   # one pair's report
.venv/bin/python scripts/demo.py 14                  # the top match's report through the live API
```
The project's postgres is published on host port 55433, because a native PostgreSQL service often holds 5432 on
Windows. Inside Docker it is still `postgres:5432`.

**Reproducibility check.** After step 3, these SHA-256 hashes must match exactly. `students_unified.csv` changed in Phase 4 by the documented `.NET` parser fix ([ADR 0006](docs/adr/0006-parser-dotnet-fix.md)); before that it was `a123ed6f…51520`.
```
636d394f399c09e0c578a280b36e6294b333593418b80174161619d507c6b606  data/processed/students_unified.csv
42c7d78b48c130c8fd7c1ee282abdec108a08f33bd8e5817586291d87a42acb1  data/processed/jobs.csv
```

---

## Phase log

### Phase 9: Skill-gap analysis (2026-10-11) ✅

**What was built** (ADR [0016](docs/adr/0016-skill-gap-analysis.md)):

| Piece | Where | Notes |
|---|---|---|
| Module | `src/scoring/skill_gap_analysis.py` | Matched; missing required and preferred, ranked by importance; transferable; partially related. The method is in its docstring. CLI per pair |
| Importance | same file | **Counterfactual re-scoring:** suitability with the skill detected minus suitability as is, by the active matching model (`config/matching_model.yaml`) through `build_features`, batched per pair. Text-similarity features are held fixed. The model is named in every report |
| Normaliser | `app/profile/skills.py` | Reused (`normalize_skills`, `skill_key`); no second normaliser |
| Tiers | `config/skill_gap.yaml` | TF-IDF: partial 0.40, transferable 0.60, tuned on 12,022 real not-detected skills with a spot check. A model without an entry gets no tiers |
| Contract | `app/schemas/agent_io.py` | `SkillGap` (one skill), `PartiallyRelatedSkill`, `SkillGapReport` (for Phase 10). It replaces Phase 7's per-pair `SkillGap`. `MatchResult.skill_gap` and `LearningAgentInput.skill_gap_report` use it |
| Stage 6 | `pipeline.skill_gaps`, `identify_skill_gaps` | Wraps the module. The tool now also receives `compute_similarity`'s percentiles, which the counterfactuals hold fixed |
| Tests | `backend/tests/test_skill_gap_analysis.py` (13) | Normaliser reuse; importance equals an independent re-score through the real pipeline; importance follows the model (a doubled model doubles it); tier boundaries; no tiers without thresholds; ordering; the not-detected note; agent end to end |
| Demo | `scripts/demo.py 14` | The top match's report through the live API. Demo 11 now shows the top gaps by importance |

**Real reports** (CLI, active model `weighted_baseline_v1+545cea3bb767`, TF-IDF `tfidf-953a75d166d8`):

| Pair | Gap profile | Matched | Missing required (importance; transferable) | Missing preferred (importance; transferable) | Partially related |
|---|---|---|---|---|---|
| S00073 × J0002158 Data Scientist / ML Engineer | near-perfect | Machine learning algorithms, Data Cleaning & Preprocessing, Model Evaluation | none | Deep Learning +0.024; Python programming +0.024 ← Python 0.6465 | Deep Learning ← Supervised Learning 0.4855 |
| S00001 × J0974289 Data Business Analyst | moderate | Data Analysis | Data Modeling +0.095; Data Visualization +0.095 | SQL Data tools +0.034 ← SQL 0.7214 | none |
| S00023 × J0071584 Cloud Systems Engineer | poor (ML student, cloud job) | none | Automation Security in the cloud, Cloud infrastructure, Cloud systems engineering: +0.083 each | DevOps practices, Disaster recovery, Scalability: +0.036 each | none |

The same S00001 pair with `--model lightgbm`: the required gaps are +0.259 each and SQL Data tools is +0.009. **The
importance is the model's own:** Baseline 3 and LightGBM weigh the same gaps very differently.

**Findings:**
- **Importance ties within a requirement class**, under every model: `features-v1` describes skills only as counts
  and ratios. Ranking same-class skills needs per-skill features (`features-v2`, then retraining). See ADR 0016.
- **Exact matches are rare.** Among jobs with 3 or more required skills, the best real coverage is 1 of 3. The
  transferable tier carries much of the signal ("Data analysis techniques" ← "Data Analysis" 0.78 in demo 14).
- **TF-IDF relatedness is lexical**, so 0.60 isn't a precision gate. Spot-check precision is about 90% from 0.40
  up, and about 50% below.

**Tests:** 309 passed, 17 skipped, 1 failed (the platform-dependent TF-IDF version pin, unchanged since Phase 8).

### Phase 8: Matching model experimentation (2026-10-11) ✅ (no model selected: no human labels yet)

**Decisions made with the owner at the start** (ADR [0015](docs/adr/0015-matching-model-target-training-and-selection.md)):
- **Labels: not graded yet** (0 of 360 pool pairs). Every quality metric reads "Not yet evaluated", no Overall Score
  is computed, and no model is selected. Baseline 3 stays active.
- **Training data.** The human pool has no train split (15 validation + 15 test students). The owner chose:
  - train on the weak labels (train split);
  - select on the human labels (validation split);
  - hold the test split back for Phase 14.

  Models are trained once. The human-label run only evaluates the same hash-checked artifacts.
- **Target: binary suitability, `relevance_label >= 2`.** The real label distribution:
  - human labels: none;
  - weak labels: 100,848 × grade 0, 3,799 × 1, **0 × 2**, 2,889 × 3 (train split: 63,489 / 2,361 / 0 / 1,830).

  With no grade 2, regression would be binary classification in disguise, and learning to rank wouldn't compare the
  six models like for like.
- **LightGBM installed** (4.7.0): all six candidates ran.

**What was built:**

| Piece | Where | Notes |
|---|---|---|
| Feature matrix | `app/matching/training_data.py` | features-v1, computed by Phase 7's own stage functions, for every weak-label and pool pair: 107,536 pairs, 0 unscored, 1.3 MB, no PII or protected column |
| Candidates | `src/scoring/matching_models.py` + `config/matching_model_candidates.yaml` | LR, DT, RF, sklearn GB, XGBoost, LightGBM. Fixed hyperparameters (a reason each), balanced classes, seed 42, 1 thread. Missing values: native in XGBoost/LightGBM; median + missing indicator in the sklearn models |
| Measurement | same file, `measure` worker | One process per model: a warm-up fit, then 3 timed fits; peak RSS; refit determinism; save → hash → reload → identical scores; latency (30 trials, 1 and 50 pairs); throughput; size; scalability (a power law over 3 training sizes, labelled EXTRAPOLATED); SHAP per model with an additivity check |
| Engine | `src/scoring/matching_model_selection_engine.py` + `config/matching_model_selection_weights.yaml` | The Phase 8 table; the full quality suite (Acc/P/R/F1/ROC-AUC/PR-AUC/MRR/NDCG/P@K/R@K, with CIs); 3 scenarios shown in full; 1,000 random weightings; a gate (beat Baseline 3); a simpler-model check for every winner and always for XGBoost; the ADR 0015 decision. Options: `--reuse-models`, `--activate`, `--record-db`, `--record-existing` |
| Wiring | `app/matching/predictor.py`, `config/matching_model.yaml`, `MATCHING_MODEL` | Stage 4 loads Baseline 3 or a trained artifact, as the flag says. Metadata: `matching_model=<key>`, `matching_model_version=<key>+<sha12>`, `experiment_id`. Refuses another feature or embedding version, or a tampered file. Baseline 3 is kept as the default and as Phase 14's "Baseline 3: structured feature matching"; Phase 7's `evaluate --register` always measures it |
| Tests | `backend/tests/test_matching_models.py` (21) | Exclusion test on the training data and the committed matrix; round trips; tamper refusal; determinism; SHAP additivity per model; rubric; normalisation; "XGBoost is never preferred by name"; decision rule; `--activate` refusal; flag and override; version refusals; end-to-end agent metadata |
| Demo | `scripts/demo.py 13` | The recorded runs. Demo 11 now names the active model |

**Real results** (`results/matching_model_evaluation/`, recorded in Postgres):
- **`mms-20261011T104041Z-nolabels`** (the training run):
  - latency for 50 pairs: LightGBM 0.45 ms, DT 1.27, LR 1.28, GB 1.44, Baseline 3 2.13, XGB 2.35, RF 9.34;
  - training time (median of 3): DT 0.20 s, LGBM 1.03, XGB 1.12, LR 4.03, RF 5.33, GB 11.12;
  - every refit was bit-identical, and every model's SHAP contributions add up to its output.
- **`mms-20261011T104404Z-weakly`** (weak labels: **NOT A SELECTION**):
  - NDCG@10: LightGBM 0.954, XGB 0.954, GB 0.929, RF 0.910, DT 0.823, LR 0.813, Baseline 3 0.515.
  - Winners: Balanced DT (0.711), Quality-prioritized LightGBM, Low-resource DT. **The winner changes by scenario.**
  - Every trained model's top SHAP feature is `branch_match`: the leakage ADR 0014 predicted.
- **Live wiring check:**
  - With `MATCHING_MODEL=decision_tree` forced, S00002 (Data Science) gets Data and Business Analyst jobs, with
    `decision_tree+8a11032b8829` in the metadata.
  - An uploaded résumé with no branch or stated interest gets 0.0 for every job, because a weak-label model has no
    other signal. This is another reason the human labels must decide.

**Environment found and fixed on this machine (2026-10-11):**
- **Setup.** There was no `.env` or `.venv`, and the Docker database dated from Phase 5. `.env` was recreated
  (gitignored) from the existing container's own settings. The stack was rebuilt, seeded and processed per the
  setup steps.
- **Line endings.** With `core.autocrlf=true`, git checked `data/` out as CRLF, which changed every data hash.
  `.gitattributes` now marks `data/**` and `models/**` as `-text`. The files were rewritten from HEAD (identical
  blobs) and match the reproducibility hashes again.
- **The TF-IDF version differs by platform.**
  - It is `tfidf-953a75d166d8` here and `tfidf-3ec097a5876e` on the M2, on identical data. The likely cause is
    rounding of `log` at the 6th decimal of the IDF values: 1,806 of them sit within 1e-9 of a rounding boundary.
  - The calibration was rebuilt for `953a…`.
  - `test_the_deployed_tfidf_version_is_unchanged_by_the_benchmark_rewrite` **fails on this platform** for that
    reason, with or without Phase 8.
  - Owner decision: whether to make the version hash platform-stable. That would change the version everywhere.
- **Baseline 3's registration** still names `match-20261009T155859Z-weakly`, which is recorded in the M2's Postgres,
  not this one.
- **Backend image:**
  - `xgboost-cpu`, because the full `xgboost` wheel bundles CUDA;
  - `libgomp1`, which LightGBM needs;
  - `models/` is copied into the image.

**Tests:** in the backend image, 296 passed, 17 skipped and 1 failed (the platform-dependent TF-IDF version pin
above).

### Phase 7: Matching Agent (2026-10-09) ✅ (prediction is an interim weighted baseline until Phase 8)

**What was built** (`backend/app/agents/matching_agent.py`, 10 tools in `backend/app/tools/matching_tools.py`, logic in `backend/app/matching/`):

| Stage | Tool | Notes |
|---|---|---|
| 1 Eligibility | `filter_eligible` | Clear failures only: degree level, minimum experience (vs the longest stated duration), configured mandatory skills (none by default), plus near-duplicates and visibility (students only). Unknown never excludes. Same-country-first tiering ([ADR 0013](docs/adr/0013-matching-eligibility-and-location-tiering.md); the prompt's "existing" `match_student()`/ADR didn't exist, so this is its first record). Counts per reason |
| 2 Retrieval | `retrieve_jobs` / `retrieve_candidates` | Phase 6's section-weighted protocol on stored pgvector embeddings (skills, experience vs responsibilities, projects vs description), weights renormalised, top 50. The model is never named in code (`EMBEDDING_MODEL`, interim TF-IDF) |
| Similarity | `compute_similarity` | 5 more signals (projects, certifications, interest vs title, past roles vs title, field vs job), each as a **calibrated percentile** over 292 x 4,941 student-job pairs (`python -m app.matching.calibration`): raw cosine is model-specific |
| 3 Features | `generate_features` | Exactly the 20 spec §11 features, `features-v1`, each defined in `feature_builder.py`. Transferable = related skill names (similarity >= calibrated threshold 0.3). Checked against the protected-attribute contract |
| 4 Prediction | `predict_suitability` | `weighted_baseline_v1`: transparent weighted sum, **placeholder weights pending Phase 8** ([ADR 0014](docs/adr/0014-matching-pipeline-and-weighted-baseline.md)); all ingredients kept for Phase 11 |
| 5 Ranking | `rank_candidates` | Ties by id; at most 1 posting per role (every posting of a role has identical text in this data) |
| 6 Skill gap | `identify_skill_gaps` | matched / partially related / missing / important missing / transferable, plus relations and the inferred required/preferred lists: the `SkillGap` contract for Phase 10 |
| Research | `evaluate_model`, `benchmark_matching` | `python -m app.matching.evaluate --register` evaluates, benchmarks and records an experiment, and registers the config |

- **Employability score:** 8 dimensions (skill, education, experience, projects, certifications, semantic, career interest, transferable), each 0–100, plus an overall score that equals suitability × 100. It is never raw cosine × 100.
- **Version metadata:** every prediction carries 7 strict version fields (`PredictionVersions`: missing or sentinel → invalid, nothing returned). `matching_model_version` is a hash of the config, and `experiment_id` is the registration run (`match-20261009T155859Z-weakly`).
- **Other changes:**
  - `seed --process` runs the Profile Agent for all 286 students (23 s, all succeeded).
  - `job_source` (all / dataset / uploaded) is now a request option.

**Real results:**
- **Registration run** (weak labels, validation, 57 students): `weighted_baseline_v1` NDCG@5 **0.518** (0.435–0.602) vs semantic similarity alone 0.483 (0.389–0.573). The difference, +0.035 (−0.047 to +0.124), is not distinguishable. The circularity check passes (tau 0.26 / 0.21).
- **Latency:** full Stage 1–6 pipeline over 4,992 postings, median **1.0 s** per student (95th percentile 1.2 s).
- **Funnel over all 286 students:** median 434 eligible postings per student (of 4,990). Mean exclusions per student: experience 1,690, degree 1,598, not visible 684.
  - **All 18 diploma holders get no eligible job**: the dataset's postings require a bachelor's or higher. Owner decision in ADR 0013.
- **3 report students** (train split, dataset jobs). Top-5s are distinct roles, and every match carries all 7 fields:
  - CSE-DS: ETL Developer, Investment Analyst, Data Analyst, BI Analyst, Marketing Analytics.
  - IT/web: Frontend Web Developer 87.4, UI Developer, Frontend Developer, Web Designer, Network Admin.
  - Mechanical: Manufacturing Engineer, Mechanical Design Engineer, Industrial Designer, Aircraft Design Engineer, HVAC Designer.
  - **Skill-match scores are low** (exact matches are rare with 92% unmapped job skill phrases), and TF-IDF's lexical artefacts show (a data student gets "Investment Analyst" via "analyst").

**Bugs found and fixed this phase:**
- **Phase 4 data loss.** Asking for an uploaded student by id rebuilt the profile from the stored columns, which are empty for uploads because the resume isn't stored, and overwrote the real profile.
  - Fix: the Profile Agent now reuses an upload's stored profile.
  - Regression test, verified to fail without the fix.
  - Two synthetic demo uploads in the dev database had already been overwritten. They were left as they are.
- **Duplicate uploads.** v1 skipped uploads without a company, so the same JD was stored twice and recommended twice at #1 and #2. Fixed by `job-dedup-v2` (ADR 0010 addendum). Dataset results are identical: 51 near-duplicates.
- **v2's first SQL was broken.** It passed an untyped NULL parameter, so the duplicate check errored and the Job Agent failed closed.
  - Caught by reading demo 3's output, which said "duplicate_check: Not evaluated".
  - Fixed, and a Postgres regression test added.
- **One role could fill a whole top 5.** Five identical "Frontend Web Developer" postings ranked 1 to 5. Fixed with the per-role cap.
- **Calibration had a quadratic loop.** It ran for more than 10 minutes; now 17 s.
- **A model name in a docstring** was caught by the no-hard-coded-model test.
- **An error message over the 120-character redaction limit** was hidden as "[TEXT len=171]". Shortened.

**Verified:**
- `pytest`: **278 passed**, 15 skipped by default. With `TEST_DATABASE_URL`: **292 passed**, 1 skipped (live Gemini).
- The PII/protected-attribute test plus 13 fairness guards pass.
- `scripts/demo.py`: all 12 demos ran against the rebuilt backend.
- Processed-data SHA-256s are unchanged.

### Phase 6: Embedding experimentation framework (2026-10-09) ✅ (the model choice waits for human labels)

**What was built.**

| Piece | Where | Notes |
|---|---|---|
| Model coverage | `src/embeddings/embedding_model_benchmark.py` | A SBERT all-mpnet-base-v2, B E5 e5-base-v2, C BGE bge-base-en-v1.5, **D all-MiniLM-L6-v2** (most-used small model; same recipe as A, so A vs D isolates size). E custom: "Not feasible — insufficient labeled data". F hybrid: Phase 7/14. TF-IDF (deployed `tfidf-3ec097a5876e`) is the floor in every table |
| Audit fixes | same | **I-3** empty sections are never embedded, and section weights are renormalised. **I-4** each job field is used once. **I-5** latency is the median of 30 warm trials plus stdev and p95, with batch latency and throughput, each model in its own offline process |
| Text view v2 | same | Job text = title/role, skills, description, responsibilities. `experience_required` and `qualification_required` are dropped: they are **randomly assigned** per posting (one role shows a median of 7 degrees and 10 experience ranges). Jobs = the 376 role representatives (role fixes the description, skills and responsibilities) |
| Fixed splits | `src/evaluation/splits.py` | Per student, SHA-256 bucket: train 180 / validation 57 / test 49. Selection uses **validation only**; train is never evaluated; test is held back for Phase 14 |
| Label loader | `src/evaluation/relevance_labels.py` | Validates every row (student, job, grade 0–3, `label_source` ∈ the 3 honest values, split = the student's fixed split, no duplicates) and **fails loudly**, listing every bad row. A provenance manifest is required. **Circular-evaluation check:** no manifest, model-made labels, or any model whose scores reproduce the labels (mean per-student Kendall tau ≥ 0.9) → "evaluation validity: LOW — labels not independently sourced". Synthetic labels are stamped "SYNTHETIC LABELS — NOT REAL GROUND TRUTH", weak labels stamped likewise |
| Metrics | `src/evaluation/retrieval_metrics.py` | P@K, R@K, MRR, NDCG@K (graded gain), Hit Rate, Kendall tau. Only judged jobs are ranked (condensed lists), ties broken by job_id, K = 5 and 10, paired bootstrap 95% CIs |
| Engine | `src/embeddings/model_evaluation_engine.py` + `config/model_selection_weights.yaml` | The spec's exact table columns (Markdown + JSON). Weights with a one-line reason each. Min-max normalisation, cost on log10. Gate: must beat the TF-IDF floor. **Sensitivity:** 3 scenarios, 1,000 random weightings, and the alternative scoring protocol. The 5 BEST MODEL lines. A cosine-range table proving raw cosine isn't comparable across families. A **Decision** line computed by the rule fixed in ADR 0012 |
| Human labelling (option 1) | `backend/app/labelling/`, `/labelling` | Pool `pool-5c6a5972b32d`: 30 validation/test engineering students (20 placement, 10 Kaggle) × 12 jobs. Of those 12, 10 are TREC-style round-robin picks from the 5 models and 2 are random. The models only choose which pairs are shown; people give every grade, blind to the source and the scores. Each model ranks 42–45% of the pooled jobs in its own top 5, so the pool is not biased to one model. PII-masked; gender, CGPA, salary and company are hidden. 4 students are graded twice (quadratic-weighted kappa). Export: `scripts/export_labels.py` |
| Weak labels | `src/evaluation/weak_labels.py` + `config/weak_label_rules.yaml` | The student's stated domain × the job's role family (regexes reviewed by hand; one over-match was fixed). 107,536 pairs, grades 3/1/0, `weakly_labelled`, `generated_by: rules`, no model involved. **The team should review the families** |
| Experiment tracking | migration 0008, `GET /experiments` | `experiment_runs` stores every spec §25 field; `experiment_metrics` gives each value a status (measured / extrapolated / not_yet_evaluated / not_feasible). `model_versions.experiment_id` is now a foreign key |
| ADR | [0012](docs/adr/0012-embedding-model-selection.md) | Proposed. The selection rule is fixed before any human label exists. Primary model today: **none (Insufficient evidence)** |

**Real results** (`results/embedding_evaluation/`, recorded in Postgres):
- `emb-20261009T074529Z-nolabels`: cost measured, every quality cell "Not yet evaluated". Speed and memory winner: MiniLM.

  | Model | CPU latency per pair | Throughput, CPU / MPS (texts/s) | Peak memory | Size | Dependencies |
  |---|---|---|---|---|---|
  | TF-IDF floor | 1.9 ms | 3,817 / n/a | 97 MB | 0.1 MB | 6 packages, 121 MB |
  | SBERT | 686 ms | 3.9 / 49.5 | 741 MB | 418 MB | 31 packages, 556 MB |
  | E5 | 654 ms | 4.0 / 61.9 | 635 MB | 418 MB | 31, 556 MB |
  | BGE | 643 ms | 4.3 / 61.9 | 654 MB | 418 MB | 31, 556 MB |
  | MiniLM | 125 ms | 25.0 / 249 | 210 MB | 87 MB | 31, 556 MB |

- `emb-20261009T074653Z-weakly` (**WEAK LABELS — RULE-BASED, NOT HUMAN GROUND TRUTH**; validity LIMITED; 54 validation students):

  | Model | NDCG@5 | MRR |
  |---|---|---|
  | BGE | 0.633 | 0.823 |
  | E5 | 0.603 | 0.834 |
  | TF-IDF | 0.580 | 0.781 |
  | MiniLM | 0.577 | 0.769 |
  | SBERT | 0.524 | 0.737 |

  - **No model is statistically distinguishable from TF-IDF.** BGE is +0.053, 95% CI −0.011 to +0.119.
  - Among eligible models, BGE wins all 3 weight scenarios and 100% of random weightings. **But the winner changes with the scoring protocol:** under section-weighted scoring, MiniLM wins.
  - TF-IDF's own Overall Score (0.755) beats every candidate's.
  - Decision: "Not a selection" (weak labels can't select).
- Cosine ranges differ completely between families. E5 scores everything 0.675–0.754; SBERT spreads from 0.034 to 0.377. This is why the Overall Score never uses mean similarity.
- The Phase 1 outputs `results/embedding_model_comparison*` are kept as audit evidence only. Their single cold call measured SBERT at 1,660 ms and BGE at 14 ms; warm medians show all three base models at 643–686 ms.

**Bugs and flaws found and fixed this phase:**
- **`record_experiment` passed 24 values for 23 columns.** The Postgres test caught it before any run was recorded.
- **A weak-label rule over-matched.** `design engineer` put "Aircraft Design Engineer" and "Mechanical Design Engineer" into civil engineering; found by listing every family's members.
- **Report wording.** It said "Not yet evaluated" where MPS doesn't apply to TF-IDF, and it hid that TF-IDF outscores the winner overall. Both are fixed: the BEST MODEL lines now state the comparison with the floor, the gate's fragility, and when the cheapest model was excluded.
- **Decoupled the deployed TF-IDF from the benchmark's text functions.** Its corpus texts are now frozen in `tfidf_baseline.py`, so the I-4 fix couldn't silently change the deployed model's version (verified unchanged).

**Verified:**
- `pytest`: **255 passed**, 14 skipped by default. With `TEST_DATABASE_URL`: **268 passed**, 1 skipped (live Gemini). The labelling and experiment DB tests use their own data and clean up; the real pool still has 0 test judgments.
- `scripts/demo.py`: all 10 demos ran against the rebuilt backend. Demo 9 is read-only by design and never saves a grade.
- Processed-data SHA-256s are unchanged, and the TF-IDF version is unchanged.

### Phase 5: Job Intelligence Agent (2026-10-08) ✅

**Pipeline** (`backend/app/agents/job_intelligence_agent.py`): 12 schema-validated tools (`backend/app/tools/job_tools.py`), for a stored job row or a free-text JD.

| Stage | Tool | Notes |
|---|---|---|
| Text extraction | `extract_job_text` / `load_job_record` | Same checks as resumes; unknown job_id is rejected without retry |
| PII | `detect_pii` | **Shared** with the Profile Agent; JD contact details are masked |
| Requirements | `parse_job_description` | Free text: `parse_job_sections`, added to `src/parsing/section_parser.py` on top of `_slice_section` (extended with line-anchored headers; resume behaviour unchanged). Stored rows: `JOB_SCHEMA` columns |
| Skills | `extract_job_skills` → `normalize_skills` | Phrase splitter for the undelimited `skills` field; **the same** normaliser and alias table as students |
| Required vs preferred | `classify_skill_requirements` | **INFERRED, heuristic v2** ([ADR 0008](docs/adr/0008-job-skill-phrases-and-required-preferred.md)). Each skill records `requirement_basis`, and every surface says it is inferred |
| Education/experience | `extract_requirements` | `guess_degree()` reused, plus B.Com/M.Com/BBA/BA/MA on the job side. Degree level 1–4; min/max years as numbers. 100% of the 4,990 rows parse |
| Work type/location | (mapped through) | `JOB_SCHEMA` names kept |
| Dedup | `check_duplicates` | Same company + title + ≥0.9 shingle Jaccard ([ADR 0010](docs/adr/0010-job-near-duplicate-detection.md)). A duplicate JD is not re-stored |
| Structured JD | `extract_job_posting` | `JobPosting` in `schemas/job.py`. `gender_preference` only in `visibility`, not in `SCORING_FIELDS` |
| Embedding | `generate_embedding` | **Shared**; interim TF-IDF stored sparse, with model name and version ([ADR 0011](docs/adr/0011-sparse-tfidf-embedding-storage.md)) |
| Validation + DB | `validate_job_posting` → `save_job` | Writes `jobs`, `job_skills`, `job_responsibilities` and `embeddings` (`entity_type="job"`); migration 0007 |

**Gender preference** ([ADR 0009](docs/adr/0009-gender-preference-visibility-filter.md)): `is_visible_to()` / the `check_job_visibility` tool, called before scoring. An unknown student gender is never hidden, and `JOB_GENDER_VISIBILITY_FILTER=false` turns the filter off. `backend/tests/test_fairness_guards.py` proves it is in no feature list, the guard rejects it, and no other module references it. **Concern recorded:** hiding postings by gender is itself discriminatory, and the ADR recommends defaulting the filter to off.

**Real-data results:**
- `seed_jobs --process`: **4,990/4,990 jobs succeeded in about 50 s**.
- **51 near-duplicates in 50 groups**, all similarity 1.0. They have the same company, title and text but different country, salary and work type. They are flagged, not deleted.
- 27,780 job skills: **43.3% inferred required** (11,938 named in role text, plus 101 fallback); 15,741 preferred.
- 29,940 sparse job vectors take **6.8 MB**, against about 480 MB dense.
- Five real rows (one per work type, five countries) ran through the live API (`results/phase5_sample_postings.json`).

**Bugs and flaws found and fixed this phase:**
- `guess_degree()` read the word "be" as B.E (narrow fix; 0 existing rows affected; ADR 0006 addendum).
- The PII phone regex missed every sentence-final number, and so did the log redactor (both fixed; detector v2; ADR 0005 addendum).
- Heuristic v1's cue words produced only false positives on real data. Commas inside "(e.g., …)" merged whole skill lists. The stemmer missed "planning"/"plan". The fallback could override an explicitly preferred skill. All four were found by inspecting real output, and v2 fixes them (ADR 0008).
- The shared normaliser turned "(e.g., Docker, Kubernetes)" into a junk skill; example lists are now dropped. Student coverage is unchanged (61/13/26%).
- **Log leak:** tool-call logs recorded argument *values* under keys the redactor didn't know (`posting`, `job`, `description`), so short JD sentences reached the logs. Found by a new JD log test. Tool-call logs now record argument **shapes only** (`[str len=N]`, `[list len=N]`, numbers, booleans) for every tool, safe by default.

**Verified:**
- `pytest`: **198 passed**, 12 skipped by default. With `TEST_DATABASE_URL`: **209 passed**, 1 skipped (live Gemini).
- Processed-data SHA-256s are unchanged by this phase's parser changes.

**Deviations from the Phase 5 prompt:**
- `parse_job_sections` did not exist, so it was added to `section_parser.py` (not "reused").
- `guess_degree()` lives in `resume_text_parser.py` (not `ingest_kaggle_resumes.py`).
- The "ADR-002/004/006 in docs/ADR.md" references don't exist; this phase's records are ADRs 0008–0011.
- Embedding storage was switched to sparse (ADR 0011).
- A feature contract (`app/matching/features.py`) was added so the gender test has a real feature list to check.

**Known limitations:**
- Required/preferred is a proxy ("named in role text"); its accuracy is unmeasured (no labels).
- 92% of job skill phrases aren't in the alias table.
- About 0.1% of phrases are under-split.
- Qualifications in the job data look randomly assigned (an Event Manager requiring an M.Tech), which is more evidence it is synthetic (audit I-7).

**Git:** Phases 4 and 5 committed together (their changes overlap in shared files) on branch `phase-4-5-profile-job-agents`, pushed, and fast-forwarded into `main`.

---

### Phase 4: Profile Agent (2026-10-08) ✅

**Pipeline** (`backend/app/agents/profile_agent.py`): each of the 9 stages is a schema-validated tool (`backend/app/tools/profile_tools.py`) called through the BaseAgent loop.

| Stage | Tool | Notes |
|---|---|---|
| 1. Text extraction | `extract_text` | Raw text, `.txt` (UTF-8 or BOM-marked UTF-16), `.pdf` (pdfplumber) and `.docx` (python-docx). Rejects empty, binary, undecodable, unsupported, over-5 MB and scanned PDFs. |
| 2. PII detection | `detect_pii` | Regex + header heuristic ([ADR 0005](docs/adr/0005-pii-detection.md)); every later stage gets masked text. Measured recall on a labelled synthetic set: url 100%, phone 83%, email 75%, address 75%, dob 67%, **name 62%**. 0/12 false positives. |
| 3. Sections | `parse_resume` / `parse_student_sections` | Wrap the **existing** `src/parsing` code (`parse_resume`, `guess_degree`, `_slice_section` for projects and certifications); nothing reimplemented. |
| 4. Skills | `extract_skills` | The existing `extract_skills()`, with one narrow bug fix ([ADR 0006](docs/adr/0006-parser-dotnet-fix.md)). |
| 5. Normalisation | `normalize_skills` | `app/profile/data/skill_aliases.csv`: 351 aliases, 179 canonical skills, each with evidence counts from students_unified.csv and jobs.csv. Maps 74% of real student skill mentions; the rest stay as written, marked `unmapped`. |
| 6. Structured profile | `extract_profile` | `StudentProfile` in `schemas/student.py`. Sentinels per [ADR 0007](docs/adr/0007-missing-value-sentinels.md). No PII, no gender. |
| 7. Embedding | `generate_embedding` | Interim TF-IDF baseline, `tfidf-3ec097a5876e`, 4,096 dimensions ([ADR 0004](docs/adr/0004-interim-tfidf-embeddings.md)). One vector per non-empty section plus `full`. |
| 8. Validation | `validate_profile` | Schema check, sentinel list, and a refusal to save if PII remains. |
| 9. Database | `save_profile` | Writes `student_profiles` (`profile` JSONB, `pii_flags` with counts only), the skill/project/certification/experience tables, and `embeddings` with model name and version, in one transaction. |

**Inputs:** a `student_id` (stored record), resume text, or a resume file. When both a student_id and a resume are given, the record supplies the branch and CGPA and the resume supplies the sections and skills. The new `POST /requests/resume-file` endpoint takes multipart uploads.

**Failure handling:**
- Each stage is retried once.
- Malformed input gives `failed / invalid_input` with a clear message and no retry, at either level.
- `detect_pii`, `extract_skills`, `normalize_skills`, `generate_embedding` and `save_profile` **degrade**: the stage is flagged, its dependants are skipped, and the result is `partial`. If PII detection fails, nothing goes to embedding or storage.
- Every other stage stops the run.

**Bugs found and fixed this phase** (all by tests or verification queries):
- The parser corrupted `.NET` into `NET` (ADR 0006). The processed data was regenerated; exactly 4 rows changed.
- `save_profile` received the embedding as a dict, not a model; it now re-validates.
- The global `agent_max_steps=10` was too small for the 11-decision combined path. Agents now set their own step budget.
- PDF and DOCX libraries took about 1 s on first import and could time out the first request. They are now imported at startup, and the TF-IDF model is fitted at startup.
- PII *counts* were being masked in logs, because the keys were `email`, `name` and so on. They are now logged as `{type, count}` entries.
- The `extract_skills` tool skipped the existing `_clean()` step that `parse_resume()` applies, so raw skills kept `\r` line endings. It now cleans the text exactly as the existing pipeline does.
- The `student_skills` unique key dropped the second skill of an item like "Git/GitHub". Migration 0005 added the right key, but its DROP was a silent no-op, because Postgres had truncated the old constraint name to 63 characters. Migration 0006 drops it by its definition instead.

**Verified:**
- `pytest`: **145 passed**, 9 skipped by default. With `TEST_DATABASE_URL`: **153 passed**, 1 skipped (live Gemini).
- A fresh Docker volume applies migrations 0001–0006, fits the embedding model at startup, and seeds 286 students. A synthetic upload stores all 7 skills (Git and GitHub both present). A `.docx` upload through `/requests/resume-file` works.
- Three real samples through the live API give the same profile on two separate runs:
  - placement-cell **S00001**: 25 skills, no missing fields
  - Civil **S00256**: "Skill not detected" (its "Skill Details" section is genuinely empty)
  - Data Science **S00200**: 3 skills, a limitation of the existing parser (ADR 0006)
- Grepping the backend logs for 7 synthetic PII values and 5 resume-content snippets finds **0 matches**. The logs hold `pii_flags` (type and count) only. The Postgres profile and `agent_runs` hold no PII.

**Deviations from the Phase 4 prompt:**
- ADRs are in `docs/adr/` (the referenced `docs/ADR.md` never existed).
- The TF-IDF baseline was re-implemented, not wrapped (the original was missing).
- `parse_student_sections` is a new thin wrapper (the prompt named it, but it didn't exist).
- The profile is a new `StudentProfile` model (Phase 2 had `StudentRecord`, the CSV mirror, which is kept).

**Known limitations (honest):**
- Name detection misses 38% on the synthetic set.
- The parser's heuristics (S00200's skills and education, section bleed) are documented in ADR 0006, not changed.
- TF-IDF is lexical, not semantic; no quality claim is made.
- The request-level status for profile requests is still `failed`, because the Matching Agent (Phase 7) is a shell.

**Git:** committed with Phase 5 on branch `phase-4-5-profile-job-agents`.

---

### Phase 4 groundwork (2026-10-08) ✅

The owner asked for every Phase 4 prerequisite to be cleared.

| Item | What was done |
|---|---|
| pandas in the backend | `backend/requirements.txt` adds numpy 2.0.2 and pandas 2.3.3, the same versions that produced the byte-identical data. The image now copies `config/` and `src/` (`PYTHONPATH=/srv/backend:/srv`). Verified inside the container: `parse_resume()` and `guess_degree()` run. Image size 548 MB (was 258 MB). |
| Student tables | `0002_create_student_tables.sql`: `student_profiles` (1:1 with STUDENT_SCHEMA), `student_skills` (raw and normalised, technical/soft, with extractor version), `student_projects`, `student_certifications`, `student_experience`. |
| Embeddings + model registry | `0003_...sql`: `embeddings` follows spec §23 (vector dimension enforced by a CHECK) plus a `section` column. `model_versions` allows one `selected` model per kind, and a model can't be marked selected without an `experiment_id` (rule 20, enforced in SQL). |
| Real students in the database | `python -m app.db.seed` reads the CSV with pandas, validates all 286 rows through `StudentRecord` and upserts them. Idempotent. Kaggle rows keep CGPA, gender and graduation year NULL. |
| Embedding decision | [ADR 0002](docs/adr/0002-no-embedding-model-before-phase-6.md): no provisional model; embeddings stay "Not yet evaluated" until Phase 6 selects one, then a backfill runs. |
| ADRs started | `docs/adr/` with 0001 (Gemini), 0002 (embeddings), 0003 (SQL migrations), as spec §44 requires. |
| Test isolation fix | An autouse fixture clears settings environment variables, so `source .env` in a shell can no longer change test results. A real false failure was caught this way. |

**Verified:**
- `pytest`: 99 passed and 7 skipped by default. With `TEST_DATABASE_URL`: **105 passed**, 1 skipped (live Gemini only).
- **Fresh database volume** (`docker compose down -v` then `up`): the container applied all 3 migrations on start, pgvector 0.8.7 is present, the seed upserted 286 rows (a second run also succeeded), and `POST /requests` for `S00001` routed and persisted.

---

### Phase 3: Supervisor Agent (2026-10-08) ✅

**What the Supervisor does now** (`backend/app/agents/supervisor.py`, `routing.py`):

1. **Routing.** The LLM (Gemini) chooses the task type and the *set* of agents.
   - Its answer must parse into `LLMRoutingDecision`, which has enum fields and a rationale, and deliberately **no numeric fields** (Rule 5 by construction).
   - The answer must also pass `validate_decision()`. For example: a resume upload must include `profile_agent`, `learning_agent` needs `matching_agent`, and `job_intelligence_agent` needs a job_id or JD.
   - If anything fails, the deterministic `FALLBACK_ROUTES` table decides instead and the reason is logged. Triggers: no LLM configured, an error, a timeout (10 s, at most 1 retry), invalid output, or a rejected decision.
   - Python, not the LLM, orders the agents into dependency waves: profile ‖ job_intelligence → matching → learning ‖ explainability. Independent agents run concurrently (`asyncio.gather`).
2. **Delegation.** A new `Delegate` action in BaseAgent lets the Supervisor hand work to agents through the same validated decision loop. Agent hand-off is separate from tool calls (spec §41).
3. **Structured I/O.** Each agent's typed input is built from the request plus upstream *structured* outputs. For example, matching's `student_id` comes from the profile's `StudentRecord`, and learning's `skill_gaps` come from matching's `SkillGap`. An agent that can't be given valid input is **skipped**, with the reason recorded.
4. **Output validation and failure policy** (documented in `FAILURE_POLICY`):
   - Every agent's envelope and its data are re-validated against the agent's output schema.
   - Only transient failures (timeout, tool_failure, internal_error) are retried, and only once.
   - Profile, job intelligence and matching are `fail_request`. Learning and explainability are `degrade`: the request becomes `partial` but the match results are kept.
5. **Aggregation.** `SupervisorResponse` has one entry per planned agent. Missing results are literally "Not yet evaluated" or "Insufficient evidence". The summary is deterministic; LLM-written explanations come with Phase 11.
6. **State and persistence.** `AgentState` is kept through the whole request and saved to the new `agent_runs` table (`backend/migrations/0001_create_agent_runs.sql`, applied by `python -m app.db.migrate`, which the container runs at start).
   - Data is PII-scrubbed before storage.
   - A database failure gives `persisted: false` plus a `persistence_failed` error; it never crashes the request.
7. **Prompt-injection defence** (`core/sanitize.py`):
   - Resume and JD bodies are **never sent to the LLM** for routing; it sees only metadata.
   - The user's typed question is Unicode-normalised, stripped of invisible characters, PII-masked, length-capped, HTML-escaped and wrapped in `<untrusted_input>` tags, with an explicit guard in the system prompt.
   - Output validation means even a compromised model can only produce a legal route.
8. **API.** `POST /requests` runs the Supervisor. `GET /requests/{request_id}` reads a persisted run.

**Verified this phase:**
- `pytest`: **97 passed, 2 skipped**. The skipped two are opt-in; the Postgres one was then run against the real database and **passed**.
  - Required tests: LLM routing success (resume, JD), LLM exception → fallback, unexpected client crash → fallback, timeout → 1 retry → fallback, invalid output → fallback, and 4 prompt-injection tests.
- **Mutation check:** each of these deliberate breakages made a test fail: removing the LLM retry, removing escaping, removing output-schema validation, removing decision validation.
- **Docker end to end:** all three containers healthy; the migration applied; `POST /requests` with a resume containing an injection and an email gave a deterministic route (profile → matching → learning → explainability; no job intelligence).
  - Profile is `not_implemented` ("Not yet evaluated"); its dependants are skipped ("Insufficient evidence"); status `failed`; `persisted: true`.
  - The Postgres row contains neither the email nor the resume text.
- **Secrets:** `.env` is gitignored. A search for Anthropic- and Google-style API key prefixes (as the Phase 3 prompt asked) finds nothing in tracked files, new files or git history. No generated secrets and no resume email appear in the container logs.

**Deviations from the Phase 3 prompt:**
- Gemini (google-genai 2.29.0) instead of Anthropic, with `GEMINI_API_KEY` in the repo-root `.env`, not `backend/.env`.
- **No key yet**, so the real Gemini call is untested. The LLM path is proven with a fake client that uses the real validation code, and the Gemini wrapper is tested against the real SDK types with the network mocked.
- Plain-SQL migrations plus a small runner, not Alembic.
- The routing LLM never sees resume or JD bodies. This is stronger than "wrap them in tags"; wrapping is used for the typed question.
- `SupervisorInput` was redesigned (`actor`, `request_kind`, `resume_text`, `jd_text`); the Phase 2 version was a placeholder.

**Still not done:** the five specialist agents are shells, so every real request currently ends `failed` with honest sentinels. There is no authentication on the API yet (Phase 16). Redis is still unused.

**Git:** Phases 2–3 and the Phase 4 groundwork are committed together on branch `phase-2-3-supervisor` (built on `phase-1-audit`) and pushed to `origin`. Not merged into `main`.

---

### Phase 2: Agent architecture (2026-10-08) ✅

**1. Pipeline split into modules** (move only; the owner left the decision to the AI)
- `preprocess.py` → `config/schema.py`, `src/parsing/*`, `src/pipeline/*`. `preprocess.py` is now a 10-line entry point.
- `embedding_model_benchmark.py` → `src/embeddings/` (moved with `git mv`; only its repo-root path changed).
- Verified: every original code line appears verbatim in the new modules (checked by script). Both processed CSV SHA-256s are unchanged. `data_quality_report.md` is unchanged. The benchmark smoke run passes from its new path.

**2. Backend skeleton (`backend/`)**
- `BaseAgent` (`agents/base.py`) is a real decision loop: `decide_next_action` → `CallTool` or `Finish`, tool call through the registry, output validation, retry (`AGENT_MAX_RETRIES`, default 2), then `fallback()`. Bad data never flows on.
  - Explicit termination: `run()` always returns an `AgentOutput` with status success, partial or failed and one of 9 termination reasons. It is bounded by `AGENT_TIMEOUT_SECONDS` and `AGENT_MAX_STEPS`, and never raises or returns None.
  - Structured JSON log per tool call and per run, tagged with `request_id`, with PII redacted.
- Tool registry (`tools/registry.py`): `@register_tool(name, output_schema=...)`. It validates input and output and only ever raises `ToolValidationError` or `ToolExecutionError`.
- Schemas:
  - `AgentState` exactly as spec §5.1. `model_versions` defaults to the 7 spec §24 keys, all "Not yet evaluated".
  - `StudentRecord` and `JobRecord` mirror `config/schema.py` 1:1 and in order. All 286 + 4,990 CSV rows load directly; source blanks become None.
  - Per-agent I/O contracts in `agent_io.py`.
- Six agent shells. Each has a full role description plus shared guardrails (no LLM arithmetic, no invented results, resume/JD text is untrusted, no PII), its spec §20 `planned_tools`, and its contracts. Each returns `failed / not_implemented / "Not yet evaluated"` until its phase.
- FastAPI: `GET /health` (dependency reachability, no secrets) and `GET /agents` (introspection).

**3. Infrastructure**
- `docker-compose.yml` uses images pinned to exact tags:
  - postgres `pgvector/pgvector:0.8.7-pg16-bookworm`, with the extension enabled by `backend/db/init/01_enable_pgvector.sql`
  - redis `redis:7.4.11-alpine` (present but deliberately unused, spec §23)
  - backend built from `backend/Dockerfile` on `python:3.11.17-slim-bookworm`, running as a non-root user
- Healthchecks on all three services. Ports bind to 127.0.0.1 only. Credentials come only from `.env`.
- Pinned backend deps: fastapi 0.143.0, uvicorn 0.54.0, pydantic 2.14.0, pydantic-settings 2.15.0. Dev: pytest 9.1.1, httpx2 2.13.1.

**Deviations from the Phase 2 prompt, and why:**

| Prompt said | Done instead | Why |
|---|---|---|
| `backend/.env.example` | `.env.example` at repo root | Compose reads the root `.env` for `${POSTGRES_*}` substitution, so postgres receives only its own credentials, not the LLM key or JWT secret |
| `ANTHROPIC_API_KEY=` | `GEMINI_API_KEY=`, `GEMINI_MODEL=`, `LLM_PROVIDER=gemini` | Owner's provider decision |
| Don't touch files outside `backend/` | Split `preprocess.py` into `config/` + `src/` | The prompt assumed those modules existed; the owner approved the split. Code was moved, not changed (verified above). |
| `pytest` may collect zero tests | 58 tests | They prove the agentic acceptance bar: validation, retry, fallback, termination, PII-safe logs, schema 1:1 with the CSVs, no secrets, no hard-coded model |
| (not listed) | Added `backend/requirements-dev.txt`, `pytest.ini`, `.dockerignore`, `backend/db/init/*.sql`, `docs/MASTER_SPEC.md` | Test deps, test discovery, lean image, pgvector enablement, spec available to the team |

**Verified this phase:**
- `pytest`: **58 passed**, 0 warnings, about 3 s.
- `docker compose up --build -d --wait`: all three containers **healthy**.
- `GET /health` → `{"status":"ok","llm_provider":"gemini","llm_key_configured":false,"llm_calls_enabled":false,"dependencies":{"postgres":"reachable","redis":"reachable"}}`
- `GET /agents` lists 6 agents, all `implemented: false`, with phases 3 / 4 / 5 / 7 / 10 / 11.
- pgvector: `vector 0.8.7` is installed, and a real distance query returns `1`. Redis answers `PONG`.
- Neither of the generated secrets appears anywhere in the container logs. Backend image size: 258 MB.
- VS Code: 0 diagnostics.

**Explicitly not done at the end of Phase 2:** no agent called the LLM, and no agent produced predictions, scores or any real output. Every agent result was "Not yet evaluated". No tools were registered, no database tables existed, and Redis was unused.

**Notes for later phases:**
- Phase 4 will need the backend image to include `config/` and `src/` plus pandas. The build context is already the repo root for this reason.
- The `.env` on the owner's machine has generated local credentials. It is gitignored and has no Gemini key yet.

**Git:** committed together with Phase 3 on branch `phase-2-3-supervisor`.

---

### Phase 1: Repository audit + environment fixes (2026-10-08) ✅

**Audit:** the full findings are in [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md).
- The repo held only `preprocess.py`, `embedding_model_benchmark.py`, two processed CSVs and the `results/` files.
- Several files the phase prompts describe did not exist: ADR.md, the TF-IDF baseline, the `config/` and `src/` modules, and the READMEs.
- `preprocess.py` is deterministic, with byte-identical output on Python 3.9.6 and 3.11.17.
- The SBERT / E5 / BGE benchmark runs for real, but it **measures only cost, not quality**: there are no relevance labels.
- `overall_score_mean` must not be used to compare models, because empty sections inflate it, especially for E5 (I-3). Latencies depend on run order (I-5).

**Fixes made after the audit:**
- Python 3.11 `.venv` with pinned `requirements.txt` (fixed the Pylance unresolved-import warnings).
- `.gitignore`.
- Repo-root-relative paths, with raw data read from `data/raw/` or `$RAW_DATA_DIR`.
- **Bug fix I-6:** NaN is now counted as missing in the data-quality report. achievements 166→216, internship_raw 0→57, technical_projects 166→167, skills_raw 33→35, certifications 166→173.

**Still open** (numbers refer to `PROJECT_AUDIT.md` §I):
- I-2: relevance labels → Phase 6 / spec §8
- I-3, I-4, I-5: benchmark methodology → Phase 6
- I-7: job data looks synthetic and Excel-truncated → must be acknowledged in research claims (spec rule 11)
- I-8: degree and skill normalisation → Phases 4 and 5
- I-9: imbalanced cohort
- I-12: public repo with real student data

**Git:** committed as `1557a50` on branch `phase-1-audit` and pushed. It is not merged into `main`.
