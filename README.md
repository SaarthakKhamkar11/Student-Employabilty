# Student Employability Intelligence & Job-Matching System

A multi-agent system that reads a student's resume, finds the jobs they are most suitable for, explains **why** with
model evidence, shows **what they are missing**, recommends **real learning resources** for each gap, and compares
them with **comparable peers**. Recruiters can do the reverse and rank candidates for a job.

Every number the system shows comes from a deterministic tool or a trained model, never from an LLM. The LLM (Gemini)
only routes requests and rewrites validated evidence into readable sentences. Every model was chosen by a selection
rule written down before the human labels existed.

## How it works

```
                 ┌──────────────────────── Supervisor Agent ────────────────────────┐
  request ──────►│ route the task (LLM, or a deterministic router) → plan → retry    │──► response
                 └──┬─────────────┬──────────────┬───────────────┬───────────────┬──┘
                    ▼             ▼              ▼               ▼               ▼
              Profile Agent  Job Intelligence  Matching Agent  Learning Agent  Explainability Agent
              parse resume,  parse the JD,     eligibility →   rank courses +  SHAP evidence →
              detect + mask  required vs       E5 retrieval →  YouTube videos  LLM prose, every
              PII, skills,   preferred skills, 20 features →   for each skill  line validated
              E5 embedding   E5 embedding      LightGBM →      gap             against the
                                               skill gaps →                    evidence
                                               cohort benchmark
```

- **Six agents**, each a decision loop over validated tools (about 50 tools in total). Each agent has its own retries,
  timeouts and graceful degradation (`backend/app/agents/`).
- **Embeddings: E5** (`intfloat/e5-base-v2`), pinned and offline. TF-IDF is the automatic fallback.
- **Matching: LightGBM** over 20 explainable features: similarity percentiles, skill coverage, and branch, education
  and experience fit. Protected attributes are rejected by a guard.
- **Explanations:** exact TreeSHAP. An LLM sentence is shown only if each of its claims traces to an evidence item;
  otherwise the structured evidence is shown.
- **Skill gaps:** importance by counterfactual re-scoring. The model scores the pair again as if the skill were
  present.
- **Learning resources:** a curated Coursera catalogue plus the YouTube Data API, ranked by E5 relevance. Never
  invented: a skill with nothing relevant gets "No resources found".
- **Cohort analytics:** peers filtered by branch, graduation year and target role, the 20 most similar by profile
  embedding, then percentile, skill coverage and relative gap. Aggregates only; no other student is ever identified.
- **Observability:** every agent execution, tool attempt and LLM call is a trace span persisted to Postgres.
  `GET /requests/{id}/trace` rebuilds the real execution tree. PII is redacted from logs and traces, and a test proves
  it.

## Model selection at a glance

| Decision | Chosen | Key evidence (human labels, validation split) |
|---|---|---|
| Embedding model (5 compared) | **E5** | NDCG@5 0.756 vs TF-IDF 0.602 (+0.154, CI +0.005 to +0.314); MRR 0.910; Balanced Overall 0.512 |
| Matching model (6 + baseline compared) | **LightGBM** | NDCG@10 0.844, F1 0.630, ROC-AUC 0.847; Balanced Overall 0.615; wins 76% of random weightings |
| Fallback (TF-IDF features) | **Logistic Regression** | Balanced Overall 0.699, NDCG@10 0.813 |

Full tables, confidence intervals, rejected alternatives and limitations: **[docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md)**.

## Tech stack

- **Backend:** Python 3.11, FastAPI, Pydantic.
- **Data stores:** PostgreSQL 16 + pgvector, Redis.
- **Models:** sentence-transformers (E5); scikit-learn, LightGBM and XGBoost; SHAP.
- **LLM and APIs:** Google Gemini (structured JSON output, validated locally) with a quota-aware model fallback chain;
  the YouTube Data API.
- **Runtime:** Docker Compose.

## Quick start

```bash
cp .env.example .env            # set POSTGRES_PASSWORD (also inside DATABASE_URL) and JWT_SECRET;
                                # optional: GEMINI_API_KEY / GEMINI_MODEL / GEMINI_FALLBACK_MODELS, YOUTUBE_API_KEY
python scripts/stage_embedding_model.py          # copies the pinned E5 weights into the build context
docker compose up --build -d --wait               # postgres + pgvector, redis, backend (migrations run on start)
docker compose exec backend python -m app.db.seed --process         # students: profiles + embeddings
docker compose exec backend python -m app.db.seed_jobs --process    # 4,990 jobs: dedup + full pipeline
curl http://127.0.0.1:8000/health
python scripts/demo.py           # 18 live demos against the running API (or: python scripts/demo.py 17)
```

Without a Gemini key, everything still works: routing is deterministic and explanations use the validated
structured evidence. Without a YouTube key, only the curated catalogue is used.

**Main endpoints:**

| Endpoint | What it does |
|---|---|
| `POST /requests` | Any request: resume upload, JD upload or a question; the Supervisor routes it |
| `POST /requests/resume-file` | Upload a PDF, DOCX or TXT resume |
| `GET /requests/{id}` | The persisted run |
| `GET /requests/{id}/trace` | The run's real agent → tool → LLM tree |
| `GET /experiments` | Every recorded model-selection run |
| `GET /health` | Dependencies, the embedding model in use, and the LLM quota state per model |

**Tests:** in the backend image (it has no pytest installed):

```bash
docker compose run --rm --no-deps -v "$PWD:/srv/work" -w /srv/work backend \
  sh -c "pip install -q pytest==9.1.1 httpx2==2.13.1 && python -m pytest -q"
```

Over 380 tests. Database tests need `TEST_DATABASE_URL`.

## Repository layout

```
backend/app/agents/     the six agents (Supervisor, Profile, Job Intelligence, Matching, Learning, Explainability)
backend/app/tools/      the tools each agent may call (validated inputs and outputs)
backend/app/matching/   eligibility, retrieval, features, prediction, ranking
backend/app/core/       config, PII-safe logging, tracing, trace reconstruction
backend/migrations/     SQL migrations (applied on start)
src/embeddings/         embedding benchmark + Model Evaluation Engine
src/scoring/            matching models + selection engine, skill gaps, SHAP, cohort analytics
src/pipeline/           the data pipeline (preprocess.py)
config/                 every tunable value, each with a written reason
results/                every evaluation report the engines wrote
docs/adr/               architecture decision records (0001–0020)
docs/MODEL_EVALUATION.md   all model comparisons and why each model was chosen
PROGRESS.md             phase-by-phase hand-off log
```

## Project status

Phases 1–12 of 18 are done: the agents, model selection, skill gaps, learning resources, explanations, tracing and
cohort analytics. Next are the research dashboard (13), the evaluation framework on the held-out test split (14), the
feedback loop (15), security hardening (16), testing (17) and documentation (18). See [PROGRESS.md](PROGRESS.md).

## Data and privacy

- Raw datasets are not in the repository (`data/raw/` is gitignored).
- PII is detected and masked before any model or LLM sees resume text, and it is kept out of logs and traces.
- The job dataset is synthetic and templated, so the results describe this data, not real job boards
  (see the limitations in `docs/MODEL_EVALUATION.md`).
