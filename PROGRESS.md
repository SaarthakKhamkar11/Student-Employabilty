# PROGRESS.md — Project hand-off log

**Rule:** update this file at the end of every phase. That means the status table, the "Pick up here" block, and a new entry in the Phase log. Anyone taking over should be able to read only this file and know exactly where to start.

Last updated: **2026-10-08** · Last completed phase: **Phase 3 (Supervisor Agent)** · Next phase: **Phase 4 (Profile Agent)**

**Latest code: branch `phase-2-3-supervisor`** (builds on `phase-1-audit`; neither is merged into `main` yet).

Source of truth for scope: [`docs/MASTER_SPEC.md`](docs/MASTER_SPEC.md). Phase 1 audit: [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md).

---

## ▶ Pick up here: Phase 4 (Profile Agent)

**Ready to start.** All Phase 4 groundwork is done and verified (see the "Phase 4 groundwork" log entry):
- The backend image includes pandas plus the existing `config/` and `src/` (the resume parser).
- The student tables and the `embeddings` / `model_versions` tables exist (migrations 0002–0003).
- The 286 real students are loaded into `student_profiles` with `docker compose exec backend python -m app.db.seed`.
- The embedding question is decided in [ADR 0002](docs/adr/0002-no-embedding-model-before-phase-6.md): no embedding model is used until Phase 6 selects one from evidence, so Phases 4–5 report embeddings as "Not yet evaluated".

What Phase 4 needs:
1. **The Phase 4 prompt.** The owner sends one prompt per phase.
2. **Optional: a Gemini key.** Put `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`, never in chat or a commit. Everything works without it via the deterministic paths. With it, run the live check: `RUN_LIVE_LLM_TESTS=1 .venv/bin/python -m pytest backend/tests/test_live_gemini.py`. **No real Gemini call has been made yet.**
3. Still for Phase 4 itself:
   - Resume *file* upload (PDF/DOCX → text, the first step of spec §5.2) is not built. `POST /requests` takes text only.
   - Skill normalisation (audit I-8) is not built.

Decisions already made (do not re-ask):
- **LLM provider: Gemini.** Wherever the spec or a prompt says "Claude / Anthropic", use Gemini (`GEMINI_API_KEY`).
- **Code layout:** the pipeline was split into `config/` + `src/` in Phase 2. `preprocess.py` is now a thin entry point.

Open items owned by the project owner (not blocking Phase 3):
- **The GitHub repo is public** (checked 2026-10-08) and contains 120 real students' records in `data/processed/students_unified.csv`. Switch it to private.
- **No relevance labels exist** (audit I-2). Phase 6 cannot measure embedding quality without them. Start labelling student–job pairs now (spec §8: `student_id, job_id, relevance_label 0-3, label_source, split`).
- **`docs/ADR.md` and the original TF-IDF baseline** are still missing. The TF-IDF baseline is needed by Phase 6 (spec §34 Baseline 2) and will be re-implemented if the original isn't found.

---

## Phase status (spec §43)

| Phase | Scope | Status |
|---|---|---|
| 1 | Repository audit | ✅ Done (2026-10-08) |
| 2 | Agent architecture | ✅ Done (2026-10-08) |
| 3 | Supervisor Agent | ✅ Done (2026-10-08). LLM path verified with a fake client only; no real Gemini call yet |
| 4 | Profile Agent | ⏭ Next |
| 5 | Job Intelligence Agent | Not started |
| 6 | Embedding experimentation framework | Not started (blocked on relevance labels for quality metrics) |
| 7 | Matching Agent | Not started |
| 8 | Matching model experimentation | Not started |
| 9 | Skill-gap analysis | Not started |
| 10 | Learning Agent | Not started |
| 11 | Explainability Agent + SHAP | Not started |
| 12 | Agent state/tracing | Not started |
| 13 | Research dashboard | Not started |
| 14 | Evaluation framework | Not started |
| 15 | Feedback loop | Not started |
| 16 | Security hardening | Not started |
| 17 | Testing | Not started |
| 18 | Documentation | Not started |

**Gaps in the spec's phase plan to resolve later:**
- No phase owns the database tables in spec §23. The first agent that persists data (Profile Agent, Phase 4) will need them.
- No phase owns the student and company dashboards in spec §30. Only the research dashboard has a phase (13).

---

## Repository layout (after Phase 3)

```
preprocess.py                 thin entry point → src/pipeline/build_unified_dataset.py
config/schema.py              JOB_SCHEMA / STUDENT_SCHEMA (single source of truth)
src/parsing/                  section_parser.py, resume_text_parser.py
src/pipeline/                 ingest_placement_cell.py, ingest_kaggle_resumes.py, ingest_real_jobs.py,
                              validate_and_report.py, build_unified_dataset.py
src/embeddings/               embedding_model_benchmark.py (SBERT/E5/BGE cost benchmark)
backend/app/agents/           base.py (BaseAgent runtime), supervisor.py + routing.py (Phase 3), 5 specialist shells
backend/app/schemas/          agent_state.py, agent_io.py, student.py, job.py
backend/app/tools/registry.py tool registry + validation wrapper
backend/app/core/             config.py (.env loading), logging.py (JSON, PII-redacting), sanitize.py (prompt-injection defence)
backend/app/llm/client.py     Gemini client (structured JSON output, validated locally)
backend/app/db/               migrate.py (SQL migration runner), store.py (agent_runs), seed.py (pandas CSV → student_profiles)
backend/migrations/           0001 agent_runs · 0002 student tables · 0003 embeddings + model_versions
docs/adr/                     architecture decision records (0001 Gemini, 0002 embeddings deferred, 0003 SQL migrations)
backend/app/main.py           FastAPI: GET /health, GET /agents, POST /requests, GET /requests/{id}
backend/tests/                106 tests (99 run by default; 7 opt-in: 6 need TEST_DATABASE_URL, 1 live Gemini)
backend/db/init/              01_enable_pgvector.sql (runs on first postgres start)
docker-compose.yml            postgres+pgvector, redis, backend
.env.example                  template; copy to .env (gitignored)
docs/MASTER_SPEC.md           the project spec
data/processed/               286 students, 4,990 jobs (committed)
data/raw/                     raw sources (gitignored; see setup)
results/                      data-quality report, embedding benchmark output
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

# 4. Embedding cost benchmark (about 50 s; downloads about 1.25 GB of models on first run)
.venv/bin/python src/embeddings/embedding_model_benchmark.py

# 5. Tests (about 7 s)
.venv/bin/python -m pytest
#    Optional: include the 6 real-database tests (start postgres first, step 6):
#    TEST_DATABASE_URL=postgresql://<user>:<password>@127.0.0.1:5432/<db> .venv/bin/python -m pytest

# 6. Docker stack. The backend applies SQL migrations on start.
cp .env.example .env    # then set POSTGRES_PASSWORD, the same password inside DATABASE_URL, and JWT_SECRET
docker compose up --build -d --wait
docker compose exec backend python -m app.db.seed    # loads the 286 students (idempotent)
curl http://127.0.0.1:8000/health    # expect postgres and redis "reachable"
curl -X POST http://127.0.0.1:8000/requests -H 'Content-Type: application/json' \
  -d '{"actor":"student","request_kind":"resume_upload","resume_text":"Skills: Python, SQL"}'
curl http://127.0.0.1:8000/requests/<request_id from the response>
docker compose down                  # add -v to also delete the database volume
```

**Reproducibility check.** After step 3, these SHA-256 hashes must match exactly:
```
a123ed6f7d37a6adf37cbfc007c62071575b6be030545f627bcfc24b84e51520  data/processed/students_unified.csv
42c7d78b48c130c8fd7c1ee282abdec108a08f33bd8e5817586291d87a42acb1  data/processed/jobs.csv
```

---

## Phase log

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
- **Secrets:** `.env` is gitignored. `git grep -i "sk-ant"` and `git grep --untracked -E "sk-ant|AIza"` find nothing, and neither does git history. No generated secrets and no resume email appear in the container logs.

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
