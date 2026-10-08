# PROGRESS.md — Project hand-off log

**Rule:** update this file at the end of every phase. That means the status table, the "Pick up here" block, and a new entry in the Phase log. Anyone taking over should be able to read only this file and know exactly where to start.

Last updated: **2026-10-08** · Last completed phase: **Phase 1** · Next phase: **Phase 2 (not started, waiting on the inputs listed below)**

---

## ▶ Pick up here

**Phase 2 has not started.** It is waiting on these decisions and inputs from the project owner:

1. **Master spec.** The phase plan below is a proposal until it is checked against the spec. The spec text has not been shared with the AI sessions so far.
2. **`docs/ADR.md`.** It is referenced but missing from the repo. Recover it or decide to recreate it.
3. **TF-IDF baseline (`src/embeddings/tfidf_baseline.py`).** It is referenced but missing. Recover it or re-implement it in Phase 6.
4. **Layout decision.** Keep `preprocess.py` as one script, or split it into `config/schema.py` + `src/pipeline/*` + `src/parsing/*`. A split only moves code; no logic changes.
5. **LLM provider.** Anthropic (per the phase prompts) or Gemini (per `Architecture.png`). The API key is needed from Phase 3 and is not configured anywhere yet.
6. **GitHub repo visibility.** The repo is **public**: an unauthenticated request returned HTTP 200 on 2026-10-08. `data/processed/students_unified.csv` contains 120 real students' records, so the owner should switch the repo to private.

---

## Phase status

The scope of each phase comes from `PROJECT_AUDIT.md` §N. It is **proposed** and must be reconciled with the master spec.

| Phase | Scope | Status |
|---|---|---|
| 1 | Repository audit + environment fixes | ✅ Done (2026-10-08) |
| 2 | Foundations + Supervisor / 5-agent skeleton (stub tools, no LLM) | ⏸ Not started (blocked on inputs above) |
| 3 | LLM integration (Supervisor routing, Explanation narration) | Not started |
| 4 | PostgreSQL + pgvector schema and loaders | Not started |
| 5 | Data quality fixes + labelled student–job eval set | Not started |
| 6 | Embedding experiments (TF-IDF / SBERT / E5 / BGE / small local) + Model Selection Engine #1 | Not started |
| 7 | Matching features + skill-gap computation | Not started |
| 8 | Matching-model comparison (LogReg / DT / RF / GBM / XGBoost / LightGBM) + Model Selection Engine #2 | Not started |
| 9 | SHAP explainability → Explanation Agent | Not started |
| 10 | Learning Agent (courses) | Not started |
| 11 | Cohort analytics | Not started |
| 12 | FastAPI backend | Not started |
| 13 | Next.js frontend | Not started |
| 14 | Redis caching, pgvector indexes, performance | Not started |
| 15 | Feedback loop + model registry | Not started |
| 16 | QA consolidation + CI | Not started |
| 17 | Docker Compose deployment, MLOps | Not started |
| 18 | Final evaluation, fairness audit, report | Not started |

---

## Setup from scratch (verified on 2026-10-08, Apple M2 / macOS)

```bash
# 1. Python 3.11 virtual environment with pinned dependencies
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
# VS Code picks up .venv automatically via .vscode/settings.json.
# If it doesn't: Cmd+Shift+P → "Python: Select Interpreter" → .venv

# 2. Raw data → data/raw/ (gitignored; never commit these)
#    Student Data.xlsx          placement-cell export, real student data (get it from the project owner)
#    UpdatedResumeDataSet.csv   Kaggle resume dataset, 962 rows
#    job_descriptions.csv       Kaggle job dataset, 695 MB, 1,048,575 rows
#    Use the team's exact copies: the job file is a truncated version, and a
#    fresh download will NOT reproduce the processed data.
#    Alternatively, leave the files elsewhere and run with RAW_DATA_DIR=/path/to/folder

# 3. Build the processed datasets (about 6 s)
.venv/bin/python preprocess.py
#    → data/processed/students_unified.csv (286 rows), data/processed/jobs.csv (4,990 rows),
#      results/data_quality_report.md

# 4. Embedding benchmark (about 50 s; downloads about 1.25 GB of models on first run)
.venv/bin/python embedding_model_benchmark.py
#    → results/embedding_model_comparison.json + _summary.csv
```

**Reproducibility check.** After step 3, these SHA-256 hashes must match exactly:
```
a123ed6f7d37a6adf37cbfc007c62071575b6be030545f627bcfc24b84e51520  data/processed/students_unified.csv
42c7d78b48c130c8fd7c1ee282abdec108a08f33bd8e5817586291d87a42acb1  data/processed/jobs.csv
```
(`shasum -a 256 data/processed/*.csv`)

---

## Phase log

### Phase 1: Repository audit + environment fixes (2026-10-08) ✅

**Audit:** the full findings are in [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md), and the key points are summarized here.
- The repo holds only `preprocess.py`, `embedding_model_benchmark.py`, two processed CSVs and the `results/` files.
- Several files described in the phase prompt **do not exist**: ADR.md, the TF-IDF baseline, the `config/` and `src/` modules, and the READMEs.
- `preprocess.py` is deterministic. Its output was byte-identical on Python 3.9.6 and on 3.11.17.
- The SBERT / E5 / BGE benchmark runs for real, but it **measures only cost, not quality**. There are no eval labels, so `recall_at_k` is null.
- `overall_score_mean` must **not** be used to compare models. Empty sections inflate it, especially for E5 (audit I-3).
- Latency numbers depend on run order. SBERT, which runs first, always shows about 1.6 s for its first call (audit I-5).

**Fixes made after the audit (at the user's request):**

| Change | File(s) |
|---|---|
| Created a Python 3.11 venv with pinned deps. This fixes the Pylance "Import could not be resolved" warnings: VS Code was using a Python with no packages installed. | `requirements.txt`, `.vscode/settings.json`, `.venv/` (gitignored) |
| Added `.gitignore` for `.venv/`, `data/raw/`, `__pycache__/` and `.DS_Store` | `.gitignore` |
| Paths now resolve from the repo root, not the current directory. Raw data is read from `data/raw/` or `$RAW_DATA_DIR`, with a clear error if files are missing. Before this, running from the repo root crashed with `FileNotFoundError`. | `preprocess.py` |
| Removed the unused `import csv` | `preprocess.py` |
| **Bug fix (audit I-6):** the quality report counted NaN as present. Corrected counts: achievements 166→**216**, internship_raw 0→**57**, technical_projects 166→**167**, skills_raw 33→**35**, certifications 166→**173**. | `preprocess.py`, `results/data_quality_report.md` (regenerated) |
| Benchmark default paths now resolve from the repo root | `embedding_model_benchmark.py` |
| Re-ran the benchmark on Python 3.11. Scores are identical (SBERT 0.1686 / E5 0.7351 / BGE 0.4177); timings were re-measured. | `results/embedding_model_comparison*.{json,csv}` (regenerated) |
| Symlinked raw files from `~/Desktop/dataset/` into `data/raw/` (owner's machine only) | `data/raw/` (gitignored) |

**Verified:** both scripts run with exit code 0. VS Code reports 0 diagnostics in both files. Both processed CSV hashes are unchanged.

**Still open, deliberately left for later phases** (numbers refer to `PROJECT_AUDIT.md` §I):
- I-2: no eval labels; precision and F1 are not implemented. → Phase 5 / 6
- I-3: empty sections distort scores. I-4: `job_description` is double-counted. I-5: timing method is unreliable. → Phase 6
- I-7: job data looks synthetic and Excel-truncated. → must be acknowledged in the final report
- I-8: degree and skill normalization. → Phase 5
- I-9: imbalanced cohort. → Phases 8, 11 and 18
- I-12: real student data in the repo. → owner to confirm visibility

**Git:** committed on branch `phase-1-audit` and pushed to `origin`. It is not merged into `main` yet. `.DS_Store` files were removed from tracking; they are now gitignored.
