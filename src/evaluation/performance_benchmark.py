"""Phase 14 Part D (spec §35): measured latency, throughput and memory at three dataset sizes (docs/adr/0021).

    python -m src.evaluation.performance_benchmark run --size real   286 real students x 4,990 real jobs
    python -m src.evaluation.performance_benchmark run --size 1k     1,000 students (286 real + 714 SYNTHETIC) x 1,000 real jobs
    python -m src.evaluation.performance_benchmark run --size 5k     5,000 students (286 real + 4,714 SYNTHETIC) x the same 1,000 jobs
    python -m src.evaluation.performance_benchmark report            results/performance_benchmark.{md,json}
    python -m src.evaluation.performance_benchmark drop              drop the throwaway benchmark databases

Run inside the backend image with DATABASE_URL pointing at the project's Postgres. Nothing touches that database's
data: each size runs against its own throwaway database on the same server (phase14_bench_real,
phase14_bench_scaled; 5k extends the 1k database), created with the project's own migrations.

What runs is the production code, unchanged: the Profile and Job Intelligence agents ingest every student and job
(as `seed --process` does), then sampled requests go through the Supervisor exactly as the portal sends them. Every
stage latency is read from the Phase 12 trace spans the runtime records while it runs (app/core/tracing.py), never
from timers added for the benchmark. Two deliberate differences from the live service, both stated in the report:
no LLM (routing is deterministic for an explicit task, and explanations use the validated structured fallback, as
on every dashboard page load) and no YouTube API (no external quota is spent: the Learning Agent uses the curated
catalogue only). Synthetic students come from src/pipeline/synthetic_scaleup.py and are flagged per size.
"""

import argparse
import asyncio
import json
import os
import platform
import resource
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "performance"
STUDENTS_CSV = ROOT / "data" / "processed" / "students_unified.csv"
JOBS_CSV = ROOT / "data" / "processed" / "jobs.csv"
CATALOG = ROOT / "data" / "learning" / "course_catalog.json"
DB_PREFIX = "phase14_bench_"
SEED = 14
SIZES = {
    "real": {"students": None, "jobs": None, "db": "phase14_bench_real"},
    "1k": {"students": 1000, "jobs": 1000, "db": "phase14_bench_scaled"},
    "5k": {"students": 5000, "jobs": 1000, "db": "phase14_bench_scaled"},
    "smoke": {"students": 12, "jobs": 40, "db": "phase14_bench_smoke"},  # a quick check of the harness; not reported
}
REQUESTS = {"student": 20, "upload": 10, "company": 10}  # sampled Supervisor requests per size

# spec §35 stage -> the tool spans that implement it (summed per entity / request)
STAGES = {
    "resume_parsing_record": ("ingest_student", ("parse_student_sections",)),
    "resume_parsing_upload": ("upload", ("extract_text", "parse_resume")),
    "jd_parsing": ("ingest_job", ("parse_job_description", "extract_requirements", "extract_job_posting")),
    "skill_extraction_student": ("ingest_student", ("extract_skills", "normalize_skills")),
    "skill_extraction_job": ("ingest_job", ("extract_job_skills", "normalize_skills", "classify_skill_requirements")),
    "embedding_student": ("ingest_student", ("generate_embedding",)),
    "embedding_job": ("ingest_job", ("generate_embedding",)),
    "similarity_jobs_for_student": ("student", ("retrieve_jobs", "compute_similarity")),
    "similarity_candidates_for_job": ("company", ("retrieve_candidates", "compute_similarity")),
    "prediction_jobs_for_student": ("student", ("generate_features", "predict_suitability")),
    "prediction_candidates_for_job": ("company", ("generate_features", "predict_suitability")),
    "ranking_jobs_for_student": ("student", ("rank_candidates",)),
    "ranking_candidates_for_job": ("company", ("rank_candidates",)),
}
AGENT_STAGES = {"explanation_student": ("student", "explainability_agent"),
                "explanation_company": ("company", "explainability_agent"),
                "matching_agent_student": ("student", "matching_agent"),
                "matching_agent_company": ("company", "matching_agent"),
                "learning_agent_student": ("student", "learning_agent")}


def _dsn_for(dsn: str, db: str) -> str:
    parts = urlsplit(dsn)
    return urlunsplit(parts._replace(path=f"/{db}"))


def _rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def _peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    return {"n": len(v), "median": round(statistics.median(v), 2), "p95": round(float(np.percentile(v, 95)), 2),
            "mean": round(statistics.fmean(v), 2), "min": round(v[0], 2), "max": round(v[-1], 2)}


# ---------------------------------------------------------------- database

def create_database(admin_dsn: str, db: str) -> None:
    import psycopg

    if not db.startswith(DB_PREFIX):
        raise ValueError(f"refusing to (re)create {db!r}: benchmark databases start with {DB_PREFIX}")
    with psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{db}"')
    with psycopg.connect(_dsn_for(admin_dsn, db), autocommit=True, connect_timeout=5) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


def drop_databases(admin_dsn: str) -> list[str]:
    import psycopg

    dropped = []
    with psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5) as conn:
        for (name,) in conn.execute("SELECT datname FROM pg_database WHERE datname LIKE %s", (DB_PREFIX + "%",)).fetchall():
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            dropped.append(name)
    return dropped


def db_facts(dsn: str) -> dict:
    import psycopg

    with psycopg.connect(dsn, connect_timeout=5) as conn:
        q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
        return {"database_size_mb": round(q("SELECT pg_database_size(current_database())") / 1024**2, 1),
                "embeddings_size_mb": round(q("SELECT pg_total_relation_size('embeddings')") / 1024**2, 1),
                "students": q("SELECT count(*) FROM student_profiles"),
                "students_with_profile": q("SELECT count(*) FROM student_profiles WHERE profile IS NOT NULL"),
                "jobs": q("SELECT count(*) FROM jobs"), "jobs_with_posting": q("SELECT count(*) FROM jobs WHERE posting IS NOT NULL"),
                "embedding_rows": q("SELECT count(*) FROM embeddings")}


# ---------------------------------------------------------------- data

def size_data(size: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.pipeline.synthetic_scaleup import sample_jobs, scale_students

    spec = SIZES[size]
    students = pd.read_csv(STUDENTS_CSV, dtype=str, keep_default_na=False)
    jobs = pd.read_csv(JOBS_CSV, dtype=str, keep_default_na=False)
    if spec["students"]:
        students = scale_students(students, spec["students"], SEED)
    if spec["jobs"]:
        jobs = sample_jobs(jobs, spec["jobs"], SEED)
    return students, jobs


def resume_text(row: dict) -> str:
    """A plain-text resume composed from a stored row, with the headings the resume parser looks for. Used only to
    exercise the upload path (extract_text -> parse_resume) on realistic content; flagged in the report."""
    parts = [("Education", f"{row.get('degree', '')} {row.get('branch', '')}".strip()),
             ("Skills", (row.get("skills_raw") or "").replace("; ", "\n")),
             ("Experience", row.get("internship_raw") or ""), ("Projects", row.get("technical_projects") or ""),
             ("Certifications", row.get("certifications") or ""), ("Achievements", row.get("achievements") or "")]
    return "\n\n".join(f"{h}\n{t}" for h, t in parts if t.strip())


# ---------------------------------------------------------------- measurement

async def ingest(agent, ids: list[str], task: str, key: str) -> dict:
    """One agent run per entity, sequentially (one worker), each inside its own trace recorder: the per-tool spans
    are the runtime's own (BaseAgent / registry), not benchmark timers."""
    from app.core.tracing import TraceRecorder, trace_var
    from app.schemas.agent_state import AgentState

    per_tool: dict[str, list[float]] = {}
    totals, statuses, t0 = [], {}, time.perf_counter()
    for i, entity in enumerate(ids, start=1):
        trace = TraceRecorder(f"bench-{entity}")
        token = trace_var.set(trace)
        try:
            out = await agent.run(AgentState(request_id=trace.request_id, task_type=task), {key: entity})
        finally:
            trace_var.reset(token)
        statuses[out.status] = statuses.get(out.status, 0) + 1
        sums: dict[str, float] = {}
        for span in trace.spans:
            if span.kind == "tool" and span.duration_ms is not None:
                sums[span.name] = sums.get(span.name, 0.0) + span.duration_ms
            if span.kind == "agent" and span.parent_id is None and span.duration_ms is not None:
                totals.append(span.duration_ms)
        for name, ms in sums.items():
            per_tool.setdefault(name, []).append(ms)
        if i % 250 == 0 or i == len(ids):
            print(f"    {task}: {i}/{len(ids)} ({time.perf_counter() - t0:.0f} s)", flush=True)
    wall = time.perf_counter() - t0
    return {"entities": len(ids), "wall_seconds": round(wall, 1),
            "throughput_per_s": round(len(ids) / wall, 2) if wall else None, "statuses": statuses,
            "agent_ms": _summary(totals), "per_tool_ms": {k: v for k, v in per_tool.items()}}


def bulk_similarity(dsn: str, model_name: str) -> dict:
    """All students x all jobs on the stored 'full' E5 vectors: the dense similarity matrix and a top-50 per student,
    in numpy (median of 5). This is the all-pairs cost; requests instead score one student against pgvector."""
    import psycopg

    def fetch(entity: str) -> np.ndarray:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            rows = conn.execute("SELECT embedding::real[] FROM embeddings WHERE entity_type = %s AND section = 'full' "
                                "AND model_name = %s AND embedding IS NOT NULL ORDER BY entity_id",
                                (entity, model_name)).fetchall()
        return np.asarray([r[0] for r in rows], dtype=np.float32)

    t = time.perf_counter()
    S, J = fetch("student"), fetch("job")
    load_s = time.perf_counter() - t
    rss0 = _rss_mb()
    runs, top = [], []
    for _ in range(5):
        t = time.perf_counter()
        M = S @ J.T
        runs.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        k = min(50, M.shape[1] - 1)
        np.argpartition(-M, k, axis=1)[:, :k]
        top.append((time.perf_counter() - t) * 1000)
    pairs = S.shape[0] * J.shape[0]
    return {"students": int(S.shape[0]), "jobs": int(J.shape[0]), "pairs": int(pairs), "dimension": int(S.shape[1]),
            "fetch_vectors_seconds": round(load_s, 2), "matrix_ms": _summary(runs), "top50_ms": _summary(top),
            "pairs_per_second": round(pairs / (statistics.median(runs) / 1000)),
            "vectors_mb": round((S.nbytes + J.nbytes) / 1024**2, 1), "matrix_mb": round(M.nbytes / 1024**2, 1),
            "rss_added_mb": round(_rss_mb() - rss0, 1)}


async def requests(supervisor, store, kind: str, items: list) -> dict:
    """Sampled Supervisor requests, one at a time, as the portal sends them; per-agent and per-tool durations are
    read back from the persisted trace (get_agent_trace)."""
    from app.core.trace_tree import get_agent_trace
    from app.schemas.agent_io import SupervisorInput

    out = {"requests": [], "per_tool_ms": {}, "per_agent_ms": {}}
    t_all = time.perf_counter()
    for item in items:
        if kind == "student":
            req = SupervisorInput(actor="student", request_kind="question", task_type="recommend_jobs_for_student",
                                  request_text="dashboard: recommend_jobs_for_student", student_id=item)
        elif kind == "upload":
            req = SupervisorInput(actor="student", request_kind="resume_upload", task_type="ingest_student_profile",
                                  resume_text=item)
        else:
            req = SupervisorInput(actor="company", request_kind="question", task_type="rank_candidates_for_job",
                                  request_text="dashboard: rank_candidates_for_job", job_id=item)
        t = time.perf_counter()
        resp = await supervisor.handle(req)
        wall = (time.perf_counter() - t) * 1000
        tree = await get_agent_trace(resp.request_id, store)
        rec = {"status": resp.status, "wall_ms": round(wall, 1), "trace_ms": tree.duration_ms if tree else None,
               "agents": {}, "skipped": []}
        if tree:
            for node in tree.root.children:
                if node.kind == "agent":
                    rec["agents"][node.name] = {"ms": node.duration_ms, "status": node.status}
                    out["per_agent_ms"].setdefault(node.name, []).append(node.duration_ms)
                    sums: dict[str, float] = {}
                    for child in node.children:
                        if child.kind == "tool":
                            sums[child.name] = sums.get(child.name, 0.0) + child.duration_ms
                    for name, ms in sums.items():
                        out["per_tool_ms"].setdefault(name, []).append(ms)
                elif node.kind == "skipped":
                    rec["skipped"].append(node.name)
        out["requests"].append(rec)
        agents = " ".join(f"{a}={v['status']}" for a, v in rec["agents"].items())
        print(f"    {kind}: {len(out['requests'])}/{len(items)} {resp.status} {wall:.0f} ms  {agents}"
              f"{'  errors: ' + str([e.model_dump() if hasattr(e, 'model_dump') else e for e in resp.errors][:2]) if resp.errors else ''}",
              flush=True)
    out["wall_seconds"] = round(time.perf_counter() - t_all, 1)
    return out


# ---------------------------------------------------------------- one size

async def run_size(size: str, admin_dsn: str) -> dict:
    spec = SIZES[size]
    dsn = _dsn_for(admin_dsn, spec["db"])
    # the agents read their settings from the environment: point them at the throwaway database, with no LLM and no
    # YouTube quota, BEFORE anything builds Settings
    os.environ.update(DATABASE_URL=dsn, YOUTUBE_API_KEY="", GEMINI_API_KEY="")
    from app.agents import build_agents
    from app.core.config import get_settings
    from app.db.migrate import apply_migrations
    from app.db.seed import seed_students
    from app.db.seed_jobs import seed_jobs
    from app.db.store import build_store
    from app.learning.catalog import load as load_catalog
    from app.schemas.job import JobRecord
    from app.schemas.student import StudentRecord
    from config.schema import JOB_SCHEMA, STUDENT_SCHEMA

    get_settings.cache_clear()
    settings = get_settings()
    assert settings.database_url.get_secret_value() == dsn and dsn.rsplit("/", 1)[1].startswith(DB_PREFIX)
    started = datetime.now(timezone.utc)
    rss_start = _rss_mb()
    students, jobs = size_data(size)
    result = {"size": size, "started_at": started.isoformat(timespec="seconds"), "database": spec["db"],
              "students": int(len(students)), "real_students": int((students["is_synthetic"] != "True").sum()),
              "synthetic_students": int((students["is_synthetic"] == "True").sum()), "jobs": int(len(jobs)),
              "jobs_are_real": True}

    import psycopg

    existing_students, existing_jobs = set(), set()
    if size == "5k":
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            existing_students = {r[0] for r in conn.execute("SELECT student_id FROM student_profiles WHERE profile IS NOT NULL")}
            existing_jobs = {r[0] for r in conn.execute("SELECT job_id FROM jobs WHERE posting IS NOT NULL")}
        # upload requests add their own student rows, so check that the 1k set is there rather than an exact count
        first_1k = set(students["student_id"].iloc[:SIZES["1k"]["students"]])
        if not first_1k <= existing_students or len(existing_jobs) != SIZES["1k"]["jobs"]:
            raise SystemExit("--size 5k extends the 1k database: run --size 1k first")
    else:
        create_database(admin_dsn, spec["db"])
        apply_migrations(dsn)
    t = time.perf_counter()
    seed_students(dsn, [StudentRecord.model_validate(r) for r in students[STUDENT_SCHEMA].to_dict("records")])
    seed_counts = seed_jobs(dsn, [JobRecord.model_validate(r) for r in jobs[JOB_SCHEMA].to_dict("records")])
    load_catalog(dsn, json.loads(CATALOG.read_text(encoding="utf-8")))
    result["raw_rows_seconds"] = round(time.perf_counter() - t, 1)
    result["job_seed"] = seed_counts

    agents = build_agents(llm=None, store=build_store(settings))
    store = build_store(settings)
    new_jobs = [j for j in jobs["job_id"] if j not in existing_jobs]
    new_students = [s for s in students["student_id"] if s not in existing_students]
    print(f"  ingest {len(new_jobs)} jobs, {len(new_students)} students", flush=True)
    rss = _rss_mb()
    result["ingest_jobs"] = await ingest(agents["job_intelligence_agent"], new_jobs, "ingest_job", "job_id")
    result["ingest_students"] = await ingest(agents["profile_agent"], new_students, "ingest_student_profile", "student_id")
    result["rss_after_ingest_mb"] = round(_rss_mb(), 1)
    result["rss_ingest_added_mb"] = round(_rss_mb() - rss, 1)

    print("  bulk similarity", flush=True)
    result["bulk_similarity"] = bulk_similarity(dsn, settings.embedding_model)

    rng = np.random.default_rng(SEED)
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        profiled = sorted(r[0] for r in conn.execute("SELECT student_id FROM student_profiles WHERE profile IS NOT NULL "
                                                     "AND student_id = ANY(%s)", (list(students["student_id"]),)))
        canonical_jobs = sorted(r[0] for r in conn.execute("SELECT job_id FROM jobs WHERE posting IS NOT NULL "
                                                           "AND near_duplicate_of IS NULL"))
    s_pick = sorted(rng.choice(profiled, size=min(REQUESTS["student"], len(profiled)), replace=False).tolist())
    u_pick = sorted(rng.choice(len(students), size=REQUESTS["upload"], replace=False).tolist())
    j_pick = sorted(rng.choice(canonical_jobs, size=min(REQUESTS["company"], len(canonical_jobs)), replace=False).tolist())
    synthetic = set(students.loc[students["is_synthetic"] == "True", "student_id"])
    result["samples"] = {"student_requests": s_pick, "synthetic_in_student_sample": sum(s in synthetic for s in s_pick),
                         "upload_rows": [students.iloc[i]["student_id"] for i in u_pick],
                         "synthetic_in_upload_sample": sum(students.iloc[i]["student_id"] in synthetic for i in u_pick),
                         "company_requests": j_pick, "seed": SEED}
    counts = {k: 1 for k in REQUESTS} if size == "smoke" else REQUESTS
    s_pick, u_pick, j_pick = s_pick[:counts["student"]], u_pick[:counts["upload"]], j_pick[:counts["company"]]
    sup = agents["supervisor"]
    print("  requests", flush=True)
    rss = _rss_mb()
    result["student_requests"] = await requests(sup, store, "student", s_pick)
    result["upload_requests"] = await requests(sup, store, "upload",
                                               [resume_text(students.iloc[i].to_dict()) for i in u_pick])
    result["company_requests"] = await requests(sup, store, "company", j_pick)
    result["rss_requests_added_mb"] = round(_rss_mb() - rss, 1)

    result["database_facts"] = db_facts(dsn)
    result["memory"] = {"rss_start_mb": round(rss_start, 1), "rss_end_mb": round(_rss_mb(), 1),
                        "peak_rss_mb": round(_peak_mb(), 1)}
    result["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["environment"] = environment(settings)
    return result


def environment(settings) -> dict:
    import torch

    with open("/proc/meminfo") as fh:
        total_kb = next(int(line.split()[1]) for line in fh if line.startswith("MemTotal:"))
    return {"python": platform.python_version(), "system": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(), "logical_cpus": os.cpu_count(),
            "ram_gb_visible": round(total_kb / 1024**2, 1), "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "embedding_model": settings.embedding_model,
            "llm": "none (deterministic routing for explicit tasks; structured explanation fallback)",
            "youtube": "disabled (curated catalogue only)", "ingest_workers": 1, "request_concurrency": 1}


# ---------------------------------------------------------------- report

def stage_table(r: dict) -> dict:
    """spec §35 stages for one size: per-entity / per-request sums of the implementing tool spans."""
    sources = {"ingest_student": r["ingest_students"]["per_tool_ms"], "ingest_job": r["ingest_jobs"]["per_tool_ms"],
               "student": r["student_requests"]["per_tool_ms"], "company": r["company_requests"]["per_tool_ms"],
               "upload": r["upload_requests"]["per_tool_ms"]}
    out = {}
    for stage, (src, tools) in STAGES.items():
        lists = [sources[src].get(t, []) for t in tools]
        n = min((len(x) for x in lists if x), default=0)
        out[stage] = _summary([sum(x[i] for x in lists if len(x) > i) for i in range(n)]) | {"tools": list(tools)}
    agents = {"student": r["student_requests"]["per_agent_ms"], "company": r["company_requests"]["per_agent_ms"]}
    for stage, (src, agent) in AGENT_STAGES.items():
        out[stage] = _summary(agents[src].get(agent, [])) | {"agent": agent}
    for kind in ("student", "upload", "company"):
        reqs = r[f"{kind}_requests"]["requests"]
        out[f"end_to_end_{kind}"] = _summary([x["trace_ms"] for x in reqs if x["trace_ms"] is not None]) | {
            "statuses": {s: sum(1 for x in reqs if x["status"] == s) for s in {x["status"] for x in reqs}}}
    return out


LABELS = {
    "resume_parsing_record": "Resume parsing: stored record (parse_student_sections)",
    "resume_parsing_upload": "Resume parsing: uploaded text (extract_text + parse_resume)",
    "jd_parsing": "JD parsing (parse_job_description + extract_requirements + extract_job_posting)",
    "skill_extraction_student": "Skill extraction, student (extract_skills + normalize_skills)",
    "skill_extraction_job": "Skill extraction, job (extract_job_skills + normalize_skills + classify_skill_requirements)",
    "embedding_student": "Embedding generation, student (generate_embedding: all sections, E5 + TF-IDF fallback)",
    "embedding_job": "Embedding generation, job (generate_embedding)",
    "similarity_jobs_for_student": "Similarity, one student vs all eligible jobs (retrieve_jobs + compute_similarity)",
    "similarity_candidates_for_job": "Similarity, one job vs all students (retrieve_candidates + compute_similarity)",
    "prediction_jobs_for_student": "Prediction, top-50 jobs (generate_features + predict_suitability)",
    "prediction_candidates_for_job": "Prediction, top-50 students (generate_features + predict_suitability)",
    "ranking_jobs_for_student": "Ranking (rank_candidates), student request",
    "ranking_candidates_for_job": "Ranking (rank_candidates), company request",
    "matching_agent_student": "Matching Agent total, student request (Stages 1-6 + cohort benchmark)",
    "matching_agent_company": "Matching Agent total, company request",
    "learning_agent_student": "Learning Agent total, student request",
    "explanation_student": "Explanation generation (Explainability Agent), student request",
    "explanation_company": "Explanation generation (Explainability Agent), company request",
    "end_to_end_student": "End to end: student job recommendations (Supervisor request)",
    "end_to_end_upload": "End to end: resume upload (Supervisor request)",
    "end_to_end_company": "End to end: company candidate ranking (Supervisor request)",
}


def _cell(s: dict) -> str:
    return "—" if not s.get("n") else f"{s['median']:,.1f} / {s['p95']:,.1f} (n={s['n']})"


def report() -> dict:
    runs = {s: json.loads((OUT / f"{s}.json").read_text(encoding="utf-8")) for s in SIZES
            if s != "smoke" and (OUT / f"{s}.json").exists()}
    if not runs:
        raise SystemExit("no results/performance/<size>.json yet: run a size first")
    tables = {s: stage_table(r) for s, r in runs.items()}

    def head(s):
        r = runs[s]
        flag = "real" if not r["synthetic_students"] else f"**{r['synthetic_students']:,} synthetic**"
        return f"{r['students']:,} students ({flag}) x {r['jobs']:,} real jobs"

    sizes = list(runs)
    out = ["# Performance benchmark (spec §35)", "",
           "Generated by `src/evaluation/performance_benchmark.py`. Every number is measured on this machine, at the "
           "sizes actually run; nothing is extrapolated. Latency cells: median / p95 in ms (n = entities or requests "
           "measured).", "",
           "| Size | Data | Synthetic? |", "|---|---|---|"]
    for s in sizes:
        r = runs[s]
        out.append(f"| `{s}` | {head(s)} | " + ("no: all real" if not r["synthetic_students"] else
                   f"**PARTIALLY SYNTHETIC**: {r['real_students']} real + {r['synthetic_students']:,} synthetic students "
                   "(src/pipeline/synthetic_scaleup.py, is_synthetic=True); jobs are a seeded sample of real postings") + " |")
    out += ["", "## Latency per stage", "", "| Stage | " + " | ".join(f"`{s}`" for s in sizes) + " |",
            "|---|" + "---|" * len(sizes)]
    for stage, label in LABELS.items():
        out.append(f"| {label} | " + " | ".join(_cell(tables[s].get(stage, {})) for s in sizes) + " |")
    out += ["", "## Throughput and memory", "", "| Measure | " + " | ".join(f"`{s}`" for s in sizes) + " |",
            "|---|" + "---|" * len(sizes)]

    def row(label, fn):
        cells = []
        for s in sizes:
            try:
                cells.append(fn(runs[s]))
            except (KeyError, TypeError, ZeroDivisionError):
                cells.append("—")
        out.append(f"| {label} | " + " | ".join(cells) + " |")

    row("Job ingestion (Job Intelligence Agent, 1 worker)", lambda r: f"{r['ingest_jobs']['entities']:,} jobs in "
        f"{r['ingest_jobs']['wall_seconds']:,} s = {r['ingest_jobs']['throughput_per_s']} jobs/s" if r["ingest_jobs"]["entities"]
        else "reused from `1k` (same 1,000 jobs)")
    row("Student ingestion (Profile Agent, 1 worker)", lambda r: f"{r['ingest_students']['entities']:,} students in "
        f"{r['ingest_students']['wall_seconds']:,} s = {r['ingest_students']['throughput_per_s']} students/s")
    row("Bulk similarity, all students x all jobs (numpy, 768-d)", lambda r: f"{r['bulk_similarity']['pairs']:,} pairs: "
        f"{r['bulk_similarity']['matrix_ms']['median']:,.1f} ms ({r['bulk_similarity']['pairs_per_second']:,} pairs/s); "
        f"top-50 {r['bulk_similarity']['top50_ms']['median']:,.1f} ms")
    for kind, label in (("student", "Student requests, sequential"), ("upload", "Resume uploads, sequential"),
                        ("company", "Company requests, sequential")):
        row(label, lambda r, k=kind: f"{len(r[f'{k}_requests']['requests'])} in {r[f'{k}_requests']['wall_seconds']} s = "
            f"{60 * len(r[f'{k}_requests']['requests']) / r[f'{k}_requests']['wall_seconds']:.1f} per minute")
    row("Benchmark process peak RSS", lambda r: f"{r['memory']['peak_rss_mb']:,.0f} MB")
    row("RSS added by ingestion / by requests", lambda r: f"{r['rss_ingest_added_mb']:,.0f} / {r['rss_requests_added_mb']:,.0f} MB")
    row("Dense similarity matrix + vectors in memory", lambda r: f"{r['bulk_similarity']['matrix_mb']:,} + "
        f"{r['bulk_similarity']['vectors_mb']:,} MB")
    row("Database size (all tables) / embeddings table", lambda r: f"{r['database_facts']['database_size_mb']:,} / "
        f"{r['database_facts']['embeddings_size_mb']:,} MB")
    row("Embedding rows stored (E5 + TF-IDF fallback)", lambda r: f"{r['database_facts']['embedding_rows']:,}")
    row("Request statuses (student / upload / company)", lambda r: " / ".join(
        str(tables_for(r)[f"end_to_end_{k}"].get("statuses", {})) for k in ("student", "upload", "company")))
    row("Synthetic students in the request samples (student / upload)",
        lambda r: f"{r['samples']['synthetic_in_student_sample']} of {len(r['samples']['student_requests'])} / "
                  f"{r['samples']['synthetic_in_upload_sample']} of {len(r['samples']['upload_rows'])}")
    env = next(iter(runs.values()))["environment"]
    out += ["", "## Conditions", "",
            f"- Machine: {env['system']} ({env['machine']}), {env['logical_cpus']} logical CPUs, {env['ram_gb_visible']} GB "
            f"visible to the container, Python {env['python']}, torch {env['torch']} ({env['torch_threads']} threads), "
            f"embedding model `{env['embedding_model']}` on CPU. Absolute numbers are specific to this machine; the "
            "relative order and the growth between sizes are what transfer.",
            f"- Production code, unchanged, against a throwaway Postgres + pgvector database per size (the live data "
            f"untouched). Ingestion: one worker ({env['ingest_workers']}), every entity. Requests: "
            f"{REQUESTS['student']} student, {REQUESTS['upload']} upload and {REQUESTS['company']} company requests per "
            "size, seeded samples, one at a time, through the Supervisor as the portal sends them.",
            f"- LLM: {env['llm']}. YouTube: {env['youtube']}. The live service adds Gemini calls (routing for free "
            "questions, explanation prose) and YouTube searches, which this benchmark does not measure.",
            "- Every stage latency comes from the Phase 12 trace spans the runtime records (agent, tool), summed per "
            "entity or request for stages made of several tools. A tool retry counts in the sum.",
            "- Upload requests use plain-text resumes composed from stored rows (headings + the row's fields), so they "
            "exercise extract_text and parse_resume on realistic content but are not real uploaded files (no PDF "
            "extraction is measured).",
            "- The E5 similarity calibration and the trained matching model are the production artifacts built on the "
            "real 286 x 4,990 data; they are reused unchanged at every size.",
            f"- **Tested ceiling: {max(runs[s]['students'] for s in sizes):,} students x "
            f"{max(runs[s]['jobs'] for s in sizes if runs[s]['students'] == max(runs[x]['students'] for x in sizes)):,} jobs** "
            f"(the largest size run; the largest job corpus run is {max(runs[s]['jobs'] for s in sizes):,} jobs, at "
            f"{min(runs[s]['students'] for s in sizes):,} real students). No claim is made beyond these.",
            "- Bulk similarity is one numpy matrix product (median of 5) in a process that also holds the E5 model; at "
            "these sizes it takes milliseconds and its run-to-run noise exceeds the size effect, so read it as an "
            "order of magnitude only.",
            f"- Orchestration overhead (Supervisor request time minus the agents' critical path, the slowest agent of each "
            "concurrent wave; median per request): "
            + "; ".join(f"`{s}` {overhead(runs[s])}" for s in sizes) + ".", ""]
    md = "\n".join(out)
    (ROOT / "results" / "performance_benchmark.md").write_text(md, encoding="utf-8")
    (ROOT / "results" / "performance_benchmark.json").write_text(
        json.dumps({"stages": tables, "runs": {s: {k: v for k, v in r.items() if k not in ("ingest_jobs", "ingest_students")}
                                                | {"ingest_jobs": _slim(r["ingest_jobs"]), "ingest_students": _slim(r["ingest_students"])}
                                                for s, r in runs.items()}}, indent=1, default=str), encoding="utf-8")
    return {"markdown": md}


WAVES = (("profile_agent", "job_intelligence_agent"), ("matching_agent",), ("learning_agent", "explainability_agent"))


def critical_path(agents: dict) -> float:
    """Agents in one Supervisor wave run concurrently (app/agents/supervisor.py), so the time inside agents is the sum
    over waves of the slowest agent in each wave, not the sum of all agents."""
    return sum(max((agents[a]["ms"] for a in wave if a in agents), default=0.0) for wave in WAVES)


def overhead(r: dict) -> str:
    """Request time spent outside every agent: routing, planning, input building, validation, persistence."""
    parts = []
    for kind in ("student", "upload", "company"):
        gaps = [x["trace_ms"] - critical_path(x["agents"]) for x in r[f"{kind}_requests"]["requests"]
                if x["trace_ms"] is not None and x["agents"]]
        parts.append(f"{kind} {statistics.median(gaps):.0f} ms" if gaps else f"{kind} —")
    return ", ".join(parts)


def tables_for(r: dict) -> dict:
    return stage_table(r)


def _slim(ingest_result: dict) -> dict:
    return {k: v for k, v in ingest_result.items() if k != "per_tool_ms"} | {
        "per_tool_ms": {t: _summary(v) for t, v in ingest_result["per_tool_ms"].items()}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Phase 14 Part D: performance at three dataset sizes")
    ap.add_argument("command", choices=["run", "report", "drop"])
    ap.add_argument("--size", choices=list(SIZES))
    args = ap.parse_args(argv)
    admin = os.environ.get("BENCH_ADMIN_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if args.command == "report":
        print(report()["markdown"])
        return
    if not admin:
        raise SystemExit("DATABASE_URL (the project's Postgres) is not set")
    if args.command == "drop":
        print("dropped", drop_databases(admin))
        return
    if not args.size:
        raise SystemExit("run needs --size")
    os.environ["BENCH_ADMIN_DATABASE_URL"] = admin
    result = asyncio.run(run_size(args.size, admin))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.size}.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"written {OUT / (args.size + '.json')}")


if __name__ == "__main__":
    main()
