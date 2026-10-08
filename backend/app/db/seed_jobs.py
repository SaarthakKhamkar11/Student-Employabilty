"""Load data/processed/jobs.csv into the jobs table, flag near-duplicates, and optionally run the full
Job Intelligence pipeline for every job.

    docker compose exec backend python -m app.db.seed_jobs            # raw rows + duplicate flags
    docker compose exec backend python -m app.db.seed_jobs --process  # + skills, requirements, embeddings

Every row is validated through JobRecord first. Raw dataset columns are upserted; derived
columns are written only by the agent.
"""

import argparse
import asyncio
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import psycopg

from app.core.config import REPO_ROOT, get_settings
from app.core.logging import get_logger, log_event
from app.jobs.dedup import DETECTOR_VERSION, find_near_duplicates
from app.schemas.job import JobRecord
from config.schema import JOB_SCHEMA

DEFAULT_JOBS_CSV = REPO_ROOT / "data" / "processed" / "jobs.csv"
logger = get_logger("seed_jobs")


def load_job_records(csv_path: Path = DEFAULT_JOBS_CSV) -> list[JobRecord]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in JOB_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns {missing}")
    if df["job_id"].duplicated().any():
        raise ValueError(f"{csv_path} has duplicate job_id values")
    return [JobRecord.model_validate(row) for row in df[JOB_SCHEMA].to_dict("records")]


def seed_jobs(dsn: str, records: list[JobRecord]) -> dict[str, int]:
    """Upsert raw rows, then (re)compute near_duplicate_of for the whole set."""
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in JOB_SCHEMA if c != "job_id")
    sql = (f"INSERT INTO jobs ({', '.join(JOB_SCHEMA)}) VALUES ({', '.join(['%s'] * len(JOB_SCHEMA))}) "
           f"ON CONFLICT (job_id) DO UPDATE SET {updates}, updated_at = now()")
    flagged = find_near_duplicates([r.model_dump() for r in records])
    with psycopg.connect(dsn, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.executemany(sql, [[getattr(r, c) for c in JOB_SCHEMA] for r in records])
        cur.execute("UPDATE jobs SET near_duplicate_of = NULL WHERE source_dataset = 'kaggle_job_descriptions'")
        cur.executemany("UPDATE jobs SET near_duplicate_of = %s WHERE job_id = %s",
                        [(v["near_duplicate_of"], k) for k, v in flagged.items()])
    log_event(logger, "near_duplicates_flagged", detector_version=DETECTOR_VERSION, flagged=len(flagged),
              checked=len(records))
    return {"rows": len(records), "near_duplicates": len(flagged)}


async def process_all(job_ids: list[str], concurrency: int = 8) -> Counter:
    from app.agents import build_agents
    from app.schemas.agent_state import AgentState

    agent = build_agents()["job_intelligence_agent"]
    semaphore, statuses = asyncio.Semaphore(concurrency), Counter()

    async def one(job_id: str) -> None:
        async with semaphore:
            out = await agent.run(AgentState(request_id=f"seed-{job_id}", task_type="ingest_job"), {"job_id": job_id})
            statuses[out.status] += 1

    await asyncio.gather(*(one(j) for j in job_ids))
    return statuses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_JOBS_CSV)
    parser.add_argument("--process", action="store_true", help="run the Job Intelligence pipeline for every job")
    args = parser.parse_args()
    settings = get_settings()
    if not settings.database_url:
        sys.exit("DATABASE_URL is not set")
    records = load_job_records(args.csv)
    result = seed_jobs(settings.database_url.get_secret_value(), records)
    print(f"jobs: upserted {result['rows']} validated rows; near-duplicates flagged: {result['near_duplicates']}")
    if args.process:
        logging.getLogger("employability").setLevel(logging.WARNING)  # 4,990 runs: keep the log readable
        t0 = time.time()
        statuses = asyncio.run(process_all([r.job_id for r in records]))
        print(f"processed {sum(statuses.values())} jobs in {time.time() - t0:.0f}s: {dict(statuses)}")
