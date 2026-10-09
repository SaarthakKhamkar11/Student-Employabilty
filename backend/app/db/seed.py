"""Load data/processed/students_unified.csv into student_profiles (idempotent upsert).

Every row goes through StudentRecord first, so an invalid row stops the load
instead of being stored. Blank cells stay NULL; nothing is filled in.

Usage:  docker compose exec backend python -m app.db.seed             # raw rows only
        docker compose exec backend python -m app.db.seed --process   # + the Profile Agent for every student
   or:  python -m app.db.seed --csv path/to/students_unified.csv   (DATABASE_URL from env or .env)

--process builds each student's structured profile and section embeddings, which the
Matching Agent needs (Phase 7): ranking candidates for a job compares every student.
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
from app.schemas.student import StudentRecord
from config.schema import STUDENT_SCHEMA

DEFAULT_STUDENTS_CSV = REPO_ROOT / "data" / "processed" / "students_unified.csv"


def load_student_records(csv_path: Path = DEFAULT_STUDENTS_CSV) -> list[StudentRecord]:
    # dtype=str + keep_default_na=False: read cells exactly as written; StudentRecord turns "" into None
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in STUDENT_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns {missing}")
    if df["student_id"].duplicated().any():
        raise ValueError(f"{csv_path} has duplicate student_id values")
    return [StudentRecord.model_validate(row) for row in df[STUDENT_SCHEMA].to_dict("records")]


def seed_students(dsn: str, records: list[StudentRecord]) -> int:
    columns = ", ".join(STUDENT_SCHEMA)
    placeholders = ", ".join(["%s"] * len(STUDENT_SCHEMA))
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in STUDENT_SCHEMA if c != "student_id")
    sql = (f"INSERT INTO student_profiles ({columns}) VALUES ({placeholders}) "
           f"ON CONFLICT (student_id) DO UPDATE SET {updates}, updated_at = now()")
    rows = [[getattr(r, c) for c in STUDENT_SCHEMA] for r in records]
    with psycopg.connect(dsn, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


async def process_all(student_ids: list[str], concurrency: int = 8) -> Counter:
    from app.agents import build_agents
    from app.schemas.agent_state import AgentState

    agent = build_agents()["profile_agent"]
    semaphore, statuses = asyncio.Semaphore(concurrency), Counter()

    async def one(student_id: str) -> None:
        async with semaphore:
            out = await agent.run(AgentState(request_id=f"seed-{student_id}", task_type="ingest_student_profile"),
                                  {"student_id": student_id})
            statuses[out.status] += 1

    await asyncio.gather(*(one(s) for s in student_ids))
    return statuses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_STUDENTS_CSV)
    parser.add_argument("--process", action="store_true", help="run the Profile Agent for every student")
    args = parser.parse_args()
    settings = get_settings()
    if not settings.database_url:
        sys.exit("DATABASE_URL is not set")
    if not args.csv.exists():
        sys.exit(f"{args.csv} not found (in Docker it is mounted from ./data/processed)")
    records = load_student_records(args.csv)
    count = seed_students(settings.database_url.get_secret_value(), records)
    print(f"student_profiles: upserted {count} validated rows from {args.csv.name}")
    if args.process:
        logging.getLogger("employability").setLevel(logging.WARNING)  # 286 runs: keep the log readable
        t0 = time.time()
        statuses = asyncio.run(process_all([r.student_id for r in records]))
        print(f"processed {sum(statuses.values())} students in {time.time() - t0:.0f}s: {dict(statuses)}")
