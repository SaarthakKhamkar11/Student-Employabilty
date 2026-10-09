"""Load the labelling pool CSV into relevance_pool (idempotent).

    docker compose exec backend python -m app.labelling.load_pool
    python -m app.labelling.load_pool --csv path/to/labelling_pool.csv   (DATABASE_URL from env or .env)

Every student and job in the pool must already be in Postgres (app.db.seed / app.db.seed_jobs),
because the labelling page shows them from there.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg

from app.core.config import REPO_ROOT, get_settings
from app.labelling.repository import PostgresLabellingRepository

DEFAULT_POOL_CSV = REPO_ROOT / "data" / "benchmark" / "labelling_pool.csv"
COLUMNS = ["pool_version", "student_id", "job_id", "split", "display_order"]


def read_pool(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns {missing}")
    if df[["pool_version", "student_id", "job_id"]].duplicated().any():
        raise ValueError(f"{csv_path} has duplicate pairs")
    df["display_order"] = df["display_order"].astype(int)
    return df[COLUMNS].to_dict("records")


def missing_entities(dsn: str, rows: list[dict]) -> dict[str, list[str]]:
    students = sorted({r["student_id"] for r in rows})
    jobs = sorted({r["job_id"] for r in rows})
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        have_s = {r[0] for r in conn.execute("SELECT student_id FROM student_profiles WHERE student_id = ANY(%s)", (students,))}
        have_j = {r[0] for r in conn.execute("SELECT job_id FROM jobs WHERE job_id = ANY(%s)", (jobs,))}
    return {"students": [s for s in students if s not in have_s], "jobs": [j for j in jobs if j not in have_j]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_POOL_CSV)
    args = parser.parse_args()
    settings = get_settings()
    if not settings.database_url:
        sys.exit("DATABASE_URL is not set")
    if not args.csv.exists():
        sys.exit(f"{args.csv} not found (in Docker it is mounted from ./data/benchmark)")
    dsn = settings.database_url.get_secret_value()
    rows = read_pool(args.csv)
    gaps = missing_entities(dsn, rows)
    if gaps["students"] or gaps["jobs"]:
        sys.exit(f"not in Postgres yet (run app.db.seed / app.db.seed_jobs first): {gaps}")
    PostgresLabellingRepository(dsn).load_pool(rows)
    versions = sorted({r["pool_version"] for r in rows})
    print(f"relevance_pool: {len(rows)} pairs, {len({r['student_id'] for r in rows})} students, pool {versions}")
