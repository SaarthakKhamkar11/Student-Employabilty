"""Apply backend/migrations/*.sql in filename order, each exactly once.

Usage: python -m app.db.migrate   (reads DATABASE_URL from the environment or .env)
"""

import sys
from pathlib import Path

import psycopg

from app.core.config import get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def apply_migrations(dsn: str, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Return the migrations applied by this call (already-applied ones are skipped)."""
    applied = []
    # autocommit, so each `conn.transaction()` below is a real transaction per migration file
    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
                            version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())""")
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        for path in sorted(migrations_dir.glob("*.sql")):
            if path.name in done:
                continue
            with conn.transaction():
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.name,))
            applied.append(path.name)
    return applied


if __name__ == "__main__":
    settings = get_settings()
    if not settings.database_url:
        sys.exit("DATABASE_URL is not set")
    names = apply_migrations(settings.database_url.get_secret_value())
    print(f"migrations applied: {names or 'none (up to date)'}")
