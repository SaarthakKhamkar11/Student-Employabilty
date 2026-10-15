"""Postgres access (psycopg 3) for the auth module.

The account tables are migration 0009's `users` (the only PII table) plus migration 0012's columns: full_name,
details, profile_confirmed_at. A student account is linked to its `student_profiles` row (users.student_id) from
signup on; a company account to its `companies` row (users.company_id). The schema is owned by the SQL migrations
(python -m app.db.migrate, run on container start), not created here.
"""
import psycopg
from psycopg.rows import dict_row

from .config import get_settings


def get_conn() -> psycopg.Connection:
    """Use as `with get_conn() as conn:` - commits on success, rolls back on exception."""
    return psycopg.connect(get_settings().database_url, row_factory=dict_row)


def init_auth_schema() -> None:
    """Kept for callers of the pre-Phase-13 module: the tables now come from migrations 0009 and 0012."""
    return None
