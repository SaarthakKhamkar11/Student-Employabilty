"""Postgres access (psycopg 3) and the auth tables."""
import psycopg
from psycopg.rows import dict_row

from .config import get_settings

# app_users is named this way so it can't clash with any existing "users"/"students" table.
# student_accounts.student_id is left NULL at signup: it is where you link the account to the
# project-assigned student_id used by the matching pipeline, once that record is created.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS app_users (
    user_id        BIGSERIAL PRIMARY KEY,
    email          TEXT        NOT NULL,
    password_hash  TEXT        NOT NULL,
    role           TEXT        NOT NULL CHECK (role IN ('student', 'recruiter')),
    full_name      TEXT        NOT NULL,
    is_active      BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at  TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS app_users_email_lower_uq ON app_users (lower(email));

CREATE TABLE IF NOT EXISTS student_accounts (
    user_id          BIGINT PRIMARY KEY REFERENCES app_users(user_id) ON DELETE CASCADE,
    student_id       TEXT UNIQUE,
    degree           TEXT NOT NULL,
    branch           TEXT NOT NULL,
    graduation_year  INT  NOT NULL CHECK (graduation_year BETWEEN 2000 AND 2100),
    gender           TEXT NOT NULL CHECK (gender IN ('male', 'female', 'other', 'prefer_not_to_say'))
);

CREATE TABLE IF NOT EXISTS recruiter_accounts (
    user_id       BIGINT PRIMARY KEY REFERENCES app_users(user_id) ON DELETE CASCADE,
    company_name  TEXT NOT NULL,
    designation   TEXT
);
"""


def get_conn() -> psycopg.Connection:
    """Use as `with get_conn() as conn:` - commits on success, rolls back on exception."""
    return psycopg.connect(get_settings().database_url, row_factory=dict_row)


def init_auth_schema() -> None:
    """Create the auth tables if they don't exist. Safe to call on every startup."""
    with get_conn() as conn:
        conn.execute(SCHEMA_SQL)
