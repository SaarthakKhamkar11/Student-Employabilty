"""Create (or reset the password of) the admin/research account from environment variables.

    docker compose exec -e ADMIN_EMAIL=... -e ADMIN_PASSWORD=... backend python -m app.auth.seed_admin

or put ADMIN_EMAIL / ADMIN_PASSWORD in .env. Nothing about the admin is hardcoded in the code or the frontend, and
the password is never printed or logged.
"""
import os
import sys

from pydantic import TypeAdapter, ValidationError

from .db import get_conn
from .schemas import Email, SignupCompleteIn
from .security import hash_password


def main() -> None:
    email, password = os.environ.get("ADMIN_EMAIL", ""), os.environ.get("ADMIN_PASSWORD", "")
    name = os.environ.get("ADMIN_NAME", "Research Admin")
    if not email or not password:
        sys.exit("set ADMIN_EMAIL and ADMIN_PASSWORD")
    try:
        email = TypeAdapter(Email).validate_python(email)
        SignupCompleteIn._password_rules(password)  # same rules as every other account
    except (ValidationError, ValueError) as exc:
        sys.exit(f"invalid admin credentials: {exc}")
    if len(password) < 12:
        sys.exit("the admin password must be at least 12 characters")
    with get_conn() as conn:
        row = conn.execute(
            "INSERT INTO users (email, password_hash, role, full_name) VALUES (%s, %s, 'admin', %s) "
            "ON CONFLICT (lower(email)) DO UPDATE SET password_hash = EXCLUDED.password_hash, role = 'admin', "
            "full_name = EXCLUDED.full_name, is_active = true, updated_at = now() "
            "RETURNING (xmax = 0) AS created",
            (email, hash_password(password), name)).fetchone()
    print(f"admin account {'created' if row['created'] else 'updated'}")


if __name__ == "__main__":
    main()
