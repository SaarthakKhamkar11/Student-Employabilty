"""What the Learning Agent reads and writes: catalogue courses, cached YouTube searches, saved recommendations.

Every resource returned is a stored row (learning_resources) with its source_reference: nothing is composed here.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Protocol

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

COLUMNS = ("resource_id", "title", "url", "provider", "description", "skills", "difficulty", "prerequisites",
           "estimated_effort", "source_reference")


def _row(r: dict) -> dict:
    out = {k: r.get(k) for k in COLUMNS}
    if "skills_covered" in r:
        out["skills"] = list(r["skills_covered"] or [])
    out["kind"] = "video" if out["resource_id"].startswith("youtube:") else "course"
    return out


class LearningRepository(Protocol):
    def courses(self) -> list[dict]: ...  # the curated catalogue (never the cached videos)

    def cached_search(self, query: str, max_age_days: float) -> list[dict] | None: ...  # None: not cached / stale

    def store_search(self, query: str, rows: list[dict], request: dict) -> None: ...

    def save_recommendations(self, rows: list[dict]) -> int: ...


class PostgresLearningRepository:
    def __init__(self, dsn: str, connect_timeout: int = 5):
        self._dsn, self._timeout = dsn, connect_timeout

    def _connect(self):
        return psycopg.connect(self._dsn, connect_timeout=self._timeout, row_factory=dict_row)

    def courses(self):
        with self._connect() as conn:
            return [_row(r) for r in conn.execute(
                "SELECT * FROM learning_resources WHERE resource_id NOT LIKE 'youtube:%' ORDER BY resource_id")]

    def cached_search(self, query, max_age_days):
        with self._connect() as conn:
            hit = conn.execute("SELECT video_ids, fetched_at FROM youtube_searches WHERE query = %s", (query,)).fetchone()
            if hit is None or hit["fetched_at"] < datetime.now(timezone.utc) - timedelta(days=max_age_days):
                return None
            rows = {r["resource_id"]: _row(r) for r in conn.execute(
                "SELECT * FROM learning_resources WHERE resource_id = ANY(%s)",
                ([f"youtube:{v}" for v in hit["video_ids"]],))}
        return [rows[f"youtube:{v}"] for v in hit["video_ids"] if f"youtube:{v}" in rows]

    def store_search(self, query, rows, request):
        with self._connect() as conn, conn.transaction():
            for r in rows:
                conn.execute(
                    "INSERT INTO learning_resources (resource_id, title, url, provider, description, skills_covered, "
                    "difficulty, prerequisites, estimated_effort, source_reference) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT (resource_id) DO UPDATE SET title = EXCLUDED.title, provider = EXCLUDED.provider, "
                    "description = EXCLUDED.description, estimated_effort = EXCLUDED.estimated_effort, "
                    "source_reference = EXCLUDED.source_reference, updated_at = now()",
                    (r["resource_id"], r["title"], r["url"], r["provider"], r["description"], r["skills"],
                     r["difficulty"], r["prerequisites"], r["estimated_effort"], r["source_reference"]))
            conn.execute(
                "INSERT INTO youtube_searches (query, video_ids, fetched_at, request) VALUES (%s, %s, now(), %s) "
                "ON CONFLICT (query) DO UPDATE SET video_ids = EXCLUDED.video_ids, fetched_at = now(), "
                "request = EXCLUDED.request",
                (query, [r["resource_id"].split(":", 1)[1] for r in rows], Jsonb(request)))

    def save_recommendations(self, rows):
        with self._connect() as conn, conn.transaction():
            for r in rows:
                conn.execute(
                    "INSERT INTO learning_recommendations (student_id, job_id, skill, resource_id, priority, reason) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (r["student_id"], r["job_id"], r["skill"], r["resource_id"], r["priority"], r["reason"]))
        return len(rows)


class InMemoryLearningRepository:
    """For tests and runs without a database: the catalogue from data/learning/course_catalog.json or given rows."""

    def __init__(self, courses: list[dict] | None = None):
        if courses is None:
            from app.learning.catalog import CATALOG

            courses = json.loads(CATALOG.read_text(encoding="utf-8"))["courses"] if CATALOG.exists() else []
        self._courses = [_row(c) for c in courses]
        self.searches: dict[str, list[dict]] = {}
        self.saved: list[dict] = []

    def courses(self):
        return list(self._courses)

    def cached_search(self, query, max_age_days):
        return self.searches.get(query)

    def store_search(self, query, rows, request):
        self.searches[query] = rows

    def save_recommendations(self, rows):
        self.saved += rows
        return len(rows)


def build_learning_repository(settings) -> LearningRepository:
    if settings.database_url:
        return PostgresLearningRepository(settings.database_url.get_secret_value())
    return InMemoryLearningRepository()
