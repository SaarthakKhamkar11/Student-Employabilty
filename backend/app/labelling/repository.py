"""Storage for the labelling page: the pool of pairs, the judgments, and the student/job rows shown."""

from typing import Any, Protocol

import psycopg
from psycopg.rows import dict_row

STUDENT_FIELDS = ("student_id", "degree", "branch", "domain", "skills_raw", "internship_raw",
                  "technical_projects", "certifications", "achievements")
JOB_FIELDS = ("job_id", "job_title", "role", "skills", "job_description", "responsibilities")


class LabellingRepository(Protocol):
    def active_pool(self) -> str | None: ...

    def pool(self, pool_version: str) -> list[dict[str, Any]]: ...

    def judgments(self, pool_version: str) -> list[dict[str, Any]]: ...

    def save_judgment(self, pool_version: str, student_id: str, job_id: str, labeller: str, label: int,
                      rubric_version: str) -> None: ...

    def student(self, student_id: str) -> dict[str, Any] | None: ...

    def jobs(self, job_ids: list[str]) -> dict[str, dict[str, Any]]: ...

    def load_pool(self, rows: list[dict[str, Any]]) -> int: ...


class PostgresLabellingRepository:
    def __init__(self, dsn: str, connect_timeout: int = 5):
        self._dsn, self._timeout = dsn, connect_timeout

    def _connect(self, **kwargs):
        return psycopg.connect(self._dsn, connect_timeout=self._timeout, **kwargs)

    def active_pool(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT pool_version FROM relevance_pool GROUP BY pool_version "
                               "ORDER BY max(created_at) DESC, pool_version LIMIT 1").fetchone()
        return row[0] if row else None

    def pool(self, pool_version: str) -> list[dict[str, Any]]:
        with self._connect(row_factory=dict_row) as conn:
            return conn.execute("SELECT student_id, job_id, split, display_order FROM relevance_pool "
                                "WHERE pool_version = %s ORDER BY student_id, display_order", (pool_version,)).fetchall()

    def judgments(self, pool_version: str) -> list[dict[str, Any]]:
        with self._connect(row_factory=dict_row) as conn:
            return conn.execute("SELECT student_id, job_id, labeller, relevance_label, label_source, rubric_version "
                                "FROM relevance_judgments WHERE pool_version = %s", (pool_version,)).fetchall()

    def save_judgment(self, pool_version, student_id, job_id, labeller, label, rubric_version) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO relevance_judgments (pool_version, student_id, job_id, labeller, relevance_label, "
                "rubric_version) VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (pool_version, student_id, job_id, labeller) DO UPDATE SET "
                "relevance_label = EXCLUDED.relevance_label, rubric_version = EXCLUDED.rubric_version, updated_at = now()",
                (pool_version, student_id, job_id, labeller, label, rubric_version))

    def student(self, student_id: str) -> dict[str, Any] | None:
        with self._connect(row_factory=dict_row) as conn:
            return conn.execute(f"SELECT {', '.join(STUDENT_FIELDS)} FROM student_profiles WHERE student_id = %s",
                                (student_id,)).fetchone()

    def jobs(self, job_ids: list[str]) -> dict[str, dict[str, Any]]:
        with self._connect(row_factory=dict_row) as conn:
            rows = conn.execute(f"SELECT {', '.join(JOB_FIELDS)} FROM jobs WHERE job_id = ANY(%s)", (job_ids,)).fetchall()
        return {r["job_id"]: r for r in rows}

    def load_pool(self, rows: list[dict[str, Any]]) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO relevance_pool (pool_version, student_id, job_id, split, display_order) "
                "VALUES (%(pool_version)s, %(student_id)s, %(job_id)s, %(split)s, %(display_order)s) "
                "ON CONFLICT (pool_version, student_id, job_id) DO NOTHING", rows)
        return len(rows)


class InMemoryLabellingRepository:
    """For tests and for running without Postgres."""

    def __init__(self, students: dict[str, dict] | None = None, jobs: dict[str, dict] | None = None):
        self._pool: list[dict] = []
        self._judgments: dict[tuple, dict] = {}
        self._students, self._jobs = students or {}, jobs or {}

    def active_pool(self) -> str | None:
        return self._pool[-1]["pool_version"] if self._pool else None

    def pool(self, pool_version):
        rows = [dict(r) for r in self._pool if r["pool_version"] == pool_version]
        return sorted(({k: r[k] for k in ("student_id", "job_id", "split", "display_order")} for r in rows),
                      key=lambda r: (r["student_id"], r["display_order"]))

    def judgments(self, pool_version):
        return [dict(v) for k, v in self._judgments.items() if k[0] == pool_version]

    def save_judgment(self, pool_version, student_id, job_id, labeller, label, rubric_version):
        if not any(r["pool_version"] == pool_version and r["student_id"] == student_id and r["job_id"] == job_id
                   for r in self._pool):
            raise KeyError("pair not in pool")
        self._judgments[(pool_version, student_id, job_id, labeller)] = {
            "student_id": student_id, "job_id": job_id, "labeller": labeller, "relevance_label": label,
            "label_source": "human_reviewed", "rubric_version": rubric_version}

    def student(self, student_id):
        return self._students.get(student_id)

    def jobs(self, job_ids):
        return {j: self._jobs[j] for j in job_ids if j in self._jobs}

    def load_pool(self, rows):
        known = {(r["pool_version"], r["student_id"], r["job_id"]) for r in self._pool}
        self._pool += [dict(r) for r in rows if (r["pool_version"], r["student_id"], r["job_id"]) not in known]
        return len(rows)


def build_labelling_repository(settings) -> LabellingRepository | None:
    if not settings.database_url:
        return None
    return PostgresLabellingRepository(settings.database_url.get_secret_value())
