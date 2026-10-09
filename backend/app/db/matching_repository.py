"""What the Matching Agent reads: structured profiles and postings (as views), and the stored embeddings.

Only students with a structured profile (Phase 4) and jobs with a posting (Phase 5) can be matched. The
student's gender is returned separately from the view, for the visibility filter only (docs/adr/0009).
The full job and student lists are cached and reloaded when the stored data changes.
"""

import threading
from typing import Protocol

import psycopg
from psycopg.rows import dict_row

from app.matching.entities import JobView, StudentView, job_view, student_view
from app.matching.vectors import InMemoryVectorIndex, PostgresVectorIndex, VectorIndex


class MatchingRepository(Protocol):
    index: VectorIndex

    def student(self, student_id: str) -> tuple[StudentView, str | None] | None: ...  # (view, visibility gender)

    def students(self) -> list[tuple[StudentView, str | None]]: ...

    def job(self, job_id: str) -> JobView | None: ...

    def jobs(self) -> list[JobView]: ...


_STUDENT_SQL = "SELECT student_id, profile, degree, country, gender FROM student_profiles WHERE profile IS NOT NULL"
_JOB_SQL = "SELECT job_id, posting, country, near_duplicate_of, source_dataset FROM jobs WHERE posting IS NOT NULL"


def _student(row) -> tuple[StudentView, str | None]:
    return student_view(row["student_id"], row["profile"], row), row["gender"]


def _job(row) -> JobView:
    return job_view(row["job_id"], row["posting"], row)


class PostgresMatchingRepository:
    def __init__(self, dsn: str, connect_timeout: int = 5):
        self._dsn, self._timeout = dsn, connect_timeout
        self.index = PostgresVectorIndex(dsn, connect_timeout)
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[tuple, list]] = {}

    def _connect(self):
        return psycopg.connect(self._dsn, connect_timeout=self._timeout, row_factory=dict_row)

    def _cached(self, key: str, stamp_sql: str, load) -> list:
        with self._connect() as conn:
            stamp = tuple(conn.execute(stamp_sql).fetchone().values())
        with self._lock:
            if key not in self._cache or self._cache[key][0] != stamp:
                self._cache[key] = (stamp, load())
            return self._cache[key][1]

    def student(self, student_id):
        with self._connect() as conn:
            row = conn.execute(_STUDENT_SQL + " AND student_id = %s", (student_id,)).fetchone()
        return _student(row) if row else None

    def students(self):
        def load():
            with self._connect() as conn:
                return [_student(r) for r in conn.execute(_STUDENT_SQL + " ORDER BY student_id")]
        return self._cached("students", "SELECT count(*), max(profile_updated_at) FROM student_profiles "
                            "WHERE profile IS NOT NULL", load)

    def job(self, job_id):
        with self._connect() as conn:
            row = conn.execute(_JOB_SQL + " AND job_id = %s", (job_id,)).fetchone()
        return _job(row) if row else None

    def jobs(self):
        def load():
            with self._connect() as conn:
                return [_job(r) for r in conn.execute(_JOB_SQL + " ORDER BY job_id")]
        return self._cached("jobs", "SELECT count(*), max(posting_updated_at), count(near_duplicate_of) FROM jobs "
                            "WHERE posting IS NOT NULL", load)


class InMemoryMatchingRepository:
    """For tests: views and vectors held in memory."""

    def __init__(self):
        self.index = InMemoryVectorIndex()
        self._students: dict[str, tuple[StudentView, str | None]] = {}
        self._jobs: dict[str, JobView] = {}

    def add_student(self, view: StudentView, visibility_gender: str | None = None) -> None:
        self._students[view.student_id] = (view, visibility_gender)

    def add_job(self, view: JobView) -> None:
        self._jobs[view.job_id] = view

    def student(self, student_id):
        return self._students.get(student_id)

    def students(self):
        return [self._students[k] for k in sorted(self._students)]

    def job(self, job_id):
        return self._jobs.get(job_id)

    def jobs(self):
        return [self._jobs[k] for k in sorted(self._jobs)]


def build_matching_repository(settings) -> MatchingRepository | None:
    if not settings.database_url:
        return None
    return PostgresMatchingRepository(settings.database_url.get_secret_value())
