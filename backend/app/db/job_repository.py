"""Reads job rows and writes Job Intelligence Agent results (jobs, job_skills, job_responsibilities, embeddings).

Dataset columns of seeded jobs are never overwritten; only the derived columns
are. An uploaded job description gets a new row whose missing dataset columns
stay NULL.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.db.profile_repository import EmbeddingRow, SaveResult, upsert_embeddings
from app.schemas.job import JobPosting
from config.schema import JOB_SCHEMA

_RAW_FROM_POSTING = ("job_title", "role", "company_name", "location", "country", "work_type", "salary_range",
                     "job_portal", "job_description", "qualification_required", "experience_required")
_SENTINELS = ("Not detected", "Not yet evaluated", "Skill not detected")


def _value(v: Any) -> Any:
    return None if isinstance(v, str) and v in _SENTINELS else v


def _children(posting: JobPosting) -> dict[str, list[tuple]]:
    skills = posting.skills if isinstance(posting.skills, list) else []
    duties = posting.responsibilities if isinstance(posting.responsibilities, list) else []
    return {"job_skills": [(s.raw, s.canonical, s.category, s.requirement, s.requirement_basis) for s in skills],
            "job_responsibilities": [(i, d) for i, d in enumerate(duties, 1)]}


class JobRepository(Protocol):
    def get_job(self, job_id: str) -> dict[str, Any] | None: ...

    def get_stored_posting(self, job_id: str) -> dict[str, Any] | None: ...

    def candidates(self, company_name: str | None, job_title: str) -> list[dict[str, Any]]: ...

    def save_posting(self, posting: JobPosting, embeddings: list[EmbeddingRow], is_new: bool) -> SaveResult: ...


class PostgresJobRepository:
    def __init__(self, dsn: str, connect_timeout: int = 5):
        self._dsn = dsn
        self._timeout = connect_timeout

    def _connect(self, **kwargs):
        return psycopg.connect(self._dsn, connect_timeout=self._timeout, **kwargs)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect(row_factory=dict_row) as conn:
            row = conn.execute(f"SELECT {', '.join(JOB_SCHEMA)}, source_dataset FROM jobs WHERE job_id = %s",
                               (job_id,)).fetchone()
        return dict(row) if row else None

    def get_stored_posting(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT posting FROM jobs WHERE job_id = %s", (job_id,)).fetchone()
        return row[0] if row else None

    def candidates(self, company_name: str | None, job_title: str) -> list[dict[str, Any]]:
        """Stored jobs with the same company and title; with no company, those that have none either."""
        if company_name:
            where, params = "lower(company_name) = lower(%s) AND ", [company_name.strip()]
        else:  # no parameter here: an untyped NULL parameter makes Postgres reject the query
            where, params = "company_name IS NULL AND ", []
        with self._connect(row_factory=dict_row) as conn:
            rows = conn.execute(
                "SELECT job_id, company_name, job_title, job_description, responsibilities, skills FROM jobs "
                f"WHERE {where}lower(job_title) = lower(%s)", (*params, job_title.strip())).fetchall()
        return [dict(r) for r in rows]

    def save_posting(self, posting: JobPosting, embeddings: list[EmbeddingRow], is_new: bool) -> SaveResult:
        jid, version = posting.job_id, posting.extractor_version
        children = _children(posting)
        duplicate = posting.duplicate_check if not isinstance(posting.duplicate_check, str) else None
        pii = posting.pii.model_dump() if not isinstance(posting.pii, str) else None
        with self._connect() as conn, conn.transaction():
            if is_new:
                cols = ["job_id", "source_dataset", *_RAW_FROM_POSTING, "responsibilities", "skills"]
                raw = posting.model_dump()
                values = [jid, "job_description_upload", *[_value(raw[c]) for c in _RAW_FROM_POSTING],
                          "\n".join(raw["responsibilities"]) if isinstance(raw["responsibilities"], list) else None,
                          ", ".join(s["raw"] for s in raw["skills"]) if isinstance(raw["skills"], list) else None]
                conn.execute(f"INSERT INTO jobs ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
                             "ON CONFLICT (job_id) DO NOTHING", values)
            conn.execute(
                "UPDATE jobs SET posting_updated_at = CASE WHEN posting IS DISTINCT FROM %s THEN now() "
                "ELSE posting_updated_at END, posting = %s, posting_extractor_version = %s, "
                "min_degree = %s, min_degree_level = %s, min_experience_years = %s, max_experience_years = %s, "
                "near_duplicate_of = %s, pii_flags = %s, updated_at = now() WHERE job_id = %s",
                (Jsonb(posting.model_dump(mode="json")), Jsonb(posting.model_dump(mode="json")), version,
                 _value(posting.min_degree),
                 _value(posting.min_degree_level), _value(posting.min_experience_years),
                 _value(posting.max_experience_years), duplicate.near_duplicate_of if duplicate else None,
                 Jsonb(pii), jid))
            for table in children:
                conn.execute(f"DELETE FROM {table} WHERE job_id = %s AND extractor_version = %s", (jid, version))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO job_skills (job_id, skill_raw, skill_normalized, skill_type, requirement, "
                    "requirement_basis, source, extractor_version) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT DO NOTHING",
                    [(jid, *row, posting.source, version) for row in children["job_skills"]])
                cur.executemany(
                    "INSERT INTO job_responsibilities (job_id, position, description, source, extractor_version) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    [(jid, pos, desc, posting.source, version) for pos, desc in children["job_responsibilities"]])
                upsert_embeddings(cur, "job", jid, embeddings)
        return SaveResult(rows={t: len(r) for t, r in children.items()}, embeddings_stored=len(embeddings))


@dataclass
class InMemoryJobRepository:
    """For tests and demos; mirrors the Postgres semantics."""

    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    postings: dict[str, dict[str, Any]] = field(default_factory=dict)
    embeddings: dict[tuple, EmbeddingRow] = field(default_factory=dict)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    def get_stored_posting(self, job_id: str) -> dict[str, Any] | None:
        return self.postings.get(job_id)

    def candidates(self, company_name: str | None, job_title: str) -> list[dict[str, Any]]:
        key = ((company_name or "").strip().lower(), job_title.strip().lower())
        return [j for j in self.jobs.values()
                if ((j.get("company_name") or "").strip().lower(), (j.get("job_title") or "").strip().lower()) == key]

    def save_posting(self, posting: JobPosting, embeddings: list[EmbeddingRow], is_new: bool) -> SaveResult:
        data = posting.model_dump(mode="json")
        if is_new:
            self.jobs[posting.job_id] = {**{c: _value(data.get(c)) for c in ("job_id", *_RAW_FROM_POSTING)},
                                         "source_dataset": "job_description_upload"}  # as the Postgres INSERT
        self.postings[posting.job_id] = data
        for e in embeddings:
            self.embeddings[(posting.job_id, e.section, e.model_name, e.model_version)] = e
        return SaveResult(rows={t: len(r) for t, r in _children(posting).items()}, embeddings_stored=len(embeddings))


def build_job_repository(settings) -> JobRepository | None:
    if not settings.database_url:
        return None
    return PostgresJobRepository(settings.database_url.get_secret_value())
