"""Reads student records and writes Profile Agent results (student_* tables and embeddings).

save_profile() runs in one transaction. Re-parsing a student replaces only the
rows written by the same extractor_version, so earlier provenance is kept.
Dataset columns of seeded students (branch, CGPA ...) are never overwritten.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.schemas.student import StudentProfile
from config.schema import STUDENT_SCHEMA


@dataclass
class EmbeddingRow:
    section: str
    model_name: str
    model_version: str
    dimension: int
    values: list[float]
    indices: list[int] | None  # set: sparse (TF-IDF, ADR 0011); None: dense
    interim: bool

    def sql_literal(self) -> tuple[str | None, str | None]:
        """(dense literal, sparse literal). pgvector sparse text format is 1-based: '{1:0.5,7:0.2}/4096'."""
        if self.indices is None:
            return "[" + ",".join(f"{x:.6g}" for x in self.values) + "]", None
        pairs = ",".join(f"{i + 1}:{v:.6g}" for i, v in zip(self.indices, self.values))
        return None, "{" + pairs + "}/" + str(self.dimension)


def upsert_embeddings(cur, entity_type: str, entity_id: str, embeddings: list["EmbeddingRow"]) -> None:
    """Shared by students and jobs: register the model in model_versions, then upsert each vector."""
    for name, version, interim in {(e.model_name, e.model_version, e.interim) for e in embeddings}:
        cur.execute(
            "INSERT INTO model_versions (model_kind, model_name, model_version, status, metadata) "
            "VALUES ('embedding', %s, %s, 'candidate', %s) ON CONFLICT DO NOTHING",
            (name, version, Jsonb({"role": "interim baseline, not selected by experiment (docs/adr/0004)"}
                                  if interim else {})))
    cur.executemany(
        "INSERT INTO embeddings (entity_id, entity_type, section, model_name, model_version, dimension, "
        "embedding, embedding_sparse) VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s::sparsevec) "
        "ON CONFLICT (entity_type, entity_id, section, model_name, model_version) DO UPDATE SET "
        "embedding = EXCLUDED.embedding, embedding_sparse = EXCLUDED.embedding_sparse, "
        "dimension = EXCLUDED.dimension, created_at = now()",
        [(entity_id, entity_type, e.section, e.model_name, e.model_version, e.dimension, *e.sql_literal())
         for e in embeddings])


@dataclass
class SaveResult:
    rows: dict[str, int]
    embeddings_stored: int


class ProfileRepository(Protocol):
    def get_student(self, student_id: str) -> dict[str, Any] | None: ...

    def get_stored_profile(self, student_id: str) -> dict[str, Any] | None: ...

    def save_profile(self, profile: StudentProfile, embeddings: list[EmbeddingRow]) -> SaveResult: ...


def _child_rows(profile: StudentProfile) -> dict[str, list[tuple]]:
    skills = profile.skills if isinstance(profile.skills, list) else []
    as_list = lambda v: v if isinstance(v, list) else []  # noqa: E731
    return {
        "student_skills": [(s.raw, s.canonical, s.category) for s in skills],
        "student_projects": [(p,) for p in as_list(profile.projects)],
        "student_certifications": [(c,) for c in as_list(profile.certifications)],
        "student_experience": [(e,) for e in as_list(profile.experience)],
    }


class PostgresProfileRepository:
    def __init__(self, dsn: str, connect_timeout: int = 5):
        self._dsn = dsn
        self._timeout = connect_timeout

    def get_student(self, student_id: str) -> dict[str, Any] | None:
        with psycopg.connect(self._dsn, connect_timeout=self._timeout, row_factory=dict_row) as conn:
            row = conn.execute(f"SELECT {', '.join(STUDENT_SCHEMA)} FROM student_profiles WHERE student_id = %s",
                               (student_id,)).fetchone()
        return dict(row) if row else None

    def get_stored_profile(self, student_id: str) -> dict[str, Any] | None:
        with psycopg.connect(self._dsn, connect_timeout=self._timeout) as conn:
            row = conn.execute("SELECT profile FROM student_profiles WHERE student_id = %s", (student_id,)).fetchone()
        return row[0] if row else None

    def save_profile(self, profile: StudentProfile, embeddings: list[EmbeddingRow]) -> SaveResult:
        version, sid = profile.extractor_version, profile.student_id
        pii = profile.pii.model_dump() if not isinstance(profile.pii, str) else None
        children = _child_rows(profile)
        with psycopg.connect(self._dsn, connect_timeout=self._timeout) as conn, conn.transaction():
            conn.execute(
                "INSERT INTO student_profiles (student_id, source_dataset, is_synthetic, profile, "
                "profile_extractor_version, profile_updated_at, pii_flags) "
                "VALUES (%s, 'resume_upload', false, %s, %s, now(), %s) "
                "ON CONFLICT (student_id) DO UPDATE SET profile = EXCLUDED.profile, "
                "profile_extractor_version = EXCLUDED.profile_extractor_version, profile_updated_at = now(), "
                "pii_flags = EXCLUDED.pii_flags, updated_at = now()",
                (sid, Jsonb(profile.model_dump(mode="json")), version, Jsonb(pii)))
            for table, rows in children.items():
                conn.execute(f"DELETE FROM {table} WHERE student_id = %s AND extractor_version = %s", (sid, version))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO student_skills (student_id, skill_raw, skill_normalized, skill_type, source, "
                    "extractor_version) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    [(sid, raw, canonical, category, profile.source, version)
                     for raw, canonical, category in children["student_skills"]])
                for table, column in (("student_projects", "description"), ("student_certifications", "name"),
                                      ("student_experience", "description")):
                    cur.executemany(f"INSERT INTO {table} (student_id, {column}, source, extractor_version) "
                                    "VALUES (%s, %s, %s, %s)",
                                    [(sid, value, profile.source, version) for (value,) in children[table]])
                upsert_embeddings(cur, "student", sid, embeddings)
        return SaveResult(rows={t: len(r) for t, r in children.items()}, embeddings_stored=len(embeddings))


@dataclass
class InMemoryProfileRepository:
    """For tests and demos. Mirrors the Postgres semantics without a database."""

    students: dict[str, dict[str, Any]] = field(default_factory=dict)
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    embeddings: dict[tuple, EmbeddingRow] = field(default_factory=dict)

    def get_student(self, student_id: str) -> dict[str, Any] | None:
        return self.students.get(student_id)

    def get_stored_profile(self, student_id: str) -> dict[str, Any] | None:
        return self.profiles.get(student_id)

    def save_profile(self, profile: StudentProfile, embeddings: list[EmbeddingRow]) -> SaveResult:
        # as in Postgres: a new id (a resume upload) gets a row with only its source; stored columns stay empty
        self.students.setdefault(profile.student_id, {"student_id": profile.student_id,
                                                      "source_dataset": "resume_upload", "is_synthetic": False})
        self.profiles[profile.student_id] = profile.model_dump(mode="json")
        for e in embeddings:
            self.embeddings[(profile.student_id, e.section, e.model_name, e.model_version)] = e
        return SaveResult(rows={t: len(r) for t, r in _child_rows(profile).items()},
                          embeddings_stored=len(embeddings))


def build_profile_repository(settings) -> ProfileRepository | None:
    """None when no database is configured: saving then fails visibly (saved=false), never silently."""
    if not settings.database_url:
        return None
    return PostgresProfileRepository(settings.database_url.get_secret_value())
