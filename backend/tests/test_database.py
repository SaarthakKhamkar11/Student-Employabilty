"""Phase 4 groundwork: student/embedding tables, model registry, and the pandas seed loader.

Postgres tests run only when TEST_DATABASE_URL is set (see PROGRESS.md).
"""

import os
from pathlib import Path

import pytest

from app.db.seed import DEFAULT_STUDENTS_CSV, load_student_records
from config.schema import STUDENT_SCHEMA

PG = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to run against Postgres")


@pytest.fixture
def students_csv() -> Path:
    if not DEFAULT_STUDENTS_CSV.exists():
        pytest.skip("processed data not present")
    return DEFAULT_STUDENTS_CSV


def test_seed_reads_and_validates_every_student_with_pandas(students_csv):
    records = load_student_records(students_csv)
    assert len(records) == 286
    kaggle = [r for r in records if r.source_dataset == "kaggle_resume_dataset"]
    assert len(kaggle) == 166 and all(r.avg_cgpa_sem5 is None and r.gender is None for r in kaggle)


def test_seed_refuses_a_csv_missing_schema_columns(tmp_path, students_csv):
    import pandas as pd

    broken = tmp_path / "broken.csv"
    pd.read_csv(students_csv, dtype=str).drop(columns=["branch"]).to_csv(broken, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_student_records(broken)


@pytest.fixture
def dsn():
    from app.db.migrate import apply_migrations

    url = os.environ["TEST_DATABASE_URL"]
    apply_migrations(url)
    return url


@PG
def test_all_migrations_create_the_expected_tables(dsn):
    import psycopg

    with psycopg.connect(dsn) as conn:
        tables = {r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
        applied = [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
    assert {"agent_runs", "student_profiles", "student_skills", "student_projects", "student_certifications",
            "student_experience", "embeddings", "model_versions"} <= tables
    assert applied == sorted(p.name for p in (Path(__file__).parents[1] / "migrations").glob("*.sql"))


@PG
def test_student_profile_columns_mirror_the_schema(dsn):
    import psycopg

    with psycopg.connect(dsn) as conn:
        cols = [r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'student_profiles' "
            "ORDER BY ordinal_position")]
    assert cols[:len(STUDENT_SCHEMA)] == STUDENT_SCHEMA


@PG
def test_seed_is_idempotent_and_keeps_blanks_null(dsn, students_csv):
    import psycopg

    from app.db.seed import seed_students

    records = load_student_records(students_csv)
    seed_students(dsn, records)
    seed_students(dsn, records)  # second run updates, never duplicates
    with psycopg.connect(dsn) as conn:
        total = conn.execute("SELECT count(*) FROM student_profiles").fetchone()[0]
        nulls = conn.execute("SELECT count(*) FROM student_profiles WHERE source_dataset = 'kaggle_resume_dataset' "
                             "AND avg_cgpa_sem5 IS NULL AND gender IS NULL AND graduation_year IS NULL").fetchone()[0]
        first = conn.execute("SELECT avg_cgpa_sem5, graduation_year FROM student_profiles WHERE student_id = %s",
                             (records[0].student_id,)).fetchone()
    assert total >= 286 and nulls == 166
    assert first == (records[0].avg_cgpa_sem5, records[0].graduation_year)


@PG
def test_embedding_dimension_must_match_the_vector(dsn):
    import psycopg

    with psycopg.connect(dsn) as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute("INSERT INTO embeddings (entity_id, entity_type, model_name, model_version, dimension, "
                         "embedding) VALUES ('S00001', 'student', 'test-model', 'v0', 4, '[1,2,3]')")
        conn.rollback()


@PG
def test_a_model_cannot_be_selected_without_experiment_evidence(dsn):
    import psycopg

    with psycopg.connect(dsn) as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute("INSERT INTO model_versions (model_kind, model_name, model_version, status) "
                         "VALUES ('embedding', 'test-model', 'v0', 'selected')")
        conn.rollback()
