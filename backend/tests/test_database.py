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
            "student_experience", "embeddings", "model_versions", "users", "companies", "jobs", "job_skills",
            "job_responsibilities", "experiment_runs", "experiment_metrics", "match_results", "skill_gaps",
            "career_recommendations", "learning_resources", "learning_recommendations", "feedback_events",
            "agent_messages", "audit_logs"} <= tables
    assert applied == sorted(p.name for p in (Path(__file__).parents[1] / "migrations").glob("*.sql"))


@PG
def test_pii_columns_exist_only_in_users_and_relevance_labels_is_a_view(dsn):
    import psycopg

    with psycopg.connect(dsn) as conn:
        pii = {r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.columns WHERE table_schema = 'public' "
            "AND column_name IN ('email', 'phone', 'password_hash', 'password')")}
        kind = conn.execute("SELECT relkind FROM pg_class WHERE relname = 'relevance_labels'").fetchone()[0]
    assert pii == {"users"}
    assert kind == "v"


@PG
def test_free_form_json_refuses_unredacted_pii_and_audit_logs_are_append_only(dsn):
    import psycopg

    with psycopg.connect(dsn) as conn:
        with pytest.raises(psycopg.errors.CheckViolation), conn.transaction():
            conn.execute("INSERT INTO audit_logs (actor_type, action, entity_type, details) "
                         """VALUES ('system', 'x', 'y', '{"note": "write to a.b@example.com"}')""")
        with pytest.raises(psycopg.errors.CheckViolation), conn.transaction():
            conn.execute("INSERT INTO feedback_events (source, event_type) VALUES ('made_up', 'click')")
        with pytest.raises(psycopg.errors.RaiseException), conn.transaction():
            conn.execute("INSERT INTO audit_logs (actor_type, action, entity_type, details) "
                         """VALUES ('system', 'x', 'y', '{"email": "[REDACTED]"}')""")  # redacted: accepted
            conn.execute("UPDATE audit_logs SET action = 'z'")
        conn.rollback()


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


@PG
def test_profile_repository_writes_profile_children_and_versioned_embeddings(dsn, embedder):
    import asyncio

    import psycopg

    from app.agents import build_agents
    from app.core.config import Settings
    from app.db.profile_repository import PostgresProfileRepository
    from app.db.seed import seed_students
    from app.schemas.agent_state import AgentState

    seed_students(dsn, load_student_records(DEFAULT_STUDENTS_CSV)) if DEFAULT_STUDENTS_CSV.exists() else None
    settings = Settings(_env_file=None, environment="test", tool_retry_backoff_seconds=0)
    agents = build_agents(settings=settings, profile_repo=PostgresProfileRepository(dsn), embedder=embedder)
    for _ in range(2):  # re-running replaces rows of the same extractor version: no duplicates
        out = asyncio.run(agents["profile_agent"].run(AgentState(request_id="pg", task_type="t"),
                                                      {"student_id": "S00001"}))
        assert out.status == "success", out.errors
    with psycopg.connect(dsn) as conn:
        profile, version, flags = conn.execute("SELECT profile, profile_extractor_version, pii_flags "
                                               "FROM student_profiles WHERE student_id = 'S00001'").fetchone()
        skills = conn.execute("SELECT count(*) FROM student_skills WHERE student_id = 'S00001'").fetchone()[0]
        emb = conn.execute("SELECT section, model_name, model_version, dimension, embedding IS NULL, "
                           "embedding_sparse IS NOT NULL FROM embeddings WHERE entity_id = 'S00001' "
                           "ORDER BY section").fetchall()
        registry = conn.execute("SELECT status, metadata->>'role' FROM model_versions "
                                "WHERE model_name = 'tfidf_baseline'").fetchall()
    assert profile["source"] == "stored_record" and version == "profile-v2" and "detected" in flags
    assert skills == len(out.data["profile"]["skills"])  # one row per normalised skill, none duplicated or lost
    assert emb and all(m == "tfidf_baseline" and v.startswith("tfidf-") and d == 4096 and no_dense and sparse
                       for _, m, v, d, no_dense, sparse in emb)  # stored sparse (ADR 0011)
    assert registry and all(status == "candidate" and "interim" in role for status, role in registry)


@PG
def test_an_uploaded_resume_creates_a_student_without_guessed_columns(dsn, embedder):
    import asyncio

    import psycopg

    from app.agents import build_agents
    from app.core.config import Settings
    from app.db.profile_repository import PostgresProfileRepository
    from app.schemas.agent_state import AgentState

    settings = Settings(_env_file=None, environment="test", tool_retry_backoff_seconds=0)
    agents = build_agents(settings=settings, profile_repo=PostgresProfileRepository(dsn), embedder=embedder)
    out = asyncio.run(agents["profile_agent"].run(AgentState(request_id="pg2", task_type="t"),
                                                  {"resume_text": "Skill Details\nPython, Git/GitHub"}))
    sid = out.data["profile"]["student_id"]
    with psycopg.connect(dsn) as conn:
        row = conn.execute("SELECT branch, career_interests, domain, avg_cgpa_sem5, source_dataset "
                           "FROM student_profiles WHERE student_id = %s", (sid,)).fetchone()
        skills = {r[0] for r in conn.execute("SELECT skill_normalized FROM student_skills WHERE student_id = %s",
                                             (sid,))}
        conn.execute("DELETE FROM embeddings WHERE entity_id = %s", (sid,))
        conn.execute("DELETE FROM student_profiles WHERE student_id = %s", (sid,))
    assert row == (None, None, None, None, "resume_upload")
    assert skills == {"Python", "Git", "GitHub"}  # "Git/GitHub" yields two rows (migration 0005)


@PG
def test_jobs_seed_flags_the_measured_near_duplicates(dsn):
    import psycopg

    from app.db.seed_jobs import DEFAULT_JOBS_CSV, load_job_records, seed_jobs
    from app.jobs.dedup import find_near_duplicates

    records = load_job_records(DEFAULT_JOBS_CSV)
    expected = len(find_near_duplicates([r.model_dump() for r in records]))
    assert seed_jobs(dsn, records) == {"rows": 4990, "near_duplicates": expected}
    assert seed_jobs(dsn, records)["rows"] == 4990  # idempotent
    with psycopg.connect(dsn) as conn:
        total, flagged = conn.execute("SELECT count(*), count(near_duplicate_of) FROM jobs "
                                      "WHERE source_dataset = 'kaggle_job_descriptions'").fetchone()
        cols = [r[0] for r in conn.execute("SELECT column_name FROM information_schema.columns "
                                           "WHERE table_name = 'jobs' ORDER BY ordinal_position")]
    assert total == 4990 and flagged == expected
    assert cols[:len(JOB_SCHEMA_COLS)] == JOB_SCHEMA_COLS


from config.schema import JOB_SCHEMA as JOB_SCHEMA_COLS  # noqa: E402


@PG
def test_job_agent_writes_skills_responsibilities_and_sparse_embeddings(dsn, embedder):
    import asyncio

    import psycopg

    from app.agents import build_agents
    from app.core.config import Settings
    from app.db.job_repository import PostgresJobRepository
    from app.db.seed_jobs import DEFAULT_JOBS_CSV, load_job_records, seed_jobs
    from app.schemas.agent_state import AgentState

    records = load_job_records(DEFAULT_JOBS_CSV)
    seed_jobs(dsn, records)
    job_id = records[0].job_id  # ids come from the full source, so the sample has gaps: use a real one
    settings = Settings(_env_file=None, environment="test", tool_retry_backoff_seconds=0)
    agent = build_agents(settings=settings, profile_repo=None, job_repo=PostgresJobRepository(dsn),
                         embedder=embedder)["job_intelligence_agent"]
    for _ in range(2):  # re-processing replaces rows of the same extractor version
        out = asyncio.run(agent.run(AgentState(request_id="pgjob", task_type="t"), {"job_id": job_id}))
        assert out.status == "success", out.errors
    posting = out.data["posting"]
    with psycopg.connect(dsn) as conn:
        skills = conn.execute("SELECT requirement, count(*) FROM job_skills WHERE job_id = %s GROUP BY 1", (job_id,)).fetchall()
        duties = conn.execute("SELECT count(*) FROM job_responsibilities WHERE job_id = %s", (job_id,)).fetchone()[0]
        emb = conn.execute("SELECT count(*), bool_and(embedding IS NULL AND embedding_sparse IS NOT NULL), "
                           "bool_and(model_name = 'tfidf_baseline') FROM embeddings "
                           "WHERE entity_type = 'job' AND entity_id = %s", (job_id,)).fetchone()
        stored = conn.execute("SELECT min_degree_level, min_experience_years, posting->'skill_split'->>'inferred' "
                              "FROM jobs WHERE job_id = %s", (job_id,)).fetchone()
    assert dict(skills) == {"required": len(posting["required_skills"]), "preferred": len(posting["preferred_skills"])}
    assert duties == len(posting["responsibilities"]) and emb[0] > 0 and emb[1] and emb[2]
    assert stored == (posting["min_degree_level"], posting["min_experience_years"], "true")


@PG
def test_an_uploaded_jd_is_stored_once_with_unguessed_columns_null(dsn, embedder):
    import asyncio

    import psycopg

    from app.agents import build_agents
    from app.core.config import Settings
    from app.db.job_repository import PostgresJobRepository
    from app.schemas.agent_state import AgentState

    settings = Settings(_env_file=None, environment="test", tool_retry_backoff_seconds=0)
    agent = build_agents(settings=settings, profile_repo=None, job_repo=PostgresJobRepository(dsn),
                         embedder=embedder)["job_intelligence_agent"]
    jd = "Field Data Analyst\nRequirements: Python, SQL\nExperience: 1-3 years"
    out = asyncio.run(agent.run(AgentState(request_id="pgjd", task_type="t"), {"jd_text": jd}))
    jid = out.data["posting"]["job_id"]
    try:
        with psycopg.connect(dsn) as conn:
            row = conn.execute("SELECT job_title, company_name, location, source_dataset, max_experience_years "
                               "FROM jobs WHERE job_id = %s", (jid,)).fetchone()
        assert row == ("Field Data Analyst", None, None, "job_description_upload", 3.0)
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("DELETE FROM embeddings WHERE entity_type = 'job' AND entity_id = %s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (jid,))


@PG
def test_duplicate_lookup_works_for_jobs_without_a_company(dsn):
    """Regression (Phase 7, job-dedup-v2): a NULL company must not reach Postgres as an untyped parameter."""
    from app.db.job_repository import PostgresJobRepository

    repo = PostgresJobRepository(dsn)
    assert isinstance(repo.candidates(None, "A title no job has"), list)
    assert isinstance(repo.candidates("Example Co", "A title no job has"), list)
