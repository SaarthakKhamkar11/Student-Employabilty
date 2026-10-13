"""Phase 6 labelling page: assignment order, what labellers see, validation, export and storage.

The Postgres tests use their own pool version, dated in the past so it never becomes the active pool,
and delete it afterwards: the team's real labels are never touched.
"""

import csv
import io
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.labelling import routes, service
from app.labelling.repository import InMemoryLabellingRepository, PostgresLabellingRepository
from app.labelling.rubric import RUBRIC_VERSION
from app.main import app

PG = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to run against Postgres")
POOL = "pool-test"
STUDENTS = [f"S{i:05d}" for i in range(1, 7)]
JOBS = ["J0000001", "J0000002", "J0000003"]


def make_repo() -> InMemoryLabellingRepository:
    students = {s: {"student_id": s, "degree": "B.E.", "branch": "Computer Engineering", "domain": "Data Engineer",
                    "skills_raw": "Python, SQL", "internship_raw": "Intern at Example Corp, contact me at a.b@example.com",
                    "technical_projects": "", "certifications": None, "achievements": "Hackathon finalist",
                    "gender": "Female", "avg_cgpa_sem5": "9.1"} for s in STUDENTS}
    jobs = {j: {"job_id": j, "job_title": "Data Engineer", "role": f"Role {j}", "skills": "Spark", "job_description": "d",
                "responsibilities": "r", "company_name": "Hidden Co", "salary_range": "$1", "gender_preference": "Male"}
            for j in JOBS}
    repo = InMemoryLabellingRepository(students, jobs)
    repo.load_pool([{"pool_version": POOL, "student_id": s, "job_id": j, "split": "validation", "display_order": i}
                    for s in STUDENTS for i, j in enumerate(JOBS, 1)])
    return repo


def grade_all(repo, student, labeller, grades=(3, 1, 0)):
    for job, g in zip(JOBS, grades):
        service.save(repo, student, job, labeller, g)


def test_overlap_students_come_first_and_a_started_student_is_resumed():
    repo = make_repo()
    overlap = service.overlap_students(POOL, STUDENTS)
    assert len(overlap) == service.OVERLAP_STUDENTS
    first = service.status(repo, "ab")["you"]["next_student"]
    assert first in overlap
    service.save(repo, first, JOBS[0], "AB", 2)
    assert service.status(repo, "AB")["you"]["next_student"] == first  # finish what you started
    grade_all(repo, first, "AB")
    nxt = service.status(repo, "CD")["you"]
    assert nxt["next_student"] in overlap  # a second person is still needed on overlap students
    assert service.status(repo, "AB")["you"]["students_completed"] == 1


def test_the_task_hides_protected_and_noisy_fields_and_masks_pii():
    repo = make_repo()
    t = service.task(repo, STUDENTS[0], "ab")
    text = str(t)
    for hidden in ("Female", "9.1", "Hidden Co", "$1", "Male", "a.b@example.com"):
        assert hidden not in text
    assert "[EMAIL]" in text and t["pii_detector"] == "pii-regex-v2"
    assert [j["job_id"] for j in t["jobs"]] == JOBS and all(j["your_label"] is None for j in t["jobs"])
    labels = [f["label"] for f in t["student"]]
    assert "Projects" not in labels and "Certifications" not in labels  # empty fields are not shown


@pytest.mark.parametrize("student, job, labeller, label, message", [
    (STUDENTS[0], JOBS[0], "a", 2, "initials"),
    (STUDENTS[0], JOBS[0], "a1", 2, "initials"),
    (STUDENTS[0], JOBS[0], "ab", 4, "must be 0, 1, 2 or 3"),
    (STUDENTS[0], "J9999999", "ab", 2, "not in the labelling pool"),
])
def test_invalid_judgments_are_refused(student, job, labeller, label, message):
    with pytest.raises(service.LabellingError, match=message):
        service.save(make_repo(), student, job, labeller, label)


def test_export_keeps_complete_students_takes_the_median_rounding_down_and_reports_agreement():
    repo = make_repo()
    grade_all(repo, STUDENTS[0], "AB", (3, 2, 0))
    grade_all(repo, STUDENTS[0], "CD", (2, 2, 1))  # medians 2.5 -> 2, 2, 0.5 -> 0
    grade_all(repo, STUDENTS[1], "AB", (1, 0, 0))
    service.save(repo, STUDENTS[2], JOBS[0], "AB", 3)  # incomplete: not exported
    rows, manifest = service.export(repo)
    by_pair = {(r["student_id"], r["job_id"]): r["relevance_label"] for r in rows}
    assert {r["student_id"] for r in rows} == {STUDENTS[0], STUDENTS[1]}
    assert [by_pair[(STUDENTS[0], j)] for j in JOBS] == [2, 2, 0]
    assert all(r["label_source"] == "human_reviewed" and r["split"] == "validation" for r in rows)
    assert manifest["generated_by"] == "human" and manifest["models_used_for_labels"] == []
    assert manifest["students_incomplete"] == 4 and manifest["agreement"]["pairs_with_two_or_more_labels"] == 3
    assert manifest["agreement"]["exact_agreement"] == pytest.approx(1 / 3, abs=1e-3)
    assert manifest["rubric_version"] == RUBRIC_VERSION


def test_the_api_serves_the_page_saves_grades_and_exports_csv():
    repo = make_repo()
    app.dependency_overrides[routes.get_repo] = lambda: repo
    client = TestClient(app)
    try:
        page = client.get("/labelling")
        assert page.status_code == 200 and "Relevance labelling" in page.text and "<script>" in page.text
        status = client.get("/labelling/api/status", params={"labeller": "ab"}).json()
        assert status["pool_version"] == POOL and status["you"]["labeller"] == "AB"
        sid = status["you"]["next_student"]
        assert client.get(f"/labelling/api/students/{sid}", params={"labeller": "ab"}).status_code == 200
        assert client.get("/labelling/api/students/S99999", params={"labeller": "ab"}).status_code == 404
        for job, g in zip(JOBS, (3, 2, 0)):
            r = client.put("/labelling/api/judgments", json={"student_id": sid, "job_id": job, "labeller": "ab",
                                                             "relevance_label": g})
            assert r.status_code == 200 and r.json()["saved"]
        bad = client.put("/labelling/api/judgments", json={"student_id": sid, "job_id": JOBS[0], "labeller": "ab",
                                                           "relevance_label": 9})
        assert bad.status_code == 400
        assert client.put("/labelling/api/judgments", json={"student_id": "x", "job_id": JOBS[0], "labeller": "ab",
                                                            "relevance_label": 1}).status_code == 422
        export = client.get("/labelling/api/export.csv")
        rows = list(csv.DictReader(io.StringIO(export.text)))
        assert export.headers["content-type"].startswith("text/csv") and len(rows) == 3
        assert list(rows[0]) == ["student_id", "job_id", "relevance_label", "label_source", "split"]
        assert client.get("/labelling/api/export.manifest.json").json()["students_exported"] == 1
    finally:
        app.dependency_overrides.clear()


def test_without_a_database_the_labelling_api_says_so():
    routes_repo = routes._repo
    routes.configure(None)
    try:
        assert TestClient(app).get("/labelling/api/status").status_code == 503
    finally:
        routes.configure(routes_repo)


@pytest.fixture
def pg_pool():
    import psycopg

    from app.db.migrate import apply_migrations

    dsn = os.environ["TEST_DATABASE_URL"]
    apply_migrations(dsn)
    version = f"test-{uuid.uuid4().hex[:8]}"
    with psycopg.connect(dsn) as conn:
        # a pair can be in only one pool version (0009), so use students and jobs no pool has yet
        students = [r[0] for r in conn.execute("SELECT student_id FROM student_profiles s WHERE NOT EXISTS "
                                               "(SELECT 1 FROM relevance_pool p WHERE p.student_id = s.student_id) "
                                               "ORDER BY student_id LIMIT 2")]
        jobs = [r[0] for r in conn.execute("SELECT job_id FROM jobs j WHERE NOT EXISTS "
                                           "(SELECT 1 FROM relevance_pool p WHERE p.job_id = j.job_id) "
                                           "ORDER BY job_id LIMIT 2")]
        for s in students:
            for i, j in enumerate(jobs, 1):
                conn.execute("INSERT INTO relevance_pool (pool_version, student_id, job_id, split, display_order, created_at) "
                             "VALUES (%s, %s, %s, 'validation', %s, '2000-01-01')", (version, s, j, i))
    yield dsn, version, students, jobs
    with psycopg.connect(dsn) as conn:
        conn.execute("DELETE FROM relevance_pool WHERE pool_version = %s", (version,))


@PG
def test_postgres_repository_round_trip_and_constraints(pg_pool):
    import psycopg

    dsn, version, students, jobs = pg_pool
    repo = PostgresLabellingRepository(dsn)
    assert repo.active_pool() is not None
    assert len(repo.pool(version)) == 4
    repo.save_judgment(version, students[0], jobs[0], "AB", 2, RUBRIC_VERSION)
    repo.save_judgment(version, students[0], jobs[0], "AB", 3, RUBRIC_VERSION)  # an edit, not a second row
    got = repo.judgments(version)
    assert len(got) == 1 and got[0]["relevance_label"] == 3 and got[0]["label_source"] == "human_reviewed"
    assert repo.student(students[0])["student_id"] == students[0] and set(repo.jobs(jobs)) == set(jobs)
    with pytest.raises(psycopg.errors.CheckViolation):
        repo.save_judgment(version, students[0], jobs[1], "ab", 1, RUBRIC_VERSION)  # initials must be upper-case
    with pytest.raises(psycopg.errors.CheckViolation):
        repo.save_judgment(version, students[0], jobs[1], "AB", 5, RUBRIC_VERSION)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        repo.save_judgment(version, students[0], "J0000000", "AB", 1, RUBRIC_VERSION)
