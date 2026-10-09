import csv
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.agent_io import AgentOutput, MatchingAgentInput, MatchResult, ProfileAgentInput
from app.schemas.agent_state import NOT_YET_EVALUATED, AgentState, ModelVersionInfo
from app.schemas.job import JobRecord
from app.schemas.student import StudentRecord
from config.schema import JOB_SCHEMA, STUDENT_SCHEMA

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def read_rows(name):
    path = PROCESSED / name
    if not path.exists():
        pytest.skip(f"{path} not present")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_record_fields_mirror_config_schema_exactly_and_in_order():
    assert list(StudentRecord.model_fields) == STUDENT_SCHEMA
    assert list(JobRecord.model_fields) == JOB_SCHEMA


def test_every_processed_student_row_loads_without_translation():
    rows = read_rows("students_unified.csv")
    students = [StudentRecord.model_validate(r) for r in rows]
    assert len(students) == 286
    kaggle = [s for s in students if s.source_dataset == "kaggle_resume_dataset"]
    # blank in the source, therefore None: never a guessed value
    for field in ("graduation_year", "gender", "avg_cgpa_sem5", "country"):
        assert all(getattr(s, field) is None for s in kaggle), field


def test_every_processed_job_row_loads_without_translation():
    jobs = [JobRecord.model_validate(r) for r in read_rows("jobs.csv")]
    assert len(jobs) == 4990


def test_students_also_load_through_pandas():
    pd = pytest.importorskip("pandas")
    path = PROCESSED / "students_unified.csv"
    if not path.exists():
        pytest.skip("processed data not present")
    df = pd.read_csv(path)
    assert len([StudentRecord.model_validate(r) for r in df.to_dict("records")]) == len(df)


def test_required_student_field_cannot_be_blank():
    with pytest.raises(ValidationError):
        StudentRecord(student_id="S1", branch="  ", career_interests="x", domain="y",
                      source_dataset="placement_cell", is_synthetic=False)


def test_agent_state_defaults_are_explicit_sentinels_not_guesses():
    s = AgentState(request_id="r1", task_type="recommend_jobs_for_student")
    assert s.status == "pending" and s.agents_called == [] and s.errors == []
    assert set(s.model_versions) == set(ModelVersionInfo.model_fields)
    assert set(s.model_versions.values()) == {NOT_YET_EVALUATED}
    assert AgentState.model_validate_json(s.model_dump_json()) == s


def test_agent_output_cannot_claim_success_without_data():
    base = dict(agent_name="a", termination_reason="finished", started_at="2026-01-01T00:00:00Z",
                finished_at="2026-01-01T00:00:01Z", duration_ms=1.0)
    with pytest.raises(ValidationError):
        AgentOutput(status="success", data=NOT_YET_EVALUATED, **base)
    with pytest.raises(ValidationError):
        AgentOutput(status="failed", data=NOT_YET_EVALUATED, **base)  # failure must carry errors


def test_prediction_requires_model_version_metadata():
    """Spec §24 / Phase 7 rule 16: all seven fields, with real values, or the prediction is not valid."""
    real = dict(embedding_model="m", embedding_version="v", matching_model="mm", matching_model_version="mv",
                feature_version="f", dataset_version="d", experiment_id="e")
    with pytest.raises(ValidationError):
        MatchResult(student_id="S1", job_id="J1", rank=1)
    with pytest.raises(ValidationError):  # sentinels are not metadata
        MatchResult(student_id="S1", job_id="J1", rank=1, model_versions=ModelVersionInfo().model_dump())
    for field in real:
        for bad in (None, "", " ", NOT_YET_EVALUATED):
            with pytest.raises(ValidationError):
                MatchResult(student_id="S1", job_id="J1", rank=1, model_versions={**real, field: bad})
        with pytest.raises(ValidationError):
            MatchResult(student_id="S1", job_id="J1", rank=1,
                        model_versions={k: v for k, v in real.items() if k != field})
    m = MatchResult(student_id="S1", job_id="J1", rank=1, model_versions=real)
    assert m.suitability_score == NOT_YET_EVALUATED


def test_input_contracts_reject_requests_they_cannot_serve():
    with pytest.raises(ValidationError):
        ProfileAgentInput()
    with pytest.raises(ValidationError):
        MatchingAgentInput(mode="jobs_for_student", job_id="J1")
