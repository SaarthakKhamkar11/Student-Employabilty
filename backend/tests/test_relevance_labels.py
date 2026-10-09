"""Phase 6 label loader hard rules: loud validation, honest label_source, circular-evaluation check."""

import json

import pandas as pd
import pytest

from src.evaluation.relevance_labels import STAMPS, LabelValidationError, circularity_check, load_labels
from src.evaluation.splits import split_for
from src.embeddings.embedding_model_benchmark import DEFAULT_JOBS, DEFAULT_STUDENTS

STUDENT, JOB, JOB2 = "S00019", "J0014654", "J0237989"


def _write(tmp_path, rows, manifest=None, name="labels.csv"):
    path = tmp_path / name
    pd.DataFrame(rows, columns=["student_id", "job_id", "relevance_label", "label_source", "split"]).to_csv(path, index=False)
    if manifest is not None:
        (tmp_path / name.replace(".csv", ".manifest.json")).write_text(json.dumps(manifest))
    return str(path)


def _row(student=STUDENT, job=JOB, label="2", source="human_reviewed", split=None):
    return [student, job, label, source, split or split_for(student)]


HUMAN = {"label_source": "human_reviewed", "generated_by": "human", "models_used_for_labels": []}


def test_a_valid_file_loads_with_breakdowns(tmp_path):
    path = _write(tmp_path, [_row(), _row(job=JOB2, label="0")], HUMAN)
    ls = load_labels(path, DEFAULT_STUDENTS, DEFAULT_JOBS)
    assert ls.source_breakdown() == {"human_reviewed": 2}
    assert ls.label_distribution() == {0: 1, 1: 0, 2: 1, 3: 0}
    assert ls.split_breakdown() == {split_for(STUDENT): {"pairs": 2, "students": 1}}


def test_every_invalid_row_is_reported_and_the_load_fails(tmp_path):
    rows = [_row(student="S99999"), _row(job="J0000000"), _row(label="4"), _row(label="2.5"),
            _row(source="model_generated"), _row(split="train" if split_for(STUDENT) != "train" else "test"),
            _row(job=JOB2), _row(job=JOB2)]
    with pytest.raises(LabelValidationError) as err:
        load_labels(_write(tmp_path, rows, HUMAN), DEFAULT_STUDENTS, DEFAULT_JOBS)
    text = str(err.value)
    for expected in ("student_id not in students_unified.csv", "job_id not in jobs.csv", "'4' is not one of",
                     "'2.5' is not one of", "label_source 'model_generated'", "differs from the student's fixed split",
                     "duplicate rows"):
        assert expected in text


def test_missing_columns_and_a_manifest_that_disagrees_with_the_csv_are_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"student_id": [STUDENT], "job_id": [JOB], "relevant": [1]}).to_csv(bad, index=False)
    with pytest.raises(LabelValidationError, match="missing required columns"):
        load_labels(str(bad), DEFAULT_STUDENTS, DEFAULT_JOBS)
    path = _write(tmp_path, [_row(source="synthetic_clearly_marked")], HUMAN, name="lies.csv")
    with pytest.raises(LabelValidationError, match="declares label_source 'human_reviewed'"):
        load_labels(path, DEFAULT_STUDENTS, DEFAULT_JOBS)


@pytest.mark.parametrize("manifest, source, level", [
    (None, "human_reviewed", "LOW"),                                             # provenance unknown
    ({**HUMAN, "models_used_for_labels": ["e5"]}, "human_reviewed", "LOW"),        # an evaluated model made them
    ({"label_source": "weakly_labelled", "generated_by": "model:sbert", "models_used_for_labels": []},
     "weakly_labelled", "LOW"),
    ({"label_source": "human_reviewed", "generated_by": "human"}, "human_reviewed", "LOW"),  # doesn't say
    (HUMAN, "human_reviewed", "OK"),
    ({"label_source": "weakly_labelled", "generated_by": "rules", "models_used_for_labels": []}, "weakly_labelled", "LIMITED"),
    ({"label_source": "synthetic_clearly_marked", "generated_by": "rules", "models_used_for_labels": []},
     "synthetic_clearly_marked", "LOW"),
])
def test_circularity_check_needs_proof_of_independent_labels(tmp_path, manifest, source, level):
    path = _write(tmp_path, [_row(source=source)], manifest)
    result = circularity_check(load_labels(path, DEFAULT_STUDENTS, DEFAULT_JOBS), source, ["tfidf", "sbert", "e5"])
    assert result["level"] == level
    if level == "LOW" and source != "synthetic_clearly_marked":
        assert result["line"] == "evaluation validity: LOW — labels not independently sourced"


def test_labels_that_a_model_reproduces_are_flagged_as_suspected_model_output(tmp_path):
    jobs = pd.read_csv(DEFAULT_JOBS, dtype=str, usecols=["job_id"])["job_id"].head(6).tolist()
    rows = [_row(job=j, label=str(g)) for j, g in zip(jobs, (3, 3, 2, 1, 0, 0))]
    labelset = load_labels(_write(tmp_path, rows, HUMAN), DEFAULT_STUDENTS, DEFAULT_JOBS)
    copycat = {(STUDENT, j): s for j, s in zip(jobs, (0.95, 0.94, 0.6, 0.4, 0.1, 0.05))}
    result = circularity_check(labelset, "human_reviewed", ["e5"], {"e5": copycat})
    assert result["level"] == "LOW" and "suspected model-generated labels" in result["reasons"][0]


def test_synthetic_and_weak_labels_carry_their_stamps():
    assert STAMPS["synthetic_clearly_marked"] == "SYNTHETIC LABELS — NOT REAL GROUND TRUTH"
    assert "NOT HUMAN GROUND TRUTH" in STAMPS["weakly_labelled"]
    assert "human_reviewed" not in STAMPS
