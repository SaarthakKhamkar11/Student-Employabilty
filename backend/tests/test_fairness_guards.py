"""gender_preference never enters a scoring feature vector (spec §11, docs/adr/0009)."""

import re
from pathlib import Path

import pytest

from app.agents.matching_agent import MatchingAgent
from app.jobs.visibility import is_visible_to
from app.matching.features import FEATURE_NAMES, ProtectedFeatureError, validate_feature_names
from app.schemas.job import JobPosting, JobVisibility
from tests.helpers import STUB_OUTPUTS

APP = Path(__file__).resolve().parents[1] / "app"
# The only modules allowed to name the field: its schema, the visibility filter, the tool that maps
# it through, the agent that builds postings, and the guard that forbids it.
ALLOWED = {"schemas/job.py", "jobs/visibility.py", "tools/job_tools.py", "agents/job_intelligence_agent.py",
           "matching/features.py"}


def test_gender_preference_is_absent_from_every_scoring_feature_list():
    for name, features in (("FEATURE_NAMES", FEATURE_NAMES), ("MatchingAgent.planned_features",
                                                              MatchingAgent.planned_features),
                           ("JobPosting.SCORING_FIELDS", JobPosting.SCORING_FIELDS)):
        assert not [f for f in features if "gender" in f.lower()], name


def test_the_feature_guard_rejects_gender_preference():
    with pytest.raises(ProtectedFeatureError):
        validate_feature_names(["skill_match_score", "gender_preference"])
    with pytest.raises(ProtectedFeatureError):
        validate_feature_names(["gender"])


def test_a_postings_scoring_view_never_contains_the_preference():
    posting = STUB_OUTPUTS["job_intelligence_agent"].posting.model_copy(
        update={"visibility": JobVisibility(gender_preference="female")})
    scoring_view = posting.model_dump(include=set(JobPosting.SCORING_FIELDS))
    assert "visibility" not in scoring_view and "female" not in str(scoring_view)


def test_no_scoring_code_reads_gender_preference():
    offenders = [str(p.relative_to(APP)) for p in APP.rglob("*.py")
                 if re.search(r"gender_preference", p.read_text()) and str(p.relative_to(APP)) not in ALLOWED]
    assert offenders == [], f"gender_preference referenced outside the visibility path: {offenders}"


@pytest.mark.parametrize("preference, student, visible", [
    ("both", "female", True), ("male", "male", True), ("male", "female", False), ("female", "male", False),
    ("female", None, True), ("male", "", True), (None, "female", True), ("unknown", "male", True),
])
def test_visibility_filter_rules(preference, student, visible):
    assert is_visible_to(preference, student) is visible


def test_visibility_filter_can_be_switched_off():
    assert is_visible_to("male", "female", enabled=False) is True
