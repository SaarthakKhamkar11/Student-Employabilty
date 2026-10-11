"""Phase 7: the Matching Agent and its stages, on a small in-memory world (tests/matching_world.py)."""

import asyncio
import dataclasses

import pytest

from app.agents import build_agents
from app.core.config import Settings
from app.matching import eligibility as elig
from app.matching import pipeline as P
from app.matching.baseline import predict
from app.matching.calibration import Calibration
from app.matching.config import MatchingConfigError, load_matching_config
from app.matching.entities import StudentView, stated_experience_years
from app.matching.feature_builder import FEATURE_VERSION
from app.matching.features import FEATURE_NAMES, PROTECTED_ATTRIBUTES
from app.schemas.agent_io import MatchingAgentOutput
from app.schemas.agent_state import AgentState
from tests.matching_world import build_world

# Every protected or personal attribute the spec names (§11, §38), plus the job's visibility preference.
FORBIDDEN = {"name", "email", "phone", "gender", "religion", "race", "caste", "address", "dob", "date_of_birth",
             "age", "marital_status", "nationality", "gender_preference", "photo", "student_id", "job_id"}


@pytest.fixture(scope="module")
def world(embedder):
    return build_world(embedder)


@pytest.fixture
def msettings() -> Settings:
    return Settings(_env_file=None, environment="test", agent_timeout_seconds=30.0, tool_timeout_seconds=10.0,
                    tool_retry_backoff_seconds=0.0)


def context(world, embedder, msettings, **changes) -> P.MatchingContext:
    repo, cfg, cal = world
    return P.MatchingContext(repo=changes.pop("repo", repo), provider=embedder, cfg=changes.pop("cfg", cfg),
                             processed_dir=str(msettings.processed_data_dir), calibration_loader=lambda *_: cal,
                             **changes)


def run_agent(world, embedder, msettings, payload, cfg=None):
    repo, wcfg, cal = world
    agents = build_agents(settings=msettings, profile_repo=None, job_repo=None, embedder=embedder,
                          matching_repo=repo, matching_overrides={"cfg": cfg or wcfg,
                                                                  "calibration_loader": lambda *_: cal})
    state = AgentState(request_id="req-test", task_type="recommend_jobs_for_student")
    return asyncio.run(agents["matching_agent"].run(state, payload)), state


# -- the PII / protected-attribute exclusion test (Phase 7 report) -------------------------------------
def test_feature_vectors_never_contain_pii_or_protected_attributes(world, embedder, msettings):
    """Spec §11/§38: no name, email, phone, gender, religion, race or other protected/PII attribute is ever a
    matching feature. Checked on real pipeline output, and by changing a student's gender: no feature moves."""
    ctx = context(world, embedder, msettings, visibility_enabled=False)
    vectors = P.run_pipeline(ctx, "jobs_for_student", "S1", None, 6, with_versions=False)["features"]["features"]
    assert vectors, "the pipeline produced feature vectors"
    for vector in vectors.values():
        assert set(vector) == set(FEATURE_NAMES)
        words = {w for key in vector for w in key.lower().split("_")} | {k.lower() for k in vector}
        assert not words & FORBIDDEN, words & FORBIDDEN
        assert not set(vector) & PROTECTED_ATTRIBUTES
    assert "gender" not in StudentView.__dataclass_fields__  # the view features read has no gender at all
    # Phase 5 double-check: the job's visibility preference never reaches a feature. Same student, other gender:
    repo, cfg, cal = world
    other = type(repo)()
    other.index = repo.index
    for view, gender in repo.students():
        other.add_student(view, "male" if gender == "female" else "female")
    for job in repo.jobs():
        other.add_job(job)
    flipped = P.run_pipeline(context(world, embedder, msettings, repo=other, visibility_enabled=False),
                             "jobs_for_student", "S1", None, 6, with_versions=False)["features"]["features"]
    assert flipped == vectors


def test_the_feature_contract_matches_the_builder():
    assert FEATURE_VERSION == "features-v1" and len(FEATURE_NAMES) == 20


# -- Stage 1 ---------------------------------------------------------------------------------------------
def test_hard_rules_exclude_only_clear_failures_and_count_them(world, embedder, msettings):
    out = P.eligibility(context(world, embedder, msettings), "jobs_for_student", "S1", None)
    # S1: B.E (level 2), 1.5 years stated, India, female
    assert out["excluded"] == {"near_duplicate": 0, "not_visible": 1, "degree": 1, "experience": 1,
                               "mandatory_skill": 0}  # J5 male-only, J2 needs a master's, J4 needs 3 years
    assert out["eligible"] == ["J1", "J3", "J6"]
    assert out["tiers"] == {"J1": "same_country", "J3": "other_country", "J6": "other_country"}
    assert out["location_rule"].startswith("widened: only 1 eligible job(s) in India")


def test_same_country_only_when_enough_remain_and_no_tiering_without_a_country(world, embedder, msettings):
    repo, cfg, cal = world
    cfg1 = dataclasses.replace(cfg, eligibility={**cfg.eligibility, "min_same_country": 1})
    out = P.eligibility(context(world, embedder, msettings, cfg=cfg1), "jobs_for_student", "S1", None)
    assert out["eligible"] == ["J1"] and out["location_rule"].startswith("same country only")
    unknown = P.eligibility(context(world, embedder, msettings), "jobs_for_student", "S2", None)
    assert unknown["location_rule"] == "not applied: the student's country is not recorded"
    assert set(unknown["tiers"].values()) == {"any"}


def test_mandatory_skills_are_the_only_skill_based_exclusion(world, embedder, msettings):
    repo, cfg, cal = world
    cfg1 = dataclasses.replace(cfg, eligibility={**cfg.eligibility, "mandatory_skills": {"by_job_id": {"J1": ["Tableau"]},
                                                                                         "by_role": {}}})
    out = P.eligibility(context(world, embedder, msettings, cfg=cfg1), "jobs_for_student", "S1", None)
    assert "J1" not in out["eligible"] and out["excluded"]["mandatory_skill"] == 1
    # without configuration, a missing required skill (J6's "Data cleaning") is a score penalty, not a filter
    assert "J6" in P.eligibility(context(world, embedder, msettings), "jobs_for_student", "S1", None)["eligible"]


def test_visibility_applies_to_what_a_student_sees_never_to_ranking_candidates(world, embedder, msettings):
    on = P.eligibility(context(world, embedder, msettings), "jobs_for_student", "S1", None)
    off = P.eligibility(context(world, embedder, msettings, visibility_enabled=False), "jobs_for_student", "S1", None)
    assert "J5" not in on["eligible"] and off["excluded"]["not_visible"] == 0
    candidates = P.eligibility(context(world, embedder, msettings), "candidates_for_job", None, "J5")
    assert "S1" in candidates["eligible"]  # a male-only posting still sees every qualified candidate
    assert candidates["excluded"]["not_visible"] == 0 and candidates["excluded"]["degree"] == 1  # S3: diploma


def test_unknown_values_never_exclude(world, embedder, msettings):
    repo, cfg, cal = world
    student, _ = repo.student("S3")
    unknown = dataclasses.replace(student, degree_level=None)
    assert elig.failed_rules(unknown, repo.job("J1"), cfg) == []
    assert elig.failed_rules(student, repo.job("J1"), cfg) == ["degree"]


def test_stated_experience_is_the_longest_stated_duration():
    years, basis = stated_experience_years({"experience": ["Intern, 6 months", "Analyst 2 years"],
                                            "skills": [{"evidence": "18 months"}]})
    assert years == 2.0 and "2.00 years" in basis
    assert stated_experience_years({"experience": "Not detected", "skills": "Skill not detected"})[0] == 0.0


# -- Stages 2-6 ------------------------------------------------------------------------------------------
def test_retrieval_puts_same_country_jobs_first_then_scores(world, embedder, msettings):
    ctx = context(world, embedder, msettings)
    stage1 = P.eligibility(ctx, "jobs_for_student", "S1", None)
    stage2 = P.retrieval(ctx, "student", "S1", stage1["eligible"], stage1["tiers"], 10)
    assert stage2["top"][0]["id"] == "J1" and stage2["top"][0]["tier"] == "same_country"
    others = [r["score"] for r in stage2["top"][1:]]
    assert others == sorted(others, reverse=True)
    assert set(stage2["top"][0]["sections"]) <= {"skills->skills", "experience->responsibilities", "projects->description"}


def test_features_match_their_written_definitions(world, embedder, msettings):
    ctx = context(world, embedder, msettings)
    f = P.features(ctx, "student", "S1", ["J1", "J4"], {"J1": {}, "J4": {}})["features"]
    j1 = f["J1"]  # required Python, SQL (both matched); preferred Tableau (missing), Power BI (matched)
    assert j1["required_skill_match_score"] == 1.0 and j1["preferred_skill_match_score"] == 0.5
    assert j1["skill_match_score"] == pytest.approx(5 / 6, abs=1e-6)
    assert (j1["matched_skill_count"], j1["missing_skill_count"], j1["required_skill_count"]) == (3.0, 1.0, 2.0)
    assert j1["matched_skill_ratio"] == 0.75 and j1["preferred_skill_ratio"] == 0.5
    assert j1["transferable_skill_score"] is None  # no required skill is missing
    assert (j1["education_match"], j1["degree_match"], j1["branch_match"]) == (1.0, 1.0, 1.0)
    assert (j1["experience_match"], j1["experience_gap"]) == (1.0, 0.0)
    j4 = f["J4"]  # needs 3 years; S1 states 1.5
    assert j4["experience_match"] == 0.5 and j4["experience_gap"] == 1.5
    assert j1["semantic_similarity"] is None  # no evidence given: None, never a default


def test_stage_6_returns_the_skill_gap_report(world, embedder, msettings):
    gap = P.skill_gaps(context(world, embedder, msettings), "student", "S1", ["J2"], {})["gaps"]["J2"]
    assert gap["matched_skills"] == ["Python", "SQL"]
    assert [g["skill"] for g in gap["missing_required"]] == ["Deep learning"] and gap["missing_preferred"] == []
    assert gap["importance_model"].startswith("weighted_baseline_v1+")
    assert gap["missing_required"][0]["importance_score"] > 0


def test_weighted_baseline_renormalises_over_present_features(world):
    _, cfg, _ = world
    features = {f: None for f in FEATURE_NAMES} | {"skill_match_score": 1.0, "semantic_similarity": 0.5}
    out = predict(features, cfg)
    w1, w2 = cfg.weights["skill_match_score"], cfg.weights["semantic_similarity"]
    assert out["suitability"] == pytest.approx((w1 * 1.0 + w2 * 0.5) / (w1 + w2), abs=1e-6)
    assert sum(out["contributions"].values()) == pytest.approx(out["suitability"], abs=1e-5)
    assert out["overall"] == round(out["suitability"] * 100, 1)
    assert out["dimensions"]["certification_relevance"] == "Insufficient evidence"
    assert "education_match" in out["missing_features"]
    nothing = predict({f: None for f in FEATURE_NAMES}, cfg)
    assert nothing["suitability"] is None and nothing["overall"] == "Insufficient evidence"


def test_ranking_is_deterministic_whatever_the_input_order():
    a = P.ranking({"J9": 0.5, "J2": 0.5, "J5": 0.9, "J1": None}, 3)
    b = P.ranking({"J1": None, "J5": 0.9, "J2": 0.5, "J9": 0.5}, 3)
    assert a == b and [r["id"] for r in a["ranked"]] == ["J5", "J2", "J9"]


def test_one_role_can_not_fill_the_list():
    scores = {"J1": 0.9, "J2": 0.9, "J3": 0.8, "J4": 0.7}
    groups = {"J1": "web developer", "J2": "web developer", "J3": "web developer", "J4": "data analyst"}
    out = P.ranking(scores, 3, groups, max_per_group=1)
    assert [r["id"] for r in out["ranked"]] == ["J1", "J4"] and out["collapsed"] == 2


def test_calibrated_percentile_uses_mid_rank_for_ties():
    cal = Calibration("m", "v", {"semantic_similarity": (0.0, 0.0, 0.0, 1.0)}, 0.3, {})
    assert cal.percentile("semantic_similarity", 0.0) == 0.375 and cal.percentile("semantic_similarity", 2.0) == 1.0
    assert cal.percentile("semantic_similarity", None) is None


# -- the agent end to end --------------------------------------------------------------------------------
REQUIRED_VERSIONS = {"embedding_model", "embedding_version", "matching_model", "matching_model_version",
                     "feature_version", "dataset_version", "experiment_id"}


def test_agent_ranks_jobs_with_full_metadata_scores_and_trace(world, embedder, msettings):
    out, state = run_agent(world, embedder, msettings, {"mode": "jobs_for_student", "student_id": "S1", "top_k": 3})
    assert out.status == "success", out.errors
    data = MatchingAgentOutput.model_validate(out.data)
    assert data.funnel.checked == 6 and data.funnel.eligible == 3 and data.funnel.returned == 3
    assert [m.rank for m in data.matches] == [1, 2, 3]
    for m in data.matches:
        assert set(m.model_versions.model_dump()) == REQUIRED_VERSIONS
        assert m.model_versions.matching_model == "weighted_baseline_v1"
        assert m.model_versions.experiment_id == "test-experiment"
        assert len(m.employability.dimensions) == 8 and "PLACEHOLDER" in m.employability.method
        assert set(m.trace.features) == set(FEATURE_NAMES)
        assert m.semantic_similarity != m.trace.retrieval["raw_score"]  # a calibrated percentile, not raw cosine
        assert m.employability.overall == round(m.suitability_score * 100, 1)
    assert state.model_versions["matching_model"] == "weighted_baseline_v1"
    assert state.model_versions["feature_version"] == "features-v1"


def test_agent_ranks_candidates_for_a_job(world, embedder, msettings):
    out, _ = run_agent(world, embedder, msettings, {"mode": "candidates_for_job", "job_id": "J1"})
    assert out.status == "success", out.errors
    data = MatchingAgentOutput.model_validate(out.data)
    assert {m.job_id for m in data.matches} == {"J1"} and "S1" in {m.student_id for m in data.matches}
    assert all(set(m.summary) >= {"branch", "domain"} for m in data.matches)


def test_pair_mode_reports_failed_rules_instead_of_hiding_the_job(world, embedder, msettings):
    out, _ = run_agent(world, embedder, msettings, {"mode": "jobs_for_student", "student_id": "S1", "job_id": "J4"})
    data = MatchingAgentOutput.model_validate(out.data)
    assert data.mode == "pair" and len(data.matches) == 1 and data.matches[0].job_id == "J4"
    assert data.pair_check == {"eligible": False, "failed_rules": ["experience"],
                               "note": "reported, not applied: this job was asked for by name"}


def test_an_unregistered_configuration_returns_no_prediction(world, embedder, msettings):
    _, cfg, _ = world
    out, _ = run_agent(world, embedder, msettings, {"mode": "jobs_for_student", "student_id": "S1"},
                       cfg=dataclasses.replace(cfg, registration={}))
    assert out.status == "failed" and out.termination_reason == "invalid_input"
    assert "not registered" in out.errors[-1].message
    assert out.data in ("Insufficient evidence", "Not yet evaluated")


def test_an_unknown_student_is_rejected_without_retries(world, embedder, msettings):
    out, _ = run_agent(world, embedder, msettings, {"mode": "jobs_for_student", "student_id": "S404"})
    assert out.termination_reason == "invalid_input" and "no structured profile" in out.errors[-1].message
    assert out.tool_calls[0].attempts == 1


# -- configuration ---------------------------------------------------------------------------------------
def test_the_shipped_matching_config_is_valid_and_versioned():
    cfg = load_matching_config()
    assert abs(sum(cfg.weights.values()) - 1) < 1e-9 and all(cfg.reasons.values())
    assert cfg.matching_model_version.startswith("weighted_baseline_v1+") and len(cfg.sha256) == 64


def test_a_protected_attribute_can_not_be_weighted(tmp_path):
    import yaml

    from app.matching.config import MATCHING_CONFIG_PATH
    from app.matching.features import ProtectedFeatureError

    raw = yaml.safe_load(MATCHING_CONFIG_PATH.read_text())
    raw["weights"]["gender"] = {"weight": 0.0, "reason": "never"}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ProtectedFeatureError):
        load_matching_config(path)
    raw["weights"].pop("gender")
    raw["weights"]["skill_match_score"]["weight"] = 0.9
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(MatchingConfigError, match="sum to 1"):
        load_matching_config(path)


# -- dedup v2 (company-less uploads) ---------------------------------------------------------------------
def test_uploaded_duplicates_without_a_company_are_caught():
    from app.jobs.dedup import find_near_duplicates, match_existing

    jd = {"company_name": None, "job_title": "Junior Data Analyst", "job_description": "Analyse sales data weekly.",
          "responsibilities": "Build dashboards.", "skills": "Python, SQL"}
    a, b = {**jd, "job_id": "JU1"}, {**jd, "job_id": "JU2"}
    assert find_near_duplicates([a, b]) == {"JU2": {"near_duplicate_of": "JU1", "similarity": 1.0}}
    assert match_existing({**jd, "job_id": None}, [a])["near_duplicate_of"] == "JU1"
    assert match_existing({**jd, "job_id": None, "job_title": "Data Engineer"}, [a]) is None
    # a company-less upload never merges with a dataset posting that has a company
    assert match_existing({**jd, "job_id": None}, [{**jd, "job_id": "J1", "company_name": "Acme"}]) is None
