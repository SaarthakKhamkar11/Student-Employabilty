"""Phase 9: skill-gap analysis (src/scoring/skill_gap_analysis.py, docs/adr/0016)."""

import asyncio
from dataclasses import replace

import pytest

from app.agents import build_agents
from app.core.config import Settings
from app.matching import pipeline as P
from app.matching.feature_builder import build_features
from app.matching.predictor import BaselinePredictor
from app.matching.skill_gap import compare_skills
from app.profile.skills import normalize_skills
from app.schemas.agent_io import MatchingAgentOutput, SkillGapReport
from app.schemas.agent_state import AgentState
from src.scoring import skill_gap_analysis as G
from tests.matching_world import build_world


@pytest.fixture(scope="module")
def world(embedder):
    return build_world(embedder)


def ctx_for(world, embedder, **changes):
    repo, cfg, cal = world
    changes.setdefault("matching_model_override", "weighted_baseline_v1")  # the synthetic world is scored by Baseline 3
    return P.MatchingContext(repo=repo, provider=embedder, cfg=cfg, processed_dir=str(Settings(_env_file=None).processed_data_dir),
                             calibration_loader=lambda *_: cal, **changes)


class NameVectors:
    """Stub skill vectors: get() returns the names; G.cosine is patched to a lookup table."""
    dimension = 1

    def get(self, names):
        return {n: n for n in names}


def with_sims(monkeypatch, table):
    monkeypatch.setattr(G, "cosine", lambda a, b, _: table.get((a, b), 0.0))


T = G.Thresholds(transferable=0.6, partial=0.4, model_name="test", tuned_on="test")


def student(skills):
    from app.matching.entities import StudentView
    return StudentView(student_id="S", skills=tuple(skills), degree_code=None, degree_level=None, branch=None,
                       domain=None, experience_years=0.0, experience_basis="", country=None)


def job(required, preferred=()):
    from app.matching.entities import JobView
    return JobView(job_id="J", title="t", role="r", skills=tuple([(s, "required") for s in required] +
                                                                 [(s, "preferred") for s in preferred]),
                   degree_code=None, degree_level=None, min_experience_years=None, country=None, near_duplicate_of=None,
                   visibility="both", source="test")


# -- the shared normaliser ---------------------------------------------------------------------------------------
def test_raw_skills_go_through_the_phase_4_normaliser():
    raw = ["PowerBI", "JS", "Python"]
    assert G.from_raw_skills(raw) == tuple(s.canonical for s in normalize_skills(raw))


def test_matching_uses_the_shared_skill_key():
    matched, missing = G.split_skills(("python", "SQL "), (("Python", "required"), ("Excel", "preferred")))
    assert matched == ["Python"] and missing == [("Excel", "preferred")]


# -- importance: counterfactual re-scoring --------------------------------------------------------------------------
def test_importance_is_the_models_own_score_difference():
    calls = {}

    def rescore(variants):
        calls.update(variants)
        return {k: 0.1 * len(v) + (0.2 if "Docker" in v else 0.0) for k, v in variants.items()}

    imp = G.counterfactual_importance(("Python",), ["Docker", "Excel"], rescore)
    assert calls == {G.ACTUAL: ("Python",), "Docker": ("Python", "Docker"), "Excel": ("Python", "Excel")}
    assert imp == {"Docker": pytest.approx(0.3), "Excel": pytest.approx(0.1)}


def test_importance_is_none_when_the_model_cannot_score():
    assert G.counterfactual_importance(("A",), ["B"], lambda v: {k: None for k in v}) == {"B": None}


def test_importance_matches_an_independent_rescore_with_the_real_pipeline(world, embedder):
    ctx = ctx_for(world, embedder)
    s, j = ctx.student("S1")[0], ctx.job("J2")
    sims = P.similarity(ctx, "student", "S1", ["J2"], {})["percentiles"]["J2"]
    report = P.skill_gaps(ctx, "student", "S1", ["J2"], {"J2": sims})["gaps"]["J2"]
    gap = report["missing_required"][0]
    related = ctx.calibration().transferable_threshold

    def suitability(skills):
        v = replace(s, skills=skills)
        f = build_features(v, j, compare_skills(v.skills, j.skills, ctx.skill_vectors(), related), sims, ctx.cfg)
        return ctx.predictor.predict_many({"x": f})["x"]["suitability"]

    assert gap["importance_score"] == round(suitability(s.skills + (gap["skill"],)) - suitability(s.skills), 6)
    assert report["importance_model"] == ctx.predictor.version


def test_importance_follows_the_active_model_not_a_constant(world, embedder, monkeypatch):
    class Doubling(BaselinePredictor):
        def predict_many(self, vectors):
            out = super().predict_many(vectors)
            return {k: {**p, "suitability": None if p["suitability"] is None else 2 * p["suitability"]}
                    for k, p in out.items()}

    base = ctx_for(world, embedder)
    doubled = ctx_for(world, embedder, predictor=Doubling(base.cfg))
    a = P.skill_gaps(base, "student", "S1", ["J2"], {})["gaps"]["J2"]["missing_required"][0]["importance_score"]
    b = P.skill_gaps(doubled, "student", "S1", ["J2"], {})["gaps"]["J2"]["missing_required"][0]["importance_score"]
    assert b == pytest.approx(2 * a, abs=2e-6)


# -- relation tiers ----------------------------------------------------------------------------------------------
def report_for(monkeypatch, sims, thresholds=T, importance=None):
    with_sims(monkeypatch, sims)
    imp = importance or {}
    return G.analyse(student(["Java", "Sales"]), job(["Java programming", "Sales strategy", "Cooking"], ["Excel"]),
                     rescore=lambda v: {k: imp.get(k, 0.0) for k in v} | {G.ACTUAL: 0.0}, vectors=NameVectors(),
                     thresholds=thresholds, importance_model="m+1", similarity_model="e 1")


def test_transferable_partial_and_unrelated_tiers(monkeypatch):
    r = report_for(monkeypatch, {("Java programming", "Java"): 0.69, ("Sales strategy", "Sales"): 0.41,
                                 ("Cooking", "Sales"): 0.39, ("Excel", "Java"): 0.1})
    by = {g["skill"]: g for g in r["missing_required"]}
    assert by["Java programming"]["transferable_candidate"] == "Java"
    assert by["Java programming"]["transferable_similarity"] == 0.69
    assert by["Sales strategy"]["transferable_candidate"] is None and by["Cooking"]["transferable_candidate"] is None
    assert r["partially_related"] == [{"skill": "Sales strategy", "requirement": "required",
                                       "related_student_skill": "Sales", "similarity": 0.41}]
    assert [g["skill"] for g in r["missing_preferred"]] == ["Excel"]
    SkillGapReport.model_validate(r)


def test_threshold_boundaries_are_inclusive_below(monkeypatch):
    r = report_for(monkeypatch, {("Java programming", "Java"): 0.6, ("Sales strategy", "Sales"): 0.4})
    by = {g["skill"]: g for g in r["missing_required"]}
    assert by["Java programming"]["transferable_candidate"] == "Java"
    assert [p["skill"] for p in r["partially_related"]] == ["Sales strategy"]


def test_without_thresholds_for_the_model_no_tiers_are_guessed(monkeypatch):
    r = report_for(monkeypatch, {("Java programming", "Java"): 0.9}, thresholds=None)
    assert all(g["transferable_candidate"] is None for g in r["missing_required"]) and r["partially_related"] == []
    assert r["thresholds"] == {} and "Not yet evaluated" in r["method"]


def test_missing_skills_are_ranked_by_importance(monkeypatch):
    r = report_for(monkeypatch, {}, importance={"Cooking": 0.05, "Sales strategy": 0.2, "Java programming": 0.1})
    assert [g["skill"] for g in r["missing_required"]] == ["Sales strategy", "Java programming", "Cooking"]


def test_the_shipped_thresholds():
    t = G.load_thresholds("tfidf_baseline")
    assert (t.transferable, t.partial) == (0.6, 0.4)
    assert G.load_thresholds("some_unconfigured_model") is None


# -- the contract and the agent -----------------------------------------------------------------------------------
def test_not_detected_language_is_kept():
    assert "not detected" in SkillGapReport.model_fields["note"].default
    assert "not proof" in SkillGapReport.model_fields["note"].default


def test_the_agent_returns_skill_gap_reports_with_importance(world, embedder):
    repo, cfg, cal = world
    settings = Settings(_env_file=None, environment="test", tool_retry_backoff_seconds=0.0)
    agents = build_agents(settings=settings, profile_repo=None, job_repo=None, embedder=embedder, matching_repo=repo,
                          matching_overrides={"cfg": cfg, "calibration_loader": lambda *_: cal})
    state = AgentState(request_id="req-p9", task_type="recommend_jobs_for_student")
    out = asyncio.run(agents["matching_agent"].run(state, {"mode": "jobs_for_student", "student_id": "S1", "top_k": 3}))
    assert out.status == "success", out.errors
    for m in MatchingAgentOutput.model_validate(out.data).matches:
        assert isinstance(m.skill_gap, SkillGapReport)
        assert m.skill_gap.importance_model == m.model_versions.matching_model_version
        assert all(g.importance_score is not None for g in m.skill_gap.missing_required)
