"""Phase 11: the Explainability Agent never returns an explanation it can't trace to evidence (docs/adr/0019)."""

import asyncio
import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from app.agents import build_agents
from app.core.config import Settings
from app.explainability import prose
from app.explainability.evidence import LABELS, build_evidence
from app.explainability.validation import validate
from app.matching.features import FEATURE_NAMES
from app.schemas.agent_io import (
    ExplainabilityAgentOutput, MatchResult, PartiallyRelatedSkill, ProseDraft, SkillGap, SkillGapReport,
)
from app.schemas.agent_state import AgentState
from src.scoring import shap_explainer as S
from tests.helpers import FakeLLM

VERSION = "lightgbm+known0000000"
# -- a known feature vector and known SHAP values -----------------------------------------------------------------
FEATURES = {f: 0.0 for f in FEATURE_NAMES} | {
    "skill_match_score": 0.571429, "required_skill_match_score": 0.666667, "preferred_skill_match_score": 0.0,
    "semantic_similarity": 0.91, "education_match": 1.0, "degree_match": 1.0, "branch_match": 1.0,
    "experience_match": 1.0, "experience_gap": 0.0, "project_relevance": 0.88, "certification_relevance": 0.4,
    "career_interest_alignment": 0.62, "transferable_skill_score": 0.0, "matched_skill_count": 2.0,
    "missing_skill_count": 2.0, "required_skill_count": 3.0, "matched_skill_ratio": 0.5,
    "preferred_skill_ratio": 0.0, "role_similarity": 0.55, "domain_similarity": 0.7}
SHAP = {f: 0.0 for f in FEATURE_NAMES} | {
    "branch_match": 1.2, "semantic_similarity": 0.8, "project_relevance": 0.45, "career_interest_alignment": 0.05,
    "certification_relevance": 0.01, "required_skill_match_score": -0.3, "preferred_skill_ratio": -0.25,
    "domain_similarity": -0.02}
BASE = -1.1
SELECTED = {"branch_match": 1, "semantic_similarity": 1, "project_relevance": 1, "required_skill_match_score": -1,
            "preferred_skill_ratio": -1}  # |SHAP| share >= 5%, top 3 per direction
SUITABILITY = round(1 / (1 + math.exp(-(BASE + sum(SHAP.values())))), 6)
REPORT = SkillGapReport(
    matched_skills=["Python", "SQL"],
    missing_required=[SkillGap(skill="Power BI", importance_score=0.21, transferable_candidate=None,
                               transferable_similarity=None)],
    missing_preferred=[SkillGap(skill="Tableau", importance_score=0.05, transferable_candidate=None,
                                transferable_similarity=None)],
    partially_related=[], importance_model=VERSION)


class KnownModel:
    """An ExplainedModel stand-in whose SHAP values are the known ones: margin = base + sum(SHAP)."""

    key, version, is_baseline, output_space, explainer_name = "lightgbm", VERSION, False, "log-odds", "known (test)"

    def __init__(self, values=SHAP, base=BASE):
        self.values, self.base, self.predictor = values, base, self

    def explainer(self):
        return self

    def model(self):
        return self

    def explain(self, frame):
        return np.array([[self.values[f] for f in FEATURE_NAMES]]), self.base

    def margin(self, frame):
        return np.array([self.base + sum(self.values.values())])

    def score(self, frame):
        return 1 / (1 + np.exp(-self.margin(frame)))


def match(**changes) -> MatchResult:
    data = {"student_id": "S1", "job_id": "J1", "rank": 1, "suitability_score": SUITABILITY,
            "semantic_similarity": 0.91, "skill_gap": REPORT.model_dump(),
            "summary": {"job_title": "Data Analyst", "role": "Data Analyst", "country": "India"},
            "trace": {"eligibility": {}, "retrieval": {"rank": 1, "raw_score": 0.81}, "similarity_raw": {},
                      "features": FEATURES, "normalised": {}, "weights": {}, "contributions": {},
                      "weight_covered": 1.0, "missing_features": []},
            "model_versions": {"embedding_model": "tfidf_baseline", "embedding_version": "tfidf-test",
                               "matching_model": "lightgbm", "matching_model_version": VERSION,
                               "feature_version": "features-v1", "dataset_version": "test",
                               "experiment_id": "test-experiment"}}
    return MatchResult.model_validate(data | changes)


# What a faithful LLM writes for this evidence (ids: P1 Python, P2 SQL, P3 education, P4 branch, P5 semantic,
# P6 project; G1 Power BI, G2 Tableau; M1 branch, M2 semantic, M3 project, M4 required-skill match, M5 ratio).
GOOD = {
    "positive_evidence": [
        {"text": "Python requirement satisfied", "evidence_ids": ["P1"]},
        {"text": "SQL requirement satisfied", "evidence_ids": ["P2"]},
        {"text": "Education requirement satisfied", "evidence_ids": ["P3"]},
        {"text": "Your branch fits the job's role family", "evidence_ids": ["P4"]},
        {"text": "Semantic similarity at the 91st percentile and project relevance at the 88th percentile of "
                 "student-job pairs", "evidence_ids": ["P5", "P6"]}],
    "potential_gaps": [{"text": "Power BI not detected (required)", "evidence_ids": ["G1"]},
                       {"text": "Tableau not detected (preferred)", "evidence_ids": ["G2"]}],
    "model_evidence": [
        {"text": "Branch fit contributed positively", "evidence_ids": ["M1"]},
        {"text": "Semantic similarity and project relevance raised the score", "evidence_ids": ["M2", "M3"]},
        {"text": "Required-skill match reduced the score", "evidence_ids": ["M4"]},
        {"text": "Preferred-skill ratio reduced the score", "evidence_ids": ["M5"]}],
    "how_to_improve": [{"text": "Develop Power BI capability, then Tableau.", "evidence_ids": ["G1", "G2"]}],
}
# Plausible-sounding but unbacked: every line here is the kind of hallucination the validator must catch.
BROKEN = {
    "positive_evidence": [
        {"text": "Power BI requirement satisfied", "evidence_ids": ["P1"]},  # Power BI was NOT detected
        {"text": "SQL requirement satisfied", "evidence_ids": ["P2"]},
        {"text": "Relevant data-analysis project", "evidence_ids": ["P6"]},  # no anchor, invented content
        {"text": "Education requirement satisfied", "evidence_ids": ["P3"]},
        {"text": "Branch fits the job", "evidence_ids": ["P4"]},
        {"text": "Semantic similarity at the 95th percentile", "evidence_ids": ["P5"]}],  # real value: 91st
    "potential_gaps": [{"text": "Excel not detected", "evidence_ids": ["G1"]},  # not in missing lists
                       {"text": "The student lacks Tableau", "evidence_ids": ["G2"]}],  # "lacks" wording
    "model_evidence": [
        {"text": "Certification relevance contributed positively", "evidence_ids": ["M1"]},  # not a contributor
        {"text": "Semantic similarity contributed positively", "evidence_ids": ["M2"]},
        {"text": "Project relevance contributed positively", "evidence_ids": ["M3"]},
        {"text": "Required-skill match raised the score", "evidence_ids": ["M4"]},  # wrong sign: SHAP -0.3
        {"text": "Missing required visualization skill reduced score", "evidence_ids": ["M5"]}],  # no such feature
    "how_to_improve": [{"text": "Take an advanced Excel course.", "evidence_ids": ["X9"]}],  # invented id
}


def run(llm=None, model=None, m=None, mode="match", settings=None):
    settings = settings or Settings(_env_file=None, environment="test", tool_retry_backoff_seconds=0.0,
                                    agent_timeout_seconds=60.0, tool_timeout_seconds=30.0)
    agents = build_agents(settings=settings, llm=llm, profile_repo=None, job_repo=None, matching_repo=None,
                          learning_repo=None, embedder=_Embedder(), fallback_embedder=None,
                          explain_model_loader=(lambda *_: model) if model is not None else None)
    state = AgentState(request_id="req-p11", task_type="explain_match")
    payload = {"mode": mode} if mode == "global" else {"match": (m or match()).model_dump()}
    out = asyncio.run(agents["explainability_agent"].run(state, payload))
    return out, ExplainabilityAgentOutput.model_validate(out.data) if isinstance(out.data, dict) else None


class _Embedder:
    model_name = "tfidf_baseline"

    def describe(self):
        return "tfidf-test", 1


def sections(text: str) -> dict[str, list[str]]:
    out, current = {}, None
    for line in text.splitlines():
        if line.endswith(":") and not line.startswith(("+", "-")):
            current = line[:-1]
            out[current] = []
        elif line.strip() and current:
            out[current].append(line)
    return out


def labelled_features(line: str) -> set[str]:
    norm = line.lower().replace("-", " ")
    return {f for f, label in LABELS.items() if label.replace("-", " ") in norm}


# -- THE evidence-traceability test -------------------------------------------------------------------------------
def test_every_bullet_corresponds_to_a_real_contributing_feature_or_input(capsys):
    llm = FakeLLM(GOOD)
    out, result = run(llm=llm, model=KnownModel())
    assert out.status == "success", out.errors
    exp = result.match_explanation
    assert exp.source == "llm_validated" and exp.generation["validations"][0]["valid"]
    text = exp.text
    print("\n" + text)  # the report shows it (pytest -s)
    s = sections(text)
    assert text.startswith(f"Suitability Score: {round(SUITABILITY * 100)}%")
    # Model Evidence: grep each bullet for the real feature names and check the +/- against the real SHAP sign.
    seen = set()
    *bullets, total = s["Model Evidence"]
    # the last line is computed, never generated: base value + every SHAP value = the model's output
    assert total == (f"= Base value {BASE:+.3f} log-odds (the model's starting point before this pair's features) "
                     f"+ all {sum(1 for v in SHAP.values() if v)} contributions {sum(SHAP.values()):+.3f} = "
                     f"{BASE + sum(SHAP.values()):+.3f} log-odds")
    for bullet in bullets:
        named = labelled_features(bullet)
        assert named, f"model-evidence bullet names no real feature: {bullet}"
        for feature in named:
            assert feature in SELECTED, f"{feature} is not a selected contributor ({bullet})"
            assert SELECTED[feature] == (1 if bullet.startswith("+") else -1), f"sign mismatch: {bullet}"
            assert math.copysign(1, SHAP[feature]) == SELECTED[feature]
        seen |= named
    assert seen == set(SELECTED), "every selected contributor is explained, and nothing else"
    # Potential Gaps: only skills in Phase 9's missing lists.
    missing = {g.skill for g in REPORT.missing_required + REPORT.missing_preferred}
    for bullet in s["Potential Gaps"]:
        assert any(skill in bullet for skill in missing), bullet
        assert not any(skill in bullet for skill in REPORT.matched_skills), bullet
    # Positive Evidence: matched skills, or a feature whose value really shows the claim.
    for bullet in s["Positive Evidence"]:
        backed = [k for k in REPORT.matched_skills if k in bullet]
        backed += [f for f in labelled_features(bullet) if FEATURES[f] >= 0.75]
        backed += ["education_match"] * (FEATURES["education_match"] == 1 and "Education" in bullet)
        backed += ["branch_match"] * (FEATURES["branch_match"] == 1 and "branch" in bullet.lower())
        assert backed, f"positive bullet without evidence: {bullet}"
        assert not any(skill in bullet for skill in missing), bullet
    assert any("Power BI" in line for line in s["How to Improve"])
    assert exp.contributions["values"] == SHAP and exp.model_versions.matching_model_version == VERSION
    # the LLM computed nothing: no score, no SHAP value appears in what it was asked to write
    assert '"shap"' not in llm.calls[0]["prompt"] and str(SUITABILITY) not in llm.calls[0]["prompt"]


def test_a_deliberately_broken_explanation_is_caught_retried_once_and_never_returned(capsys):
    llm = FakeLLM(BROKEN)  # both attempts hallucinate
    out, result = run(llm=llm, model=KnownModel())
    exp = result.match_explanation
    first = exp.generation["validations"][0]
    print("\nvalidation errors on the broken draft:\n  " + "\n  ".join(first["errors"]))
    print("\nreturned instead:\n" + exp.text)
    assert not first["valid"]
    joined = "\n".join(first["errors"])
    for expected in ("names 'Power BI', which none of its cited evidence involves",  # positive claim of a gap
                     "cites P6 but does not name 'project relevance'",
                     "states number(s) ['95']",
                     "cites G1 but does not name 'Power BI'",
                     "never something the student lacks",
                     "names 'certification relevance'",
                     "its SHAP evidence lowered the score, the line must say so",
                     "cites evidence that does not exist: ['X9']"):
        assert expected in joined, expected
    assert len(llm.calls) == 2 and "STRICT MODE" in llm.calls[1]["prompt"]  # retried once, more strictly
    assert out.status == "partial" and out.errors[0].code == "explanation_fallback"
    assert exp.source == "structured_fallback" and exp.generation["llm_attempts"] == 2
    assert exp.generation["validations"][-1] == {"attempt": 0, "valid": True, "errors": [],
                                                 "lines_checked": exp.generation["validations"][-1]["lines_checked"]}
    for invented in ("Excel", "lacks", "Certification relevance", "95th", "visualization"):
        assert invented not in exp.text  # nothing from the rejected drafts reaches the output


def test_a_draft_that_fails_once_is_retried_and_the_valid_retry_is_used():
    out, result = run(llm=FakeLLM(BROKEN, GOOD), model=KnownModel())
    exp = result.match_explanation
    assert out.status == "success"
    assert [v["valid"] for v in exp.generation["validations"]] == [False, True]
    assert exp.source == "llm_validated" and exp.potential_gaps[0] == "- Power BI not detected (required)"


def test_without_an_llm_the_structured_evidence_is_returned_and_validated():
    out, result = run(llm=None, model=KnownModel())
    exp = result.match_explanation
    assert out.status == "success" and exp.source == "structured_fallback"
    assert "no LLM configured" in exp.generation["fallback_reason"]
    assert exp.model_evidence[0].startswith("+ Branch fit") and "SHAP +1.200 log-odds" in exp.model_evidence[0]
    assert exp.potential_gaps[0].startswith("- Power BI not detected (required skill; model importance +0.210")
    assert exp.how_to_improve == ["Develop Power BI: required skills not detected, most important to the model first."]


def test_an_llm_error_counts_as_a_failed_attempt_then_falls_back():
    from app.llm.client import LLMTimeoutError

    out, result = run(llm=FakeLLM(LLMTimeoutError("no response within 25s")), model=KnownModel())
    exp = result.match_explanation
    assert exp.source == "structured_fallback" and "LLMTimeoutError" in exp.generation["fallback_reason"]


def test_shap_from_a_different_model_than_the_prediction_is_refused():
    other = KnownModel()
    other.version = "lightgbm+other0000000"
    out, result = run(model=other)
    assert out.status == "failed" and out.termination_reason == "invalid_input"
    assert "the prediction came from" in out.errors[0].message


def test_shap_that_does_not_reproduce_the_predicted_score_is_refused():
    out, _ = run(model=KnownModel(base=0.0))  # same features, another score: not the prediction being explained
    assert out.status == "failed" and "not the same prediction" in out.errors[0].message


# -- the validator on its own -------------------------------------------------------------------------------------
def bundle(report=REPORT, shap=SHAP, features=FEATURES):
    from app.tools.explainability_tools import ContributionsOut  # noqa: F401 (contract the tool returns)

    total = sum(abs(v) for v in shap.values())
    picked = sorted(((f, v) for f, v in shap.items() if v and abs(v) / total >= 0.05), key=lambda kv: -abs(kv[1]))
    contrib = {"positive": [{"feature": f, "shap": v, "share": round(abs(v) / total, 4)} for f, v in picked if v > 0][:3],
               "negative": [{"feature": f, "shap": v, "share": round(abs(v) / total, 4)} for f, v in picked if v < 0][:3],
               "values": shap, "output_space": "log-odds"}
    return build_evidence(suitability=SUITABILITY, features=features, report=report, contributions=contrib,
                          semantic_raw=0.81, student=None, job=None, positive_similarity_percentile=0.75)


def test_mixed_directions_and_left_out_evidence_fail():
    draft = json.loads(json.dumps(GOOD))
    draft["model_evidence"] = [{"text": "Branch fit raised and required-skill match reduced the score",
                                "evidence_ids": ["M1", "M4"]}]
    draft["potential_gaps"] = draft["potential_gaps"][:1]
    errors = "\n".join(validate(bundle(), ProseDraft.model_validate(draft), 1).errors)
    assert "mixes contributions that raised and lowered" in errors
    assert "potential_gaps: leaves out evidence ['G2']" in errors
    assert "model_evidence: leaves out evidence ['M2', 'M3', 'M5']" in errors


def test_improvement_must_target_the_most_important_gap():
    draft = json.loads(json.dumps(GOOD))
    draft["how_to_improve"] = [{"text": "Develop Tableau.", "evidence_ids": ["G2"]}]
    assert "must address the most important gap (['G1'])" in "\n".join(
        validate(bundle(), ProseDraft.model_validate(draft), 1).errors)


@pytest.mark.parametrize("report", [
    REPORT,
    SkillGapReport(matched_skills=[], missing_required=[], missing_preferred=[], partially_related=[]),
    SkillGapReport(  # digits in skill names, a transferable candidate, a partially related skill, tied importance
        matched_skills=["Web 2.0", "C++"],
        missing_required=[SkillGap(skill="Python 3", importance_score=0.083, transferable_candidate="Python",
                                   transferable_similarity=0.6465),
                          SkillGap(skill="AWS S3", importance_score=0.083, transferable_candidate=None,
                                   transferable_similarity=None)],
        missing_preferred=[SkillGap(skill="Deep Learning", importance_score=None, transferable_candidate=None,
                                    transferable_similarity=None)],
        partially_related=[PartiallyRelatedSkill(skill="Deep Learning", requirement="preferred",
                                                 related_student_skill="Supervised Learning", similarity=0.4855)]),
], ids=["data-analyst", "no-skills", "edge-cases"])
def test_the_structured_fallback_always_passes_its_own_validation(report):
    b = bundle(report)
    result = validate(b, prose.structured_draft(b), attempt=0)
    assert result.valid, result.errors
    assert sum(len(v) for v in prose.rendered_lines(b, prose.structured_draft(b)).values()) >= len(b.model)


def test_ties_are_not_presented_as_a_ranking():
    b = bundle(SkillGapReport(matched_skills=[], missing_required=[
        SkillGap(skill=s, importance_score=0.083, transferable_candidate=None, transferable_similarity=None)
        for s in ("Cloud infrastructure", "Scalability")], missing_preferred=[], partially_related=[]))
    assert "equal model importance (their order is not a ranking)" in prose.structured_draft(b).how_to_improve[0].text


# -- the real Phase 8 models (no database) ------------------------------------------------------------------------
def real_rows(model, n=5):
    frame = S.matrix_for(model.predictor.manifest)
    return frame[frame["split"] == "validation"].iloc[:n]


# Since the features-v2 selections (2026-10-15) the active E5 model is logistic regression and the TF-IDF fallback is
# Baseline 3 (no SHAP needed), so the tree path is checked with an explicit override on the fallback's trained artifacts.
@pytest.mark.parametrize("embedding, override, explainer", [("e5_base_v2", None, "LinearExplainer"),
                                                            ("tfidf_baseline", "lightgbm", "TreeExplainer"),
                                                            ("tfidf_baseline", "logistic_regression", "LinearExplainer")])
def test_real_models_get_the_explainer_their_type_needs_and_values_add_up(embedding, override, explainer):
    model = S.load_model(embedding, override)
    assert explainer in model.explainer_name, (model.key, model.explainer_name)
    for _, row in real_rows(model).iterrows():
        features = {f: (None if pd.isna(row[f]) else float(row[f])) for f in FEATURE_NAMES}
        out = S.explain_pair(model, features)
        assert out["additivity_error"] <= S.ADDITIVITY_TOLERANCE
        assert abs(out["suitability"] - model.predictor.predict_many({"c": features})["c"]["suitability"]) <= 1e-6


def test_global_and_individual_are_separate_paths_with_separate_outputs():
    # the fallback's trained Logistic Regression (the active fallback is Baseline 3): fast enough for a unit test
    model = S.load_model("tfidf_baseline", "logistic_regression")
    out, result = run(mode="global", model=model)
    assert out.status == "success", out.errors
    assert result.match_explanation == "Not yet evaluated" and result.global_explanation.kind == "global"
    g = result.global_explanation
    assert [r.feature for r in g.importance_ranking] and len(g.importance_ranking) == len(FEATURE_NAMES)
    assert g.rows == int((S.matrix_for(model.predictor.manifest)["split"] == "validation").sum())
    assert set(g.distributions) == set(FEATURE_NAMES) and {"mean", "std", "histogram"} <= set(g.distributions["branch_match"])
    assert len(g.summary_plot["features"]["branch_match"]["shap"]) == g.summary_plot["rows"] <= 300
    assert g.additivity_max_abs_error <= S.ADDITIVITY_TOLERANCE
    ranks = [r.mean_abs_shap for r in g.importance_ranking]
    assert ranks == sorted(ranks, reverse=True)
    _, result_m = run(model=KnownModel())  # and the match path returns no global explanation
    assert result_m.global_explanation == "Not yet evaluated" and result_m.match_explanation.student_id == "S1"


def test_the_matching_agents_own_result_is_explained_end_to_end(embedder):
    """Baseline 3 on the synthetic world: contributions are the model's own (exact), the student and job records
    add detail, and the agent explains the Matching Agent's real MatchResult."""
    from tests.matching_world import build_world

    repo, cfg, cal = build_world(embedder)
    settings = Settings(_env_file=None, environment="test", tool_retry_backoff_seconds=0.0, agent_timeout_seconds=60.0,
                        tool_timeout_seconds=30.0, matching_model="weighted_baseline_v1")
    agents = build_agents(settings=settings, profile_repo=None, job_repo=None, embedder=embedder, matching_repo=repo,
                          learning_repo=None, matching_overrides={"cfg": cfg, "calibration_loader": lambda *_: cal})
    state = AgentState(request_id="req-p11-e2e", task_type="explain_match")
    matched = asyncio.run(agents["matching_agent"].run(state, {"mode": "jobs_for_student", "student_id": "S1",
                                                                 "job_id": "J1"}))
    top = matched.data["matches"][0]
    out = asyncio.run(agents["explainability_agent"].run(state, {"match": top}))
    assert out.status == "success", out.errors
    exp = ExplainabilityAgentOutput.model_validate(out.data).match_explanation
    assert abs(sum(exp.contributions["values"].values()) - top["suitability_score"]) <= 1e-6
    assert "not SHAP" in exp.contributions["explainer"]
    kinds = {i.kind for i in exp.evidence.positive}
    assert "matched_skill" in kinds and "experience" not in kinds  # J1 states no minimum: nothing to satisfy
    assert re.search(r"Suitability Score: \d+%", exp.text)


def test_the_schema_sent_to_gemini_is_inlined_but_replies_are_still_validated_in_full():
    from pydantic import BaseModel, Field

    from app.llm.client import LLMInvalidOutputError, parse_structured, response_schema

    sent = json.dumps(response_schema(ProseDraft))
    assert "$ref" not in sent and "$defs" not in sent and "minLength" not in sent and "maxItems" not in sent
    assert set(response_schema(ProseDraft)["properties"]) == set(ProseDraft.model_fields)
    assert response_schema(ProseDraft)["properties"]["model_evidence"]["items"]["required"] == ["text", "evidence_ids"]

    class Named(BaseModel):  # a property called like a dropped keyword is kept
        title: str = Field(max_length=5)
        pattern: list[str] = Field(min_length=1)

    assert set(response_schema(Named)["properties"]) == {"title", "pattern"}
    with pytest.raises(LLMInvalidOutputError):  # the limits Gemini isn't sent still apply to its reply
        parse_structured(json.dumps({"positive_evidence": [{"text": "x", "evidence_ids": []}], "potential_gaps": [],
                                     "model_evidence": [], "how_to_improve": []}), ProseDraft)


def test_a_failed_first_attempt_then_a_good_draft_is_validated_exactly_once():
    """Seen live (Phase 12 trace): attempt 1 returned no draft (invalid JSON), attempt 2 a good one. That draft
    used to be validated twice."""
    from app.llm import LLMInvalidOutputError

    out, result = run(llm=FakeLLM(LLMInvalidOutputError("response failed ProseDraft validation"), GOOD),
                      model=KnownModel())
    assert out.status == "success", out.errors
    g = result.match_explanation.generation
    assert result.match_explanation.source == "llm_validated" and g["llm_attempts"] == 2
    assert [(v["attempt"], v["valid"]) for v in g["validations"]] == [(2, True)]
    assert [c.tool_name for c in out.tool_calls].count("validate_explanation") == 1
