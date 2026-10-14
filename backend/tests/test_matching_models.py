"""Phase 8: matching-model candidates, the selection engine's rules, and the config-flag wiring (docs/adr/0015)."""

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agents import build_agents
from app.core.config import REPO_ROOT, Settings
from app.matching.config import load_matching_config
from app.matching.features import FEATURE_NAMES, PROTECTED_ATTRIBUTES, ProtectedFeatureError
from app.matching.predictor import PredictorUnavailable, TrainedPredictor, active_choice, build_predictor
from app.schemas.agent_io import MatchingAgentOutput
from app.schemas.agent_state import AgentState
from src.scoring import matching_model_selection_engine as E
from src.scoring import matching_models as M
from tests.matching_world import build_world

CANDIDATES = ("logistic_regression", "decision_tree", "random_forest", "gradient_boosting", "xgboost", "lightgbm")


def synthetic(n=600, seed=0) -> pd.DataFrame:
    """Feature rows where 'suitable' depends on skill match and interest alignment, with missing values."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.random((n, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    frame.loc[rng.random(n) < 0.3, "certification_relevance"] = np.nan
    frame.loc[rng.random(n) < 0.1, "preferred_skill_match_score"] = np.nan
    signal = frame["skill_match_score"] + frame["career_interest_alignment"] + rng.normal(0, 0.2, n)
    frame["relevance_label"] = np.where(signal > 1.3, 3, np.where(signal > 0.9, 1, 0))
    return frame


@pytest.fixture(scope="module")
def cands():
    return M.load_candidates()


@pytest.fixture(scope="module")
def trained(cands, tmp_path_factory):
    """Every candidate trained on the same synthetic rows, saved and reloaded through the production path."""
    folder = tmp_path_factory.mktemp("models")
    data = synthetic()
    X, y = M.feature_matrix(data), M.target(data["relevance_label"])
    out = {}
    for key in CANDIDATES:
        spec = cands["candidates"][key]
        model = M.fit(key, spec, X, y, seed=42, threads=1)
        path, fmt = M.save(model, key, spec, folder)
        out[key] = (model, path, fmt, M.sha256(path))
    return folder, data, out


# -- the feature-exclusion test, against the Phase 8 training data ----------------------------------------------
def test_training_data_refuses_protected_attributes():
    data = synthetic(20)
    for col in ("gender", "religion", "name", "email", "phone", "race"):
        with pytest.raises(ValueError, match="protected"):
            M.feature_matrix(data.assign(**{col: "x"}))
    assert list(M.feature_matrix(data).columns) == list(FEATURE_NAMES)


def test_the_committed_feature_matrix_holds_only_ids_and_contract_features():
    files = sorted((REPO_ROOT / "data" / "benchmark" / "matching_features").glob("*.csv.gz"))
    if not files:
        pytest.skip("no feature matrix built yet (python -m app.matching.training_data)")
    for path in files:
        columns = list(pd.read_csv(path, nrows=1).columns)
        assert columns == ["student_id", "job_id", "split", "retrieval_scored", *FEATURE_NAMES]
        assert not {c.lower() for c in columns} & PROTECTED_ATTRIBUTES
        assert not any("gender" in c.lower() for c in columns)
        M.feature_matrix(pd.read_csv(path, nrows=50))


def test_a_protected_feature_name_can_not_enter_the_contract():
    from app.matching.features import validate_feature_names

    with pytest.raises(ProtectedFeatureError):
        validate_feature_names([*FEATURE_NAMES, "gender"])


# -- candidates --------------------------------------------------------------------------------------------------
def test_all_six_candidates_are_configured_with_reasons(cands):
    assert tuple(cands["candidates"]) == CANDIDATES
    assert cands["training"]["target"] == "relevance_label >= 2"
    for key, spec in cands["candidates"].items():
        assert set(spec["reasons"]) >= set(spec["params"]), key


def test_every_candidate_round_trips_through_its_saved_artifact(trained):
    _, data, out = trained
    X = M.feature_matrix(data)
    for key, (model, path, fmt, digest) in out.items():
        loaded = M.load(path, key, fmt, digest)
        assert np.allclose(loaded.score(X), model.predict_proba(X)[:, 1], atol=1e-6), key
        assert loaded.score(X).min() >= 0 and loaded.score(X).max() <= 1


def test_a_tampered_artifact_is_refused_before_it_is_deserialised(trained):
    _, _, out = trained
    _, path, fmt, digest = out["logistic_regression"]
    with pytest.raises(ValueError, match="does not match the manifest"):
        M.load(path, "logistic_regression", fmt, "0" * 64)


def test_every_candidate_learns_the_signal(trained):
    from sklearn.metrics import roc_auc_score

    _, _, out = trained
    holdout = synthetic(400, seed=1)
    X, y = M.feature_matrix(holdout), M.target(holdout["relevance_label"])
    for key, (_, path, fmt, digest) in out.items():
        assert roc_auc_score(y, M.load(path, key, fmt, digest).score(X)) > 0.85, key


def test_explanations_add_up_to_each_models_output_per_model(trained):
    _, data, out = trained
    X = M.feature_matrix(data)
    for key, (_, path, fmt, digest) in out.items():
        loaded = M.load(path, key, fmt, digest)
        explainer = M.Explainer(loaded, X.iloc[:100])
        values, base = explainer.explain(X.iloc[:50])
        assert values.shape == (50, len(FEATURE_NAMES)), key
        assert np.max(np.abs(values.sum(axis=1) + base - loaded.margin(X.iloc[:50]))) < 1e-4, key
        assert ("LinearExplainer" if key == "logistic_regression" else "TreeExplainer") in explainer.name, key


def test_refits_with_the_same_seed_are_identical(cands):
    data = synthetic()
    X, y = M.feature_matrix(data), M.target(data["relevance_label"])
    for key in CANDIDATES:
        a = M.fit(key, cands["candidates"][key], X, y, 42, 1).predict_proba(X)[:, 1]
        b = M.fit(key, cands["candidates"][key], X, y, 42, 1).predict_proba(X)[:, 1]
        assert np.array_equal(a, b), key


def test_complexity_points_follow_the_rubric(cands):
    lr, _ = M.complexity_points(cands["candidates"]["logistic_regression"])
    xgb, reasons = M.complexity_points(cands["candidates"]["xgboost"])
    assert lr == 2 + 2  # imputer + scaler, 3 hyperparameters
    assert xgb == 2 + 2 + 1 and any("xgboost" in r for r in reasons)  # dependency, 4 params, own format


# -- the engine's rules -------------------------------------------------------------------------------------------
def test_the_shipped_weights_are_valid():
    cfg = E.load_weights()
    assert set(cfg["scenarios"]) == {"balanced", "quality_prioritized", "low_resource_deployment"}
    assert cfg["gate"]["must_beat_baseline"] and cfg["evaluation"]["selection_split"] == "validation"


def criteria(**overrides):
    base = {"f1": 0.5, "ndcg": 0.5, "mrr": 0.5, "latency": 1.0, "training_time": 1.0, "memory": 10.0,
            "complexity": 3, "explainability": 3}
    return {m: {**base, **overrides.get(m, {})} for m in ("logistic_regression", "xgboost", "weighted_baseline_v1")}


def test_normalisation_aligns_directions_and_handles_zero_costs():
    c = criteria(xgboost={"ndcg": 0.6, "latency": 10.0}, weighted_baseline_v1={"training_time": 0.0})
    n = E.normalise(c, "min_max", ["latency", "training_time", "memory"])
    assert n["ndcg"]["xgboost"] == 1.0 and n["ndcg"]["logistic_regression"] == 0.0
    assert n["latency"]["xgboost"] == 0.0 and n["latency"]["logistic_regression"] == 1.0
    assert n["training_time"]["weighted_baseline_v1"] == 1.0


def test_without_quality_there_is_no_overall_score():
    c = criteria()
    for m in c:
        c[m]["ndcg"] = None
    n = E.normalise(c, "min_max", [])
    assert "ndcg" not in n and E.overall_scores(n, {"ndcg": 0.5, "latency": 0.5}) is None


def test_xgboost_is_never_preferred_by_name():
    """A simpler model that measures better wins; nothing in the engine favours XGBoost."""
    c = criteria(logistic_regression={"ndcg": 0.62, "f1": 0.6, "latency": 0.5, "complexity": 3, "explainability": 5},
                 xgboost={"ndcg": 0.60, "f1": 0.6, "latency": 2.0, "complexity": 6, "explainability": 3})
    cfg = E.load_weights()
    n = E.normalise(c, "min_max", cfg["normalization"]["log_scale"])
    for sc in cfg["scenarios"].values():
        overall = E.overall_scores(n, {k: v["weight"] for k, v in sc["weights"].items()})
        assert E.pick_winner(overall, ["logistic_regression", "xgboost"]) == "logistic_regression"


def report(source, level="OK", win="logistic_regression", overall=None, distinguishable=True):
    return {"labels": {"provided": source is not None, "used_source": source,
                       "validity": {"level": level, "reasons": []}},
            "scenarios": {"balanced": {"winner": win, "overall": overall or {win: 0.8, "weighted_baseline_v1": 0.5}}},
            "confidence": {"vs_baseline": {win: {"distinguishable": distinguishable}}},
            "config": {"evaluation": {"primary_k": 10}},
            "models": {win: {"display": win}, "weighted_baseline_v1": {"display": "Baseline 3"}}}


def test_only_human_labels_with_ok_validity_can_select():
    assert E.recommend(report(None))["decision"] == "no_selection"
    assert E.recommend(report("weakly_labelled", "LIMITED"))["decision"] == "no_selection"
    assert E.recommend(report("human_reviewed", "LOW"))["decision"] == "no_selection"
    assert E.recommend(report("human_reviewed"))["model"] == "logistic_regression"


def test_baseline_is_kept_when_the_winner_is_not_better_and_costs_more():
    r = report("human_reviewed", overall={"xgboost": 0.4, "weighted_baseline_v1": 0.6}, win="xgboost",
               distinguishable=False)
    assert E.recommend(r)["decision"] == "keep_baseline"


def test_activation_needs_a_select_decision(tmp_path, monkeypatch):
    path = tmp_path / "matching_model.yaml"
    path.write_text((REPO_ROOT / "config" / "matching_model.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(E, "ACTIVE_CONFIG", path)
    with pytest.raises(SystemExit, match="refused"):
        E.activate({"recommendation": {"decision": "no_selection"}})
    E.activate({"recommendation": {"decision": "select", "model": "decision_tree"},
                "artifacts_experiment": "mms-a", "experiment_id": "mms-b"})
    choice = active_choice(path=path)
    assert (choice.active, choice.artifacts_experiment, choice.selection_experiment) == ("decision_tree", "mms-a", "mms-b")
    assert not E.register_artifacts("mms-c")  # a trained model is active: its artifacts are not swapped


# -- wiring: the config flag, metadata, refusals ------------------------------------------------------------------
def test_the_shipped_flag_is_the_human_label_selection():
    # docs/adr/0015 and 0012: one human-label selection per embedding model. The TF-IDF fallback runs the model
    # selected on TF-IDF features.
    primary = active_choice()
    assert primary.embedding_model == "e5_base_v2"
    assert primary.active == "lightgbm" and primary.selection_experiment == "mms-20261014T151058Z-human"
    assert isinstance(build_predictor(load_matching_config(), embedding_model="e5_base_v2"), TrainedPredictor)
    fb = active_choice(embedding_model="tfidf_baseline")
    assert fb.active == "logistic_regression" and fb.selection_experiment == "mms-20261014T132323Z-human"
    assert isinstance(build_predictor(load_matching_config(), embedding_model="tfidf_baseline"), TrainedPredictor)


def manifest_folder(tmp_path, trained, key="decision_tree", embedding_version=None, feature_version="features-v1"):
    folder, _, out = trained
    _, path, fmt, digest = out[key]
    target = tmp_path / "mms-test"
    target.mkdir(exist_ok=True)
    (target / path.name).write_bytes(path.read_bytes())
    (target / "manifest.json").write_text(json.dumps({
        "experiment_id": "mms-test", "feature_version": feature_version, "embedding_version": embedding_version,
        "models": {key: {"file": path.name, "format": fmt, "sha256": digest}}}), encoding="utf-8")
    flag = tmp_path / "matching_model.yaml"
    flag.write_text("active: weighted_baseline_v1\nartifacts_experiment: mms-test\nselection_experiment: none\n",
                    encoding="utf-8")
    return flag, tmp_path


def test_the_flag_or_override_loads_a_trained_model_with_its_own_version(tmp_path, trained, embedder):
    version, _ = embedder.describe()
    flag, models_dir = manifest_folder(tmp_path, trained, embedding_version=version)
    p = build_predictor(load_matching_config(), "decision_tree", active_path=flag, models_dir=models_dir)
    assert isinstance(p, TrainedPredictor) and p.name == "decision_tree"
    assert p.version == f"decision_tree+{trained[2]['decision_tree'][3][:12]}" and p.experiment_id == "mms-test"
    p.check(version)
    with pytest.raises(PredictorUnavailable, match="not among the trained models"):
        build_predictor(load_matching_config(), "xgboost", active_path=flag, models_dir=models_dir)


def test_a_trained_model_refuses_other_embedding_or_feature_versions(tmp_path, trained):
    flag, models_dir = manifest_folder(tmp_path, trained, embedding_version="tfidf-other")
    p = build_predictor(load_matching_config(), "decision_tree", active_path=flag, models_dir=models_dir)
    with pytest.raises(PredictorUnavailable, match="embeddings are"):
        p.check("tfidf-current")


def test_the_agent_returns_trained_model_metadata_end_to_end(tmp_path, trained, embedder):
    version, _ = embedder.describe()
    flag, models_dir = manifest_folder(tmp_path, trained, embedding_version=version)
    repo, cfg, cal = build_world(embedder)
    predictor = build_predictor(cfg, "decision_tree", active_path=flag, models_dir=models_dir)
    settings = Settings(_env_file=None, environment="test", tool_retry_backoff_seconds=0.0)
    agents = build_agents(settings=settings, profile_repo=None, job_repo=None, embedder=embedder, matching_repo=repo,
                          matching_overrides={"cfg": cfg, "calibration_loader": lambda *_: cal, "predictor": predictor})
    state = AgentState(request_id="req-p8", task_type="recommend_jobs_for_student")
    out = asyncio.run(agents["matching_agent"].run(state, {"mode": "jobs_for_student", "student_id": "S1", "top_k": 3}))
    assert out.status == "success", out.errors
    data = MatchingAgentOutput.model_validate(out.data)
    for m in data.matches:
        v = m.model_versions
        assert v.matching_model == "decision_tree" and v.matching_model_version == predictor.version
        assert v.experiment_id == "mms-test" and v.feature_version == "features-v1"
        assert 0 <= m.suitability_score <= 1 and m.employability.overall == round(m.suitability_score * 100, 1)
        assert "trained Phase 8 model" in m.employability.method
    assert data.notes[0].startswith("decision_tree: trained Phase 8 model")


def test_each_embedding_model_gets_its_own_matching_model(tmp_path):
    # docs/adr/0012: a trained model refuses other embeddings, so the TF-IDF fallback has its own entry
    path = tmp_path / "matching_model.yaml"
    path.write_text("embedding_model: e5_base_v2\nactive: lightgbm\nartifacts_experiment: mms-e5\n"
                    "selection_experiment: sel-e5\nfallback:\n  tfidf_baseline:\n    active: logistic_regression\n"
                    "    artifacts_experiment: mms-tfidf\n    selection_experiment: sel-tfidf\n", encoding="utf-8")
    primary = active_choice(path=path, embedding_model="e5_base_v2")
    fallback = active_choice(path=path, embedding_model="tfidf_baseline")
    assert (primary.active, primary.artifacts_experiment) == ("lightgbm", "mms-e5")
    assert (fallback.active, fallback.artifacts_experiment, fallback.selection_experiment) == \
        ("logistic_regression", "mms-tfidf", "sel-tfidf")
    with pytest.raises(PredictorUnavailable, match="no matching model is configured"):
        active_choice(path=path, embedding_model="bge")
