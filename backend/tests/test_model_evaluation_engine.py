"""Phase 6 Model Evaluation Engine: weights config, normalisation, sensitivity, best-model lines and the report.

Heavy parts (model downloads, cost measurement, encoding) are replaced by fakes; the real runs are in
results/embedding_evaluation/.
"""

import json

import pandas as pd
import pytest
import yaml

import src.embeddings.model_evaluation_engine as engine
from src.embeddings.embedding_model_benchmark import load_benchmark_data
from src.evaluation import NOT_YET_EVALUATED
from src.evaluation.relevance_labels import STAMPS
from src.evaluation.splits import split_for

MODELS = ("tfidf", "sbert", "e5", "bge", "minilm")
# (latency ms, peak MB, size MB, bytes/vector, throughput) per model, roughly the measured shape
COST = {"tfidf": (2, 97, 0.1, 532, 3700), "sbert": (600, 900, 417, 3072, 5), "e5": (620, 910, 418, 3072, 5),
        "bge": (610, 905, 418, 3072, 5), "minilm": (120, 210, 87, 1536, 24)}


def fake_costs():
    models = {}
    for key, (lat, mem, size, bpv, tput) in COST.items():
        models[key] = {
            "version": f"{key}-v", "revision": f"{key}-rev", "dimension": bpv // 4, "size_mb": size, "load_seconds": 1.0,
            "memory_mb": {"after_load": mem / 2, "after_encoding": mem, "peak": mem},
            "cpu": {"single_pair_latency_ms": {"median": lat, "p95": lat * 1.2, "stdev": 1.0, "trials": 30},
                    "batch_latency_ms": {"median": 64000 / tput, "batch_size": 64, "stdev": 1.0, "trials": 7},
                    "throughput_texts_per_s": tput},
            "cpu_threads": 4, "deterministic_cpu_max_abs_diff": 0.0, "offline": True, "bytes_per_vector": bpv,
            "truncation": {"max_seq_length": None if key == "tfidf" else 512, "students_truncated_pct": 0.0,
                           "jobs_truncated_pct": 0.0},
            "mps": "Not applicable (CPU-only model)" if key == "tfidf" else {"throughput_texts_per_s": tput * 10,
                                                                               "cpu_vs_mps_max_abs_diff": 1e-7},
        }
        if key == "tfidf":
            models[key]["mean_nonzeros_per_job_vector"] = 66.5
    return {"measured_at": "2026-10-09T00:00:00+00:00", "models": models,
            "hardware": {"machine": "arm64", "system": "test", "cpu_cores": 8, "ram_gb": 16}}


@pytest.fixture
def labelled(tmp_path):
    """Synthetic labels for 4 validation students x 6 real jobs, and scores where BGE agrees with them best,
    TF-IDF somewhat, and MiniLM is inverted (below the floor)."""
    students, jobs, _ = load_benchmark_data()
    val = [s for s in students["student_id"] if split_for(s) == "validation"][:4]
    job_ids = list(jobs["job_id"][:6])
    grades = (3, 2, 1, 0, 0, 0)
    rows = [{"student_id": s, "job_id": j, "relevance_label": g, "label_source": "synthetic_clearly_marked",
             "split": "validation"} for s in val for j, g in zip(job_ids, grades)]
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    (tmp_path / "synthetic.manifest.json").write_text(json.dumps(
        {"label_source": "synthetic_clearly_marked", "generated_by": "test fixture", "models_used_for_labels": []}))
    quality = {"bge": (0.9, 0.8, 0.7, 0.1, 0.2, 0.3), "e5": (0.8, 0.9, 0.5, 0.6, 0.1, 0.2),
               "sbert": (0.7, 0.3, 0.8, 0.2, 0.4, 0.1), "tfidf": (0.5, 0.1, 0.6, 0.4, 0.3, 0.2),
               "minilm": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)}
    scores = {p: {m: {(s, j): v for s in val for j, v in zip(job_ids, quality[m])} for m in MODELS}
              for p in ("full_text", "section_weighted")}
    ranges = {m: {"mean": 0.5, "sd": 0.1, "p5": 0.3, "p95": 0.7} for m in MODELS}
    return str(path), (scores, {m: f"{m}-v" for m in MODELS}, ranges)


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(engine, "measure_costs", lambda reuse: fake_costs())

    def install(score_result):
        monkeypatch.setattr(engine, "score_all", lambda device, protocols, extra_job_ids=(): score_result)
    return install


def test_the_shipped_weights_config_is_valid_and_every_weight_has_a_reason():
    cfg = engine.load_config()
    assert set(cfg["scenarios"]) == {"balanced", "quality_prioritized", "low_resource_deployment"}
    assert cfg["evaluation"]["selection_split"] == "validation" and cfg["evaluation"]["ks"] == [5, 10]


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c["scenarios"]["balanced"]["weights"]["mrr"].update(weight=0.5), "sum to"),
    (lambda c: c["scenarios"]["balanced"]["weights"]["mrr"].update(reason=""), "one-line reason"),
    (lambda c: c["scenarios"]["balanced"]["weights"].pop("size"), "must weight exactly"),
    (lambda c: c["evaluation"].update(selection_split="train"), "train"),
])
def test_a_broken_weights_config_is_refused(tmp_path, mutate, message):
    with open(engine.WEIGHTS, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    mutate(cfg)
    path = tmp_path / "weights.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=message):
        engine.load_config(str(path))


def test_normalisation_respects_direction_and_log_scale():
    criteria = {"a": {"ndcg": 0.8, "latency": 1.0}, "b": {"ndcg": 0.4, "latency": 10.0}, "c": {"ndcg": 0.6, "latency": 100.0}}
    n = engine.normalise(criteria, "min_max", log_scale=["latency"])
    assert n["ndcg"] == pytest.approx({"a": 1.0, "b": 0.0, "c": 0.5})
    assert n["latency"] == pytest.approx({"a": 1.0, "b": 0.5, "c": 0.0})  # log10: 0, 1, 2 -> lower is better
    assert "ndcg" not in engine.normalise({"a": {"ndcg": None}, "b": {"ndcg": 0.5}})  # missing -> not comparable


def test_overall_score_needs_label_quality_and_the_winner_must_be_eligible():
    normalised = {"ndcg": {"a": 1.0, "b": 0.0}, "latency": {"a": 0.0, "b": 1.0}}
    assert engine.overall_scores({"latency": normalised["latency"]}, {"ndcg": 0.5, "latency": 0.5}) is None
    scores = engine.overall_scores(normalised, {"ndcg": 0.7, "latency": 0.3})
    assert engine.pick_winner(scores, ["a", "b"]) == "a" and engine.pick_winner(scores, ["b"]) == "b"
    stability = engine.random_weight_stability(normalised, {"ndcg": 0.7, "latency": 0.3}, ["a", "b"], 200, 20, 1)
    assert sum(stability.values()) == pytest.approx(100.0, abs=0.2)


def test_without_labels_quality_and_overall_read_not_yet_evaluated(fakes, labelled):
    fakes(labelled[1])
    report = engine.run(None, "cpu", reuse_cost=True)
    assert report["labels"]["provided"] is False
    assert all(report["criteria"][m]["ndcg"] is None for m in MODELS)
    assert all(s["overall"] is None for s in report["scenarios"].values())
    assert report["best_models"]["quality"].startswith(NOT_YET_EVALUATED)
    assert report["best_models"]["overall"].startswith(NOT_YET_EVALUATED)
    assert report["best_models"]["speed"].startswith("D. Small")  # measured cost still decides speed
    md = report["markdown"]
    assert "| Model | MRR | NDCG@K | Recall@K | Precision@K | Latency | Memory | Size | Dimension | Complexity | Scalability | Overall Score |" in md
    assert "| E. Custom / domain-adapted | Not feasible — insufficient labeled data |" in md
    assert "not comparable across model families" in md
    assert md.count(NOT_YET_EVALUATED) >= 20


def test_synthetic_labels_are_stamped_and_the_full_decision_is_made(fakes, labelled):
    path, score_result = labelled
    fakes(score_result)
    report = engine.run(path, "cpu", reuse_cost=True)
    stamp = STAMPS["synthetic_clearly_marked"]
    md = report["markdown"]
    assert md.count(stamp) >= 3 and md.splitlines()[2] == f"> **{stamp}**"
    assert report["validity"] == "evaluation validity: LOW — synthetic labels, not real ground truth"
    assert report["labels"]["train_rows_evaluated"] == 0 and report["labels"]["used_split"] == "validation"
    assert "minilm" not in report["eligible"]  # inverted scores: below the TF-IDF floor
    assert report["scenarios"]["quality_prioritized"]["winner"] == "bge"
    assert report["scenarios"]["balanced"]["overall"] is not None
    assert all(line and NOT_YET_EVALUATED not in line for line in report["best_models"].values())
    assert "(below floor)" in md and "Does the winner change across scenarios?" in md


def test_unknown_split_or_source_selection_finds_nothing_to_evaluate(fakes, labelled, tmp_path):
    path, score_result = labelled
    fakes(score_result)
    with pytest.raises(ValueError, match="no human_reviewed labels"):
        engine.run(path, "cpu", reuse_cost=True, label_source="human_reviewed")


@pytest.mark.skipif(not __import__("os").environ.get("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to run against Postgres")
def test_an_experiment_is_recorded_with_the_spec_fields_and_served_by_the_api(fakes, labelled):
    import os

    import psycopg
    from fastapi.testclient import TestClient

    from app import experiments
    from app.db.migrate import apply_migrations
    from app.main import app

    dsn = os.environ["TEST_DATABASE_URL"]
    apply_migrations(dsn)
    path, score_result = labelled
    fakes(score_result)
    report = engine.run(path, "cpu", reuse_cost=True)
    report["experiment_id"] += "-pytest"
    engine.record_experiment(dsn, report)
    original = experiments._dsn
    experiments.configure(dsn)
    try:
        client = TestClient(app)
        listed = client.get("/experiments", params={"kind": "embedding_selection"}).json()
        assert report["experiment_id"] in {e["experiment_id"] for e in listed}
        full = client.get(f"/experiments/{report['experiment_id']}").json()
        for field in ("dataset_version", "dataset_size", "train_split", "validation_split", "test_split", "random_seed",
                      "feature_version", "embedding_model", "embedding_version", "prediction_model", "hyperparameters",
                      "metrics", "runtime_seconds", "memory", "hardware", "notes", "limitations"):
            assert full[field] not in (None, "", {}), field
        assert full["validation_split"]["evaluated"] is True and full["train_split"]["evaluated"] is False
        statuses = {(m["model_name"], m["metric"]): m["status"] for m in full["metric_rows"]}
        assert statuses[("custom", "overall_score_balanced")] == "not_feasible"
        assert statuses[("bge", "scalability")] == "extrapolated" and statuses[("bge", "ndcg@5")] == "measured"
        assert client.get("/experiments/emb-unknown").status_code == 404
    finally:
        experiments.configure(original)
        with psycopg.connect(dsn) as conn:
            conn.execute("DELETE FROM experiment_runs WHERE experiment_id = %s", (report["experiment_id"],))


def _report(source, level, winner, floor_overall, winner_overall, distinguishable):
    return {"labels": {"provided": source is not None, "used_source": source,
                       "validity": {"level": level, "reasons": ["x"]}},
            "scenarios": {"balanced": {"winner": winner, "overall": {"tfidf": floor_overall, winner: winner_overall}
                                       if winner else None}},
            "confidence": {"vs_floor": {winner: {"distinguishable": distinguishable}} if winner else {}}}


@pytest.mark.parametrize("args, decision", [
    ((None, None, None, 0, 0, False), "no_selection"),                         # no labels
    (("weakly_labelled", "LIMITED", "bge", 0.5, 0.6, True), "no_selection"),   # weak labels never select
    (("human_reviewed", "LOW", "bge", 0.5, 0.6, True), "no_selection"),        # circularity check failed
    (("human_reviewed", "OK", None, 0.8, 0, False), "keep_floor"),             # nothing beat the floor
    (("human_reviewed", "OK", "bge", 0.8, 0.6, False), "keep_floor"),          # floor cheaper, gain not shown
    (("human_reviewed", "OK", "bge", 0.8, 0.6, True), "select"),               # real gain: worth the cost
    (("human_reviewed", "OK", "bge", 0.5, 0.6, False), "select"),              # wins the trade-off outright
])
def test_the_adr_0012_decision_rule(args, decision):
    assert engine.recommend(_report(*args))["decision"] == decision
