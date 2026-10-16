"""Phase 14: the evaluation framework's own logic (docs/adr/0021).

The real runs (E5, the trained model, the benchmark databases) happen in the backend image and are stored in
results/; these tests pin the parts whose mistakes would silently change a reported number: the keyword scorer, the
verdict rule, the scale-up generator's flags and determinism, the benchmark's database guard, the leakage
helpers, the RQ9 route audit and the Phase 6 recorder fix.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from app.matching.features import FEATURE_NAMES
from src.evaluation import baseline_comparison as bc
from src.evaluation import performance_benchmark as pb
from src.evaluation import trace_route_audit as tra
from src.pipeline import leakage_report as lr
from src.pipeline import synthetic_scaleup as syn

ROOT = Path(__file__).resolve().parents[2]
STUDENTS = ROOT / "data" / "processed" / "students_unified.csv"
JOBS = ROOT / "data" / "processed" / "jobs.csv"


# ---------------------------------------------------------------- Part A: baselines

def test_keyword_score_counts_exact_and_whole_word_matches_of_the_jobs_skills():
    student = frozenset({"python", "sql", "power bi", "c"})
    assert bc.keyword_score(student, frozenset({"python", "sql"})) == 1.0
    # whole words, either direction: "sql" inside "advanced sql", "power bi dashboards" contains "power bi"
    assert bc.keyword_score(student, frozenset({"advanced sql", "power bi dashboards", "java"})) == pytest.approx(2 / 3)
    # a one-letter skill matches only exactly: "c" must not hit "c#" or "objective c"
    assert bc.keyword_score(frozenset({"c"}), frozenset({"c#", "objective c"})) == 0.0
    assert bc.keyword_score(frozenset({"c"}), frozenset({"c"})) == 1.0
    # no word-fragment matches: "java" is not in "javascript"
    assert bc.keyword_score(frozenset({"java"}), frozenset({"javascript"})) == 0.0


def test_a_job_without_skills_cannot_be_keyword_scored():
    assert bc.keyword_score(frozenset({"python"}), frozenset()) is None


def test_the_verdict_needs_a_confidence_interval_that_excludes_zero():
    ev = {"bootstrap_resamples": 500, "seed": 42}
    students = [f"S{i}" for i in range(15)]

    def quality(a_values, b_values):
        return {"a": {"per_student": {s: {"ndcg@5": v} for s, v in zip(students, a_values)}},
                "b": {"per_student": {s: {"ndcg@5": v} for s, v in zip(students, b_values)}}}

    clear = bc.compare(quality([0.9] * 15, [0.5] * 15), "a", "b", "ndcg@5", ev)
    assert clear["distinguishable"] and clear["verdict"] == "a outperforms b"
    noisy = bc.compare(quality([0.9, 0.1] * 7 + [0.6], [0.5] * 15), "a", "b", "ndcg@5", ev)
    assert not noisy["distinguishable"] and "not statistically distinguishable" in noisy["verdict"]


def test_the_comparison_config_is_fixed_and_never_evaluates_train():
    cfg = bc.load_config()
    assert cfg["answer_split"] == "test" and "train" not in cfg["splits"]
    assert set(cfg["baselines"]) == set(bc.ORDER)
    assert cfg["comparisons"]["RQ3"] == [["b5_hybrid", "b4_semantic"]]
    assert all(a in bc.ORDER and b in bc.ORDER for pairs in cfg["comparisons"].values() for a, b in pairs)


def test_the_stored_comparison_reproduces_phase_6_and_8_on_validation():
    stored = ROOT / "results" / "baseline_comparison.json"
    if not stored.exists():
        pytest.skip("run python -m src.evaluation.baseline_comparison in the backend image first")
    report = json.loads(stored.read_text(encoding="utf-8"))
    assert report["reproduction"] and all(r["match"] for r in report["reproduction"])
    test = report["quality"]["test"]
    assert {k: v["pairs"] for k, v in test.items()} == {k: 180 for k in bc.ORDER}  # the identical pairs for all five


# ---------------------------------------------------------------- Part D: scale-up data and the benchmark guard

def test_scale_up_keeps_the_real_rows_and_flags_every_synthetic_one():
    real = pd.read_csv(STUDENTS, dtype=str, keep_default_na=False)
    big = syn.scale_students(real, 1000)
    assert len(big) == 1000 and big["student_id"].is_unique
    pd.testing.assert_frame_equal(big.iloc[: len(real)].reset_index(drop=True), real.astype(str))
    added = big.iloc[len(real):]
    assert (added["is_synthetic"] == "True").all() and (added["synthetic_fields"] != "").all()
    assert not set(added["student_id"]) & set(real["student_id"])
    # internally consistent: every recombined skill list comes from a real student of the same source and branch
    real_skills = real.groupby(["source_dataset", "branch"])["skills_raw"].agg(set).to_dict()
    for row in added.itertuples(index=False):
        pool = real_skills.get((row.source_dataset, row.branch))
        if pool is not None and len(pool) > 1:
            assert row.skills_raw in pool


def test_scale_up_is_seeded_and_a_larger_size_extends_a_smaller_one():
    real = pd.read_csv(STUDENTS, dtype=str, keep_default_na=False)
    small, again, large = syn.scale_students(real, 600), syn.scale_students(real, 600), syn.scale_students(real, 900)
    pd.testing.assert_frame_equal(small, again)
    pd.testing.assert_frame_equal(large.iloc[:600], small)


def test_job_sample_is_real_seeded_and_unique():
    jobs = pd.read_csv(JOBS, dtype=str, keep_default_na=False)
    a, b = syn.sample_jobs(jobs, 1000), syn.sample_jobs(jobs, 1000)
    assert len(a) == 1000 and a["job_id"].is_unique and set(a["job_id"]) <= set(jobs["job_id"])
    pd.testing.assert_frame_equal(a, b)


def test_the_benchmark_refuses_to_recreate_a_database_it_does_not_own():
    with pytest.raises(ValueError, match="refusing"):
        pb.create_database("postgresql://u:p@127.0.0.1:1/x", "employability")


def test_stage_latencies_are_per_entity_sums_of_the_implementing_tools():
    def ingest(tools):
        return {"per_tool_ms": tools}

    fake = {"ingest_students": ingest({"extract_skills": [1.0, 2.0], "normalize_skills": [10.0, 20.0],
                                       "parse_student_sections": [5.0, 5.0], "generate_embedding": [100.0, 300.0]}),
            "ingest_jobs": ingest({}),
            **{f"{k}_requests": {"per_tool_ms": {}, "per_agent_ms": {}, "requests": []}
               for k in ("student", "company", "upload")}}
    table = pb.stage_table(fake)
    assert table["skill_extraction_student"]["median"] == pytest.approx(16.5)  # (11 + 22) / 2
    assert table["embedding_student"]["n"] == 2 and table["jd_parsing"]["n"] == 0


# ---------------------------------------------------------------- Part C: leakage helpers

def test_every_contract_feature_is_reviewed_for_leakage():
    assert set(lr.FEATURE_INPUTS) == set(FEATURE_NAMES)
    flagged = {f for f, (_, shared) in lr.FEATURE_INPUTS.items() if shared}
    assert {"branch_match", "career_interest_alignment"} <= flagged


def test_near_duplicate_texts_are_grouped_transitively():
    texts = {"a": "python sql machine learning data analysis dashboards", "b": "python sql machine learning data analysis dashboards",
             "c": "autocad solidworks thermodynamics gearbox design", "d": ""}
    pairs = lr._near_pairs(texts)
    assert [(a, b) for a, b, _ in pairs] == [("a", "b")]
    assert lr._components(pairs) == [{"a", "b"}]


# ---------------------------------------------------------------- RQ9: route audit

def _t(minute=0):
    return datetime(2026, 10, 15, 12, minute, tzinfo=timezone.utc)


def test_route_audit_accepts_skips_and_flags_an_agent_outside_the_route():
    tops = [{"request_id": "r1", "task_type": "cohort_benchmark", "status": "success", "routing_source": "deterministic_fallback",
             "agents_called": ["profile_agent"], "duration_ms": 10.0, "started_at": _t()},
            {"request_id": "r2", "task_type": "learning_recommendations", "status": "success",
             "routing_source": "deterministic_fallback", "agents_called": ["profile_agent", "matching_agent"],
             "duration_ms": 20.0, "started_at": _t(1)},
            {"request_id": "r3", "task_type": "cohort_benchmark", "status": "success", "routing_source": "llm",
             "agents_called": ["profile_agent", "matching_agent"], "duration_ms": 30.0, "started_at": _t(2)}]
    kid = lambda parent, name: {"parent": parent, "agent_name": name, "status": "success", "duration_ms": 1.0,  # noqa: E731
                                "attempt": 1, "tool_call_count": 1}
    kids = [kid("r1", "profile_agent"), kid("r2", "profile_agent"), kid("r2", "matching_agent"),
            kid("r3", "profile_agent"), kid("r3", "matching_agent")]
    msgs = [{"run": "r2", "req": "r2", "message_type": "skip", "recipient": "learning_agent",
             "content": {"reason": "not needed: no skill gaps were identified"}, "duration_ms": 0.0}]
    out = tra.audit(tops, kids, msgs)
    assert out["tasks"]["learning_recommendations"]["route_accounted"] == 1  # the skip completes the route
    cohort = out["tasks"]["cohort_benchmark"]
    assert cohort["route_accounted"] == 1 and cohort["requests_running_an_agent_outside_the_route"] == 1
    assert cohort["agents_not_needed_that_ran"] == ["matching_agent"]


# ---------------------------------------------------------------- Phase 13 finding: the Phase 6 recorder

def test_recorder_rows_follow_the_report_not_this_processes_model_list():
    import src.embeddings.model_evaluation_engine as engine

    report = json.loads((ROOT / "results" / "embedding_evaluation" / "emb-20261014T145548Z-human.json")
                        .read_text(encoding="utf-8"))
    assert "e5_domain" in report["models"] and "e5_domain" not in engine.RUNNABLE  # PHASE6_DOMAIN_MODEL unset here
    rows = engine.metric_rows(report)
    slot_e = [r for r in rows if r[0] == "e5_domain"]
    assert slot_e and all(r[5] in ("measured", "extrapolated") for r in slot_e)
    assert not any(r[0] == "custom" for r in rows)  # slot E ran, so it is not "not feasible"
