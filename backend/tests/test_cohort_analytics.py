"""Phase 12: cohort benchmarking (spec §18, docs/adr/0020).

The world here holds every REAL student of data/processed/students_unified.csv (their real ids, branches,
graduation years, interests, domains and listed skills), embedded with the test TF-IDF provider, plus the synthetic
jobs of tests/matching_world.py. THE privacy test: a benchmark for student A, through the tool and through the
Matching Agent, never contains any other real student_id.
"""

import asyncio
import json
import re
from dataclasses import replace
from functools import lru_cache

import pandas as pd
import pytest

from app.agents import build_agents
from app.agents.job_intelligence_agent import job_section_texts
from app.agents.profile_agent import section_texts
from app.core.config import Settings
from app.db.matching_repository import InMemoryMatchingRepository
from app.matching import pipeline as P
from app.matching.calibration import Calibration, build_calibration
from app.matching.config import load_matching_config
from app.matching.entities import job_view, student_view
from app.schemas.agent_io import CohortBenchmark, MatchingAgentOutput
from app.schemas.agent_state import AgentState
from app.tools.matching_tools import build_context
from src.scoring import cohort_analytics as C
from tests.matching_world import JOBS

CSV = "data/processed/students_unified.csv"
A = "S00001"  # placement cell, Computer Science and Engineering (Data Science), 2027
B = "S00262"  # Kaggle resume dataset, Information Technology, "Java Developer", no graduation year


@lru_cache(maxsize=1)
def real_students() -> pd.DataFrame:
    return pd.read_csv(CSV, dtype=str, keep_default_na=False)


def real_profile(row) -> dict:
    """The structured profile the Profile Agent builds, from the record's real fields (skills: the listed ones)."""
    skills = [s.strip() for s in re.split(r"[,;\n]", row["skills_raw"]) if 1 < len(s.strip()) <= 40][:25]
    text = lambda v: v if v.strip() else "Not detected"  # noqa: E731
    return {"student_id": row["student_id"], "source": "stored_record", "degree": row["degree"] or "Not detected",
            "branch": row["branch"] or "Not detected",
            "graduation_year": int(float(row["graduation_year"])) if row["graduation_year"] else "Not detected",
            "avg_cgpa_sem5": None, "education": f"{row['degree']} {row['branch']}".strip() or "Not detected",
            "skills": [{"raw": s, "canonical": s, "category": "technical", "method": "exact", "evidence": None}
                       for s in skills] or "Skill not detected",
            "projects": text(row["technical_projects"]), "experience": text(row["internship_raw"]),
            "certifications": text(row["certifications"]), "achievements": "Not detected",
            "career_interests": row["career_interests"] or "Not detected", "domain": row["domain"] or "Not detected",
            "pii": "Not yet evaluated", "missing_fields": [], "extractor_version": "test"}


@pytest.fixture(scope="module")
def world(embedder):
    repo = InMemoryMatchingRepository()
    version, _ = embedder.describe()
    for _, row in real_students().iterrows():
        prof = real_profile(row)
        repo.add_student(student_view(row["student_id"], prof, {"country": row["country"] or None}), None)
        texts = section_texts(prof)
        for section, vec in zip(texts, embedder.embed(list(texts.values()), "query")):
            if vec is not None:
                repo.index.add("student", row["student_id"], section, vec, embedder.model_name, version)
    for jid, post in JOBS.items():
        repo.add_job(job_view(jid, post, {"country": post["country"], "near_duplicate_of": None,
                                          "source_dataset": "kaggle_job_descriptions"}))
        texts = job_section_texts(post)
        for section, vec in zip(texts, embedder.embed(list(texts.values()))):
            if vec is not None:
                repo.index.add("job", jid, section, vec, embedder.model_name, version)
    cfg = load_matching_config()
    cfg = replace(cfg, registration={"experiment_id": "test-experiment", "config_sha256": cfg.sha256})
    data = build_calibration([s for s, _ in repo.students()], repo.jobs(), embedder, cfg.features["transferable"],
                             [n for j in repo.jobs() for n, _ in j.skills])
    cal = Calibration(model_name=data["model_name"], model_version=data["model_version"],
                      quantiles={k: tuple(v) for k, v in data["quantiles"].items()},
                      transferable_threshold=data["transferable"]["threshold"], reference_pairs=data["reference_pairs"])
    settings = Settings(_env_file=None, environment="test", tool_retry_backoff_seconds=0.0, agent_timeout_seconds=120.0,
                        tool_timeout_seconds=120.0, matching_model="weighted_baseline_v1")
    ctx = build_context(repo, embedder, settings, cfg=cfg, calibration_loader=lambda *_: cal)
    return repo, cfg, cal, settings, ctx


def other_ids(student_id: str) -> set[str]:
    return set(real_students()["student_id"]) - {student_id}


# -- THE privacy test ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("student_id, job_id", [(A, "J1"), (A, None), (B, "J5")])
def test_a_benchmark_for_student_a_never_contains_another_real_student_id(world, student_id, job_id):
    *_, ctx = world
    result = CohortBenchmark.model_validate(C.benchmark(ctx, student_id, job_id))
    text = result.model_dump_json()
    leaked = sorted(i for i in other_ids(student_id) if i in text)
    print(f"\n{student_id} vs {job_id}: cohort {result.cohort_size}, comparable {result.comparable_size}, "
          f"{len(text)} chars searched for {len(other_ids(student_id))} other real student_ids -> {len(leaked)} found")
    assert result.comparable_size >= 5 and not isinstance(result.skill_coverage, str)  # real statistics were made
    assert leaked == []
    assert set(json.loads(text)) == set(CohortBenchmark.model_fields)  # nothing but the documented aggregates


def test_the_matching_agents_cohort_result_and_its_trace_hold_no_other_student_id(world):
    repo, cfg, cal, settings, _ = world
    agents = build_agents(settings=settings, profile_repo=None, job_repo=None, embedder=_embedder(world),
                          matching_repo=repo, learning_repo=None,
                          matching_overrides={"cfg": cfg, "calibration_loader": lambda *_: cal})
    out = asyncio.run(agents["matching_agent"].run(AgentState(request_id="req-cohort", task_type="test"),
                                                   {"mode": "jobs_for_student", "student_id": A}))
    assert out.status == "success", out.errors
    data = MatchingAgentOutput.model_validate(out.data)
    assert isinstance(data.cohort_benchmark, CohortBenchmark)
    assert data.cohort_benchmark.job_id == data.matches[0].job_id  # benchmarked against the top-ranked job
    cohort = data.cohort_benchmark.model_dump_json()
    assert not [i for i in other_ids(A) if i in cohort]
    assert out.tool_calls[-1].tool_name == "cohort_benchmark" and out.tool_calls[-1].success


def _embedder(world):
    return world[-1].provider


# -- what the statistics are --------------------------------------------------------------------------------------
def test_filters_use_the_real_fields_and_relax_only_when_the_cohort_is_too_small(world):
    repo, *_, ctx = world
    a = C.benchmark(ctx, A, "J1")
    applied = {f["name"]: f for f in a["filters"]}
    assert applied["branch"]["applied"] and applied["branch"]["value"] == "Computer Science and Engineering (Data Science)"
    assert applied["graduation_year"]["applied"] and applied["graduation_year"]["value"] == "2027"
    assert a["target_role"]["source"] == "job" and "data_ml" in a["target_role"]["families"]  # J1: Data Analyst
    eligible = [s for s, _ in repo.students() if s.student_id != A and s.branch == applied["branch"]["value"]
                and s.graduation_year == 2027 and C.student_target(s, ctx.cfg)[0] & {"data_ml"}]
    assert a["cohort_size"] == len(eligible) and a["relaxed"] == []
    assert a["comparable_size"] == min(len(eligible), C.load_config()["comparable"]["top_n"])
    # B (Kaggle): no graduation year stated, so that filter is not applied, never guessed
    b = {f["name"]: f for f in C.benchmark(ctx, B, "J5")["filters"]}
    assert not b["graduation_year"]["applied"] and "not stated" in b["graduation_year"]["note"]


def test_the_students_own_score_is_the_matching_agents_score_and_the_percentile_is_counted(world):
    *_, ctx = world
    result = C.benchmark(ctx, A, "J1")
    s = result["suitability"]
    stage2 = P.retrieval(ctx, "student", A, ["J1"], {}, 1)  # the Matching Agent's direction: jobs for a student
    sims = P.similarity(ctx, "student", A, ["J1"], {r["id"]: r["score"] for r in stage2["top"]})
    own = P.prediction(ctx, P.features(ctx, "student", A, ["J1"], sims["percentiles"])["features"])
    assert s["student_score"] == pytest.approx(own["predictions"]["J1"]["suitability"], abs=1e-6)
    # recompute the peer scores independently and count
    comparable = _comparable_ids(ctx, result)
    stage2 = P.retrieval(ctx, "job", "J1", comparable, {}, len(comparable))
    ids = [r["id"] for r in stage2["top"]]
    sims = P.similarity(ctx, "job", "J1", ids, {r["id"]: r["score"] for r in stage2["top"]})
    peer = [p["suitability"] for p in P.prediction(ctx, P.features(ctx, "job", "J1", ids, sims["percentiles"])[
        "features"])["predictions"].values() if p["suitability"] is not None]
    assert s["cohort_average"] == pytest.approx(sum(peer) / len(peer), abs=1e-6)
    below = sum(v < s["student_score"] for v in peer) + 0.5 * sum(v == s["student_score"] for v in peer)
    assert s["percentile"] == pytest.approx(round(100 * below / len(peer), 1))
    assert s["peers_scored"] == len(peer)


def _comparable_ids(ctx, result) -> list[str]:
    """Test-only: re-derive the comparable peers (the module never returns them)."""
    from app.embeddings.provider import QUERY

    student, _ = ctx.student(A)
    cohort = [s.student_id for s, _ in ctx.repo.students() if s.student_id != A and s.branch == student.branch
              and s.graduation_year == student.graduation_year
              and C.student_target(s, ctx.cfg)[0] & set(result["target_role"]["families"])]
    version, dim = ctx.provider.describe()
    sims = ctx.repo.index.similarities("student", "full", ctx.provider.embed([student.texts["full"]], QUERY)[0],
                                       ctx.provider.model_name, version, dim, ids=cohort)
    return [i for i, _ in sorted(sims.items(), key=lambda kv: (-kv[1], kv[0]))[:result["comparable_size"]]]


def test_skill_coverage_and_the_relative_gap_are_shares_of_the_comparable_peers(world):
    *_, ctx = world
    result = C.benchmark(ctx, A, "J1")
    cov, gap = result["skill_coverage"], result["relative_skill_gap"]
    n = result["comparable_size"]
    holders = C.load_config()["privacy"]["min_skill_holders"]
    assert all(row["peers_with_share"] * n >= holders - 1e-9 for row in cov["common_skills"])
    assert cov["coverage"] == pytest.approx(round(len(cov["covered"]) / len(cov["common_skills"]), 3))
    job_skills = {name for name, _ in ctx.job("J1").skills}
    listed = gap["distinctive_gaps"] + gap["shared_gaps"]
    assert {r["skill"] for r in listed} <= job_skills and len(listed) == gap["student_missing_count"]
    assert all(r["peers_with_share"] > 0.5 for r in gap["distinctive_gaps"])
    assert all(r["peers_with_share"] <= 0.5 for r in gap["shared_gaps"])


def test_too_small_a_cohort_returns_only_counts(world):
    *_, ctx = world
    cfg = {**C.load_config(), "privacy": {"min_cohort_size": 10_000, "min_skill_holders": 3}}
    result = CohortBenchmark.model_validate(C.benchmark(ctx, A, "J1", config=cfg))
    assert result.suitability == result.skill_coverage == result.relative_skill_gap == "Insufficient evidence"
    assert result.relaxed == ["graduation_year", "target_role"]  # both dropped before giving up; branch never
    assert next(f for f in result.filters if f.name == "branch").applied
    assert "privacy" in result.notes[0]


def test_the_profile_agent_runs_the_cohort_benchmark_for_a_cohort_question(world, sup_settings):
    """Routed by the Supervisor's deterministic router: task cohort_benchmark -> profile_agent -> the tool."""
    from app.db.profile_repository import InMemoryProfileRepository
    from app.db.store import InMemoryAgentRunStore
    from app.core.trace_tree import get_agent_trace

    repo, cfg, cal, settings, _ = world
    row = real_students().set_index("student_id").loc[A]
    profiles = InMemoryProfileRepository(students={A: {"student_id": A, **row.to_dict(), "source_dataset":
                                                       "placement_cell", "is_synthetic": False}})
    store = InMemoryAgentRunStore()
    agents = build_agents(settings=settings.model_copy(update={"supervisor_timeout_seconds": 120.0}), store=store,
                          profile_repo=profiles, job_repo=None, embedder=_embedder(world), fallback_embedder=None,
                          matching_repo=repo, learning_repo=None,
                          matching_overrides={"cfg": cfg, "calibration_loader": lambda *_: cal})
    resp = asyncio.run(agents["supervisor"].handle({"actor": "student", "request_kind": "question",
                                                    "request_text": "How do I compare with my peers?",
                                                    "student_id": A, "job_id": "J1"}))
    assert resp.task_type == "cohort_benchmark" and list(resp.results) == ["profile_agent"]
    cohort = resp.results["profile_agent"].data["cohort_benchmark"]
    assert cohort["job_id"] == "J1" and isinstance(cohort["suitability"], dict)
    tree = asyncio.run(get_agent_trace(resp.request_id, store))
    assert [t.name for t in tree.root.children[0].children][-1] == "cohort_benchmark"
    assert not [i for i in other_ids(A) if i in json.dumps(store.records, default=str) + tree.model_dump_json()]


def test_branches_compare_without_spacing_or_case():
    from src.scoring.cohort_analytics import branch_key

    assert branch_key("Computer Science and Engineering(Data Science)") == \
        branch_key("computer science and engineering (Data Science)")
    assert branch_key("Information Technology") != branch_key("Computer Engineering")
    assert branch_key(None) is None and branch_key("") is None
