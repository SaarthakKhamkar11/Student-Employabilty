"""Job Intelligence Agent (Phase 5): both pipelines, duplicates, malformed input, degradation, shared tools."""

import asyncio
import json

import pandas as pd
import pytest

from app.agents import build_agents
from app.db.job_repository import InMemoryJobRepository
from app.db.profile_repository import InMemoryProfileRepository
from app.db.store import InMemoryAgentRunStore
from app.schemas.agent_state import NOT_YET_EVALUATED, AgentState
from app.schemas.job import JobPosting
from tests.helpers import JOB
from tests.test_profile_agent import _break

JD = """Junior Data Analyst
About the role
Analyse sales data for retail clients. Questions: hr@example.com or +91 98765 43210.
Responsibilities:
- Build Power BI dashboards
- Write SQL queries and clean data in Python
Requirements: Python, SQL, Excel
Nice to have: Tableau, statistics
Experience: 0-2 years
Education: B.Tech or BCA"""


@pytest.fixture
def repo():
    return InMemoryJobRepository(jobs={"J0000001": dict(JOB, source_dataset="kaggle_job_descriptions")})


@pytest.fixture
def agents(settings, repo, embedder):
    return build_agents(settings=settings, profile_repo=InMemoryProfileRepository(), job_repo=repo, embedder=embedder)


def run(agents, payload):
    state = AgentState(request_id="req-job-test", task_type="ingest_job")
    return asyncio.run(agents["job_intelligence_agent"].run(state, payload)), state


def test_stored_row_becomes_a_posting_with_inferred_split_and_visibility_only_gender(agents, repo):
    out, state = run(agents, {"job_id": "J0000001"})
    assert out.status == "success", out.errors
    p = JobPosting.model_validate(out.data["posting"])
    assert p.source == "stored_record" and p.job_title == "Data Analyst" and p.company_name == "Example Co"
    assert p.min_degree == "B.TECH" and p.min_degree_level == 2
    assert (p.min_experience_years, p.max_experience_years) == (0.0, 2.0)
    assert p.skill_split.inferred is True and "INFERRED" in p.skill_split.note
    assert all(s.requirement_basis for s in p.skills)
    assert p.visibility.gender_preference == "both"
    assert out.data["saved"] and repo.postings["J0000001"]["job_id"] == "J0000001"
    assert state.model_versions["embedding_model"] == "tfidf_baseline"


def test_free_text_jd_uses_explicit_sections_and_masks_pii(agents, repo):
    out, _ = run(agents, {"jd_text": JD})
    assert out.status == "success", out.errors
    p = out.data["posting"]
    assert p["source"] == "job_description_text" and p["job_id"].startswith("JU")
    assert p["required_skills"] == ["Python", "SQL", "Excel"] and p["preferred_skills"] == ["Tableau", "Statistics"]
    assert {s["requirement_basis"] for s in p["skills"]} == {"explicit_required_section", "explicit_preferred_section"}
    assert p["responsibilities"] == ["Build Power BI dashboards", "Write SQL queries and clean data in Python"]
    assert p["pii"]["counts"] == {"email": 1, "phone": 1}
    assert "hr@example.com" not in json.dumps(p) and "98765" not in json.dumps(p)
    # no extractor reads company or location from free text: not evaluated, never guessed
    assert p["company_name"] == p["location"] == NOT_YET_EVALUATED
    assert out.data["saved"] and p["job_id"] in repo.jobs


def test_student_and_job_skills_share_one_normaliser(agents):
    profile_tools, job_tools = agents["profile_agent"].allowed_tools, agents["job_intelligence_agent"].allowed_tools
    assert {"normalize_skills", "detect_pii", "generate_embedding"} <= set(profile_tools) & set(job_tools)
    registry = agents["job_intelligence_agent"].registry
    assert registry is agents["profile_agent"].registry  # one registration of each shared tool
    out, _ = run(agents, {"jd_text": "Data Analyst\nRequirements: PowerBI, JS, ML"})
    assert out.data["posting"]["required_skills"] == ["Power BI", "JavaScript", "Machine Learning"]


def test_every_job_embedding_records_model_name_and_version(agents, repo):
    out, _ = run(agents, {"job_id": "J0000001"})
    info = out.data["embedding"]
    assert info["model_name"] == "tfidf_baseline" and info["model_version"].startswith("tfidf-") and info["interim"]
    rows = [e for k, e in repo.embeddings.items() if k[0] == "J0000001"]
    assert rows and all(e.model_name == "tfidf_baseline" and e.model_version == info["model_version"]
                        and e.indices is not None for e in rows)


def test_a_duplicate_jd_is_not_stored_again(agents, repo):
    first, _ = run(agents, {"jd_text": JD.replace("Junior Data Analyst", "Data Analyst")})
    jid = first.data["posting"]["job_id"]
    repo.jobs[jid].update(company_name="Example Co", responsibilities="", skills="")
    # same company + title + identical text as the stored J0000001 row would be required; here: same upload twice
    repo.jobs[jid]["job_description"] = repo.postings[jid]["job_description"]
    stored = dict(repo.jobs)
    flagged = asyncio.run(agents["job_intelligence_agent"].registry.call(
        "check_duplicates", {"job": {"company_name": "Example Co", "job_title": "Data Analyst",
                                     "job_description": repo.jobs[jid]["job_description"],
                                     "responsibilities": "", "skills": ""}}))
    assert flagged.near_duplicate_of in (jid, "J0000001") and repo.jobs == stored


@pytest.mark.parametrize("payload, reason", [({"jd_text": ""}, "empty"), ({"jd_text": "\x00\x01\x02\x03 x"}, "binary")])
def test_malformed_jd_is_a_structured_failure(agents, payload, reason):
    out, _ = run(agents, payload)
    assert out.status == "failed" and out.termination_reason == "invalid_input"
    assert "job description rejected" in out.errors[0].message and reason in out.errors[0].message


def test_unknown_job_id_is_rejected(agents):
    out, _ = run(agents, {"job_id": "J9999999"})
    assert out.termination_reason == "invalid_input" and "unknown job_id" in out.errors[0].message


def test_classifier_failure_degrades_split_to_not_yet_evaluated(agents):
    calls = _break(agents, "classify_skill_requirements")
    out, _ = run(agents, {"job_id": "J0000001"})
    assert calls["n"] == 2 and out.status == "partial"
    p = out.data["posting"]
    assert p["required_skills"] == p["preferred_skills"] == p["skill_split"] == NOT_YET_EVALUATED
    assert out.data["saved"]


def test_unverified_duplicates_are_never_stored(agents, repo):
    _break(agents, "check_duplicates")
    out, _ = run(agents, {"jd_text": JD})
    assert out.status == "partial" and out.data["stages"]["save_job"] == "skipped" and not out.data["saved"]
    assert out.data["posting"]["duplicate_check"] == NOT_YET_EVALUATED


def test_inferred_marker_reaches_the_api_level_response(sup_settings, embedder):
    supervisor = build_agents(settings=sup_settings, store=InMemoryAgentRunStore(),
                              profile_repo=InMemoryProfileRepository(), job_repo=InMemoryJobRepository(),
                              embedder=embedder)["supervisor"]
    resp = asyncio.run(supervisor.handle({"actor": "company", "request_kind": "jd_upload", "jd_text": JD}))
    posting = resp.results["job_intelligence_agent"].data["posting"]
    assert resp.results["job_intelligence_agent"].status == "success"
    assert posting["skill_split"]["inferred"] is True and "INFERRED" in posting["skill_split"]["note"]
    # Phase 7: Matching runs, but with no database to match against it fails instead of inventing a ranking
    assert resp.results["matching_agent"].termination_reason == "tool_failure"
    assert resp.results["matching_agent"].data in ("Not yet evaluated", "Insufficient evidence")


def test_real_dataset_rows_all_produce_valid_postings(settings, embedder):
    rows = pd.read_csv("data/processed/jobs.csv", dtype=str, keep_default_na=False).sample(25, random_state=1)
    repo = InMemoryJobRepository(jobs={r["job_id"]: dict(r) for r in rows.to_dict("records")})
    agents = build_agents(settings=settings, profile_repo=None, job_repo=repo, embedder=embedder)
    for job_id in rows.job_id:
        out, _ = run(agents, {"job_id": job_id})
        assert out.status == "success", (job_id, out.errors)
        assert out.data["posting"]["required_skills"] != "Skill not detected"


def test_jd_logs_carry_pii_flags_but_never_contact_details(agents, caplog):
    import logging

    from app.core.logging import JsonFormatter

    caplog.set_level(logging.DEBUG, logger="employability")
    run(agents, {"jd_text": JD})
    log_text = "\n".join(JsonFormatter().format(r) for r in caplog.records)
    assert '"event": "pii_flags"' in log_text
    assert "hr@example.com" not in log_text and "98765" not in log_text and "Analyse sales data" not in log_text
