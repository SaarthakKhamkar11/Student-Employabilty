"""Profile Agent (Phase 4): the pipeline, malformed resumes, degradation, PII safety, model metadata."""

import asyncio
import base64
import json
import logging

import pytest

from app.agents import build_agents
from app.agents.supervisor import SupervisorAgent
from app.core.logging import JsonFormatter
from app.db.profile_repository import InMemoryProfileRepository
from app.db.store import InMemoryAgentRunStore
from app.schemas.agent_state import NOT_YET_EVALUATED, SKILL_NOT_DETECTED, AgentState

# Synthetic resume: every personal detail below is invented.
SYNTHETIC_RESUME = """Asha Kulkarni | asha.kulkarni@example.com | +91 98765 43210
Address: 12 MG Road, Pune 411001
linkedin.com/in/asha-kulkarni-test
Education Details
B.E Computer Engineering, Example Institute of Technology
Skill Details
Python- Exprience - 24 months
JS, PowerBI, React.js, Git/GitHub
Machine learning
Company Details
Data Analyst Intern at Example Analytics, 6 months
Projects
Crop disease detection with CNNs
Sales dashboard in Power BI
Certifications
AWS Cloud Practitioner
"""
PII_VALUES = ["Asha Kulkarni", "asha.kulkarni@example.com", "98765 43210", "12 MG Road", "411001",
              "linkedin.com/in/asha-kulkarni-test"]
STORED = {"student_id": "S00001", "degree": "B.E/ B.Tech", "branch": "Computer Science and Engineering (Data Science)",
          "graduation_year": 2027, "gender": "female", "avg_cgpa_sem5": 8.3, "achievements": "SIH 2024 finalist",
          "career_interests": "Placements", "domain": "Data Analyst and Machine Learning",
          "internship_raw": "Example Minds - Python Developer - 2 months", "technical_projects": "Crop Disease Detection",
          "skills_raw": "Programming Languages: C, C++, Python\nTools: PowerBI, Git", "certifications": "",
          "country": "India", "source_dataset": "placement_cell", "is_synthetic": False, "synthetic_fields": None}


@pytest.fixture
def repo():
    return InMemoryProfileRepository(students={"S00001": dict(STORED)})


@pytest.fixture
def agents(settings, repo, embedder):
    return build_agents(settings=settings, profile_repo=repo, embedder=embedder)


def run(agents, payload, state=None):
    state = state or AgentState(request_id="req-profile-test", task_type="ingest_student_profile")
    return asyncio.run(agents["profile_agent"].run(state, payload)), state


def profile_of(out):
    return out.data["profile"]


# -- the three input paths ---------------------------------------------------------------------
def test_resume_text_becomes_a_structured_masked_profile(agents, repo):
    out, state = run(agents, {"resume_text": SYNTHETIC_RESUME})
    assert out.status == "success", out.errors
    p = profile_of(out)
    assert p["source"] == "resume" and p["student_id"].startswith("U")
    canon = {s["canonical"]: s for s in p["skills"]}
    assert {"Python", "JavaScript", "Power BI", "React", "Git", "GitHub", "Machine Learning"} <= set(canon)
    assert canon["Python"]["evidence"] == "24 months" and canon["JavaScript"]["method"] == "alias"
    assert not any("\r" in s["raw"] for s in p["skills"])
    assert p["degree"] == "B.E" and p["projects"] == ["Crop disease detection with CNNs", "Sales dashboard in Power BI"]
    assert p["certifications"] == ["AWS Cloud Practitioner"]
    # a resume does not state these: no extractor, so not evaluated (never guessed)
    assert p["branch"] == p["graduation_year"] == p["avg_cgpa_sem5"] == NOT_YET_EVALUATED
    assert {"branch", "graduation_year", "avg_cgpa_sem5"} <= set(p["missing_fields"])
    assert p["pii"]["counts"] == {"name": 1, "email": 1, "phone": 1, "address": 1, "url": 1}
    assert not any(v in json.dumps(p) for v in PII_VALUES)  # the structured profile carries no PII
    assert out.data["saved"] and repo.profiles[p["student_id"]]["source"] == "resume"


def test_stored_record_path_uses_columns_and_drops_gender(agents):
    out, _ = run(agents, {"student_id": "S00001"})
    assert out.status == "success", out.errors
    p = profile_of(out)
    assert p["source"] == "stored_record" and p["branch"].startswith("Computer Science")
    assert p["graduation_year"] == 2027 and p["avg_cgpa_sem5"] == 8.3
    assert {"C", "C++", "Python", "Power BI", "Git"} <= {s["canonical"] for s in p["skills"]}
    assert p["certifications"] == "Not detected"  # blank in the placement form: looked for, not there
    assert "gender" not in p and "female" not in json.dumps(p)


def test_stored_record_plus_resume_merges_both(agents):
    out, _ = run(agents, {"student_id": "S00001", "resume_text": SYNTHETIC_RESUME})
    p = profile_of(out)
    assert out.status == "success" and p["source"] == "stored_record+resume" and p["student_id"] == "S00001"
    assert p["graduation_year"] == 2027 and "React" in {s["canonical"] for s in p["skills"]}


def test_resume_without_a_skills_section_says_skill_not_detected(agents):
    out, _ = run(agents, {"resume_text": "Education Details\r\nB.Sc Physics\r\nCompany Details\r\nTutor, 1 year"})
    assert profile_of(out)["skills"] == SKILL_NOT_DETECTED


# -- embeddings carry model name and version -------------------------------------------------------
def test_every_stored_embedding_records_model_name_and_version(agents, repo):
    out, state = run(agents, {"resume_text": SYNTHETIC_RESUME})
    info = out.data["embedding"]
    assert info["model_name"] == "tfidf_baseline" and info["model_version"].startswith("tfidf-")
    assert info["interim"] is True and info["dimension"] == 4096
    assert set(info["sections"]) == {"skills", "experience", "education", "projects", "certifications", "full"}
    rows = [e for k, e in repo.embeddings.items() if k[0] == profile_of(out)["student_id"]]
    assert len(rows) == 6 and all(e.model_name == "tfidf_baseline" and e.model_version == info["model_version"]
                                  for e in rows)
    assert state.model_versions["embedding_model"] == "tfidf_baseline"


def test_sections_with_sentinels_are_never_embedded(agents, repo):
    out, _ = run(agents, {"resume_text": "Skill Details\nPython, SQL"})
    assert set(out.data["embedding"]["sections"]) == {"skills", "full"}  # no empty-text vectors (audit I-3)


# -- malformed resumes ---------------------------------------------------------------------------
@pytest.mark.parametrize("payload, reason", [
    ({"resume_text": ""}, "empty"),
    ({"resume_text": "   \n\t "}, "empty"),
    ({"resume_text": "\x01\x02\x03\x04\x05\x06\x07\x08 garbage"}, "binary or garbled"),
    ({"resume_file_b64": base64.b64encode(b"\x80\x81\xfe\x90 not utf8 \xc3\x28").decode(),
      "resume_filename": "resume.txt"}, "not valid UTF-8"),
    ({"resume_file_b64": base64.b64encode(b"%PDF-1.4 broken").decode(), "resume_filename": "cv.pdf"}, "PDF"),
    ({"resume_file_b64": base64.b64encode(b"MZ\x90\x00binary").decode(), "resume_filename": "cv.exe"},
     "unsupported file type"),
    ({"resume_file_b64": "not base64!!", "resume_filename": "cv.txt"}, "base64"),
])
def test_malformed_resume_is_a_structured_failure_not_a_crash(agents, payload, reason):
    out, state = run(agents, payload)
    assert out.status == "failed" and out.termination_reason == "invalid_input"
    assert out.errors[0].code == "invalid_input" and reason in out.errors[0].message
    assert out.tool_calls[0].attempts == 1  # deterministic rejection: not retried
    assert state.errors and state.errors[0]["code"] == "invalid_input"


def test_supervisor_sees_failed_with_a_clear_error_for_an_empty_resume(sup_settings, embedder):
    supervisor: SupervisorAgent = build_agents(settings=sup_settings, store=InMemoryAgentRunStore(),
                                               profile_repo=InMemoryProfileRepository(),
                                               embedder=embedder)["supervisor"]
    resp = asyncio.run(supervisor.handle({"actor": "student", "request_kind": "resume_upload", "resume_text": ""}))
    profile = resp.results["profile_agent"]
    assert resp.status == "failed" and profile.status == "failed"
    assert profile.termination_reason == "invalid_input" and profile.attempts == 1  # no Supervisor retry either
    assert any(e["code"] == "invalid_input" and "empty" in e["message"] for e in resp.errors)


def test_unknown_student_without_resume_is_rejected(agents):
    out, _ = run(agents, {"student_id": "S99999"})
    assert out.termination_reason == "invalid_input" and "unknown student_id" in out.errors[0].message


# -- degradation: retried once, then degrade -------------------------------------------------------
def _break(agents, tool, exc=RuntimeError("simulated failure")):
    registry = agents["profile_agent"].registry
    spec = registry.get(tool)
    calls = {"n": 0}

    def broken(**kwargs):
        calls["n"] += 1
        raise exc

    registry._tools[tool] = spec.__class__(spec.name, broken, spec.output_schema, spec.input_schema, spec.description)
    return calls


def test_pii_detector_failure_degrades_without_sending_text_to_ml_or_storage(agents, repo):
    calls = _break(agents, "detect_pii")
    out, _ = run(agents, {"resume_text": SYNTHETIC_RESUME})
    assert calls["n"] == 2  # retried once
    assert out.status == "partial" and out.data["stages"]["detect_pii"] == "failed"
    assert out.data["stages"]["generate_embedding"] == "skipped" and out.data["stages"]["save_profile"] == "skipped"
    p = profile_of(out)
    assert p["pii"] == NOT_YET_EVALUATED and p["skills"]  # the parsed sections are kept
    assert "asha.kulkarni@example.com" not in json.dumps(p)  # fallback mask still hides emails/phones
    assert not out.data["saved"] and not repo.profiles


def test_skill_extractor_failure_degrades_to_not_yet_evaluated(agents, repo):
    calls = _break(agents, "extract_skills")
    out, _ = run(agents, {"resume_text": SYNTHETIC_RESUME})
    assert calls["n"] == 2 and out.status == "partial"
    assert profile_of(out)["skills"] == NOT_YET_EVALUATED  # not "Skill not detected": extraction never happened
    assert out.data["stages"]["normalize_skills"] == "skipped" and out.data["saved"]


def test_normalizer_failure_keeps_skills_marked_unmapped(agents):
    _break(agents, "normalize_skills")
    out, _ = run(agents, {"resume_text": SYNTHETIC_RESUME})
    assert out.status == "partial" and all(s["method"] == "unmapped" for s in profile_of(out)["skills"])


def test_embedding_failure_still_saves_the_profile(agents):
    _break(agents, "generate_embedding")
    out, _ = run(agents, {"resume_text": SYNTHETIC_RESUME})
    assert out.status == "partial" and out.data["saved"] and out.data["embedding"] == NOT_YET_EVALUATED


def test_database_failure_returns_the_profile_unsaved(agents):
    _break(agents, "save_profile", ConnectionError("db down"))
    out, _ = run(agents, {"resume_text": SYNTHETIC_RESUME})
    assert out.status == "partial" and not out.data["saved"] and profile_of(out)["skills"]


def test_a_profile_containing_pii_is_never_saved(agents):
    validate = agents["profile_agent"].registry.get("validate_profile").func
    out, _ = run(agents, {"resume_text": SYNTHETIC_RESUME})
    leaked = dict(profile_of(out), achievements="call me on 98765 43210")
    result = validate(profile=leaked)
    assert result["valid"] is False and "PII present" in result["problems"][0]


# -- no PII in logs ----------------------------------------------------------------------------------
def test_logs_contain_pii_flags_but_never_pii(agents, caplog):
    caplog.set_level(logging.DEBUG, logger="employability")
    run(agents, {"resume_text": SYNTHETIC_RESUME})
    lines = [JsonFormatter().format(r) for r in caplog.records]
    flags = next(json.loads(line) for line in lines if '"event": "pii_flags"' in line)
    assert {f["type"]: f["count"] for f in flags["flags"]} == {"name": 1, "email": 1, "phone": 1, "address": 1,
                                                               "url": 1}
    log_text = "\n".join(lines)
    assert not [v for v in PII_VALUES if v in log_text]
    assert "Crop disease detection" not in log_text  # no resume content either
