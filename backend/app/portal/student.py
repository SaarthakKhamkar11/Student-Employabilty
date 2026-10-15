"""Student dashboard API (Phase 13). Every route requires a student account and acts on that student only.

    GET  /student/profile                 the stored Phase 4 profile + signup facts + embedding metadata
    POST /student/resume                  upload a resume -> Supervisor ingest_student_profile (parsed for review)
    PUT  /student/profile                 confirm the parsed profile, with corrections
    GET  /student/overview                recommend_jobs_for_student: ranked jobs, employability, top-job
                                          skill gaps, learning, explanation, cohort benchmark
    GET  /student/match/{job_id}          explain_match for one job: suitability, skill gaps, explanation
    GET  /student/learning/{job_id}       learning_recommendations for one job (Phase 10)
    GET  /student/benchmark               cohort_benchmark (Phase 12, aggregates only), for a job or the top job
    POST /student/match/{job_id}/ask-ai   Gemini prose for that match, on request only (cached)
"""

import base64
from collections import defaultdict
from statistics import mean
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.auth import require_role
from app.portal.corrections import ProfileCorrection, correct_profile
from app.portal.service import Portal, agent_data
from app.profile.text_extraction import MAX_FILE_BYTES
from app.schemas.agent_io import SupervisorInput

router = APIRouter(prefix="/student", tags=["student dashboard"])
_portal: Portal | None = None
Student = Depends(require_role("student"))


def configure(portal: Portal) -> None:
    global _portal
    _portal = portal


def portal() -> Portal:
    if _portal is None:
        raise HTTPException(503, "portal not configured")
    return _portal


def _sid(user: dict) -> str:
    if not user.get("student_id"):
        raise HTTPException(409, "This account has no student record. Sign up again as a student.")
    return user["student_id"]


def _need_profile(student_id: str) -> dict[str, Any]:
    row = portal().one("SELECT profile, profile_updated_at FROM student_profiles WHERE student_id = %s", (student_id,))
    if row is None or row["profile"] is None:
        raise HTTPException(409, "Upload your resume first: there is no parsed profile yet.")
    return row


@router.get("/profile")
def get_profile(user: dict = Student):
    sid = _sid(user)
    p = portal()
    row = p.one(
        "SELECT sp.student_id, sp.degree, sp.branch, sp.graduation_year, sp.career_interests, sp.domain, sp.profile, "
        "sp.profile_updated_at, sp.profile_extractor_version, u.profile_confirmed_at, u.details "
        "FROM student_profiles sp JOIN users u ON u.student_id = sp.student_id WHERE sp.student_id = %s", (sid,))
    if row is None:
        raise HTTPException(404, "student record not found")
    embeddings = p.query("SELECT model_name, model_version, section, dimension, created_at FROM embeddings "
                         "WHERE entity_type = 'student' AND entity_id = %s ORDER BY model_name, section", (sid,))
    stated = {k: row[k] for k in ("degree", "branch", "graduation_year", "career_interests", "domain")}
    return {
        "student_id": sid, "profile": row["profile"], "stated_at_signup": {**stated, **(row["details"] or {})},
        "extractor_version": row["profile_extractor_version"],
        "profile_updated_at": row["profile_updated_at"].isoformat() if row["profile_updated_at"] else None,
        "confirmed_at": row["profile_confirmed_at"].isoformat() if row["profile_confirmed_at"] else None,
        "embeddings": [{**e, "created_at": e["created_at"].isoformat()} for e in embeddings],
    }


@router.post("/resume")
async def upload_resume(file: UploadFile = File(...), user: dict = Student):
    """Runs the full ingest through the Supervisor; the parsed profile is returned for the student to confirm."""
    sid = _sid(user)
    data = await file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"File larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(422, "The file is empty")
    request = SupervisorInput(actor="student", request_kind="resume_upload", task_type="ingest_student_profile",
                              student_id=sid, resume_file_b64=base64.b64encode(data).decode(),
                              resume_filename=file.filename)
    run = await portal().run(request, user_id=user["user_id"])
    prof = run["agents"].get("profile_agent") or {}
    if prof.get("status") not in ("success", "partial") or not isinstance(agent_data(run, "profile_agent", "profile"), dict):
        errors = "; ".join(f"{e.get('code')}: {e.get('message')}" for e in run.get("errors") or [] if isinstance(e, dict))
        raise HTTPException(422, f"The resume could not be parsed ({prof.get('status', 'failed')}). {errors}".strip())
    # A new resume needs a new confirmation
    portal().query("UPDATE users SET profile_confirmed_at = NULL WHERE user_id = %s RETURNING user_id", (user["user_id"],))
    return {"run": run, "profile": agent_data(run, "profile_agent", "profile"),
            "embedding": agent_data(run, "profile_agent", "embedding"),
            "stages": agent_data(run, "profile_agent", "stages")}


@router.put("/profile")
async def confirm_profile(body: ProfileCorrection, user: dict = Student):
    sid = _sid(user)
    p = portal()
    row = _need_profile(sid)
    registry = p.quiet.agents["profile_agent"].registry
    profile, changed, saved = await correct_profile(registry, row["profile"], body)
    if changed:  # the stated columns are what the Profile Agent and the matching features read
        cols = {k: getattr(body, k) for k in ("degree", "branch", "graduation_year", "career_interests", "domain")
                if k in changed}
        if cols:
            sets = ", ".join(f"{k} = %s" for k in cols)
            p.query(f"UPDATE student_profiles SET {sets}, updated_at = now() WHERE student_id = %s RETURNING student_id",
                    (*[(v or None) for v in cols.values()], sid))
    p.query("UPDATE users SET profile_confirmed_at = now() WHERE user_id = %s RETURNING user_id", (user["user_id"],))
    p.audit(user["user_id"], "profile_confirmed", "student", sid, {"changed_fields": changed, **saved})
    return {"profile": profile, "changed_fields": changed, **saved}


def _career_paths(matches: list[dict]) -> list[dict]:
    """Career directions = the student's ranked matches grouped by the jobs' role. An aggregation of the Matching
    Agent's output, not a separate model (there is no career model in Phases 2-12)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for m in matches:
        role = (m.get("summary") or {}).get("role") or (m.get("summary") or {}).get("job_title")
        if role and isinstance(m.get("suitability_score"), (int, float)):
            groups[role].append(m)
    paths = [{"role": role, "jobs": len(ms), "best_rank": min(x["rank"] for x in ms),
              "mean_suitability": mean(x["suitability_score"] for x in ms),
              "example_titles": sorted({(x.get("summary") or {}).get("job_title") for x in ms} - {None})[:3]}
             for role, ms in groups.items()]
    return sorted(paths, key=lambda r: (r["best_rank"], -r["jobs"]))


@router.get("/overview")
async def overview(refresh: bool = False, user: dict = Student):
    sid = _sid(user)
    _need_profile(sid)
    run = await portal().result("recommend_jobs_for_student", actor="student", student_id=sid, refresh=refresh,
                                also_accept=("ingest_student_profile",), user_id=user["user_id"])
    matches = agent_data(run, "matching_agent", "matches")
    return {"run": run, "career_paths": _career_paths(matches) if isinstance(matches, list) else []}


@router.get("/match/{job_id}")
async def match(job_id: str, refresh: bool = False, user: dict = Student):
    sid = _sid(user)
    _need_profile(sid)
    p = portal()
    run = await p.result("explain_match", actor="student", student_id=sid, job_id=job_id, refresh=refresh,
                         user_id=user["user_id"])
    matches = agent_data(run, "matching_agent", "matches")
    versions = matches[0].get("model_versions") if isinstance(matches, list) and matches else None
    return {"run": run, "ai_summary": p.cached_ai(sid, job_id, versions)}


@router.get("/learning/{job_id}")
async def learning(job_id: str, refresh: bool = False, user: dict = Student):
    sid = _sid(user)
    _need_profile(sid)
    run = await portal().result("learning_recommendations", actor="student", student_id=sid, job_id=job_id,
                                refresh=refresh, user_id=user["user_id"])
    return {"run": run}


@router.get("/benchmark")
async def benchmark(job_id: str | None = None, refresh: bool = False, user: dict = Student):
    sid = _sid(user)
    _need_profile(sid)
    run = await portal().result("cohort_benchmark", actor="student", student_id=sid, job_id=job_id, refresh=refresh,
                                user_id=user["user_id"])
    return {"run": run}


@router.post("/match/{job_id}/ask-ai")
async def ask_ai(job_id: str, user: dict = Student):
    sid = _sid(user)
    _need_profile(sid)
    return await portal().ask_ai(actor="student", student_id=sid, job_id=job_id, user_id=user["user_id"])
