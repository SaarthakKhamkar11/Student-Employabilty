"""Company dashboard API (Phase 13). Every route requires a company account ("recruiter" role) and only touches
jobs that account posted.

    GET  /company/overview                          KPIs from stored runs and shortlist events
    GET  /company/jobs                              the company's jobs
    POST /company/jobs                              JD text, a JD file, or the job creation form -> Supervisor
                                                    ingest_job (parsed for review)
    GET  /company/jobs/{job_id}                     the stored posting (for confirmation)
    PUT  /company/jobs/{job_id}                     confirm the parsed posting, with corrections
    GET  /company/jobs/{job_id}/candidates          rank_candidates_for_job + the current shortlist
    GET  /company/jobs/{job_id}/candidates/{sid}    explain_match for one candidate (skill comparison + explanation)
    POST /company/jobs/{job_id}/candidates/{sid}/ask-ai
    POST /company/jobs/{job_id}/shortlist           shortlist / un-shortlist: writes a feedback_events row
    GET  /company/shortlist                         every currently shortlisted candidate
"""

import base64
from statistics import mean
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from app.auth import require_role
from app.portal.corrections import JobCorrection, correct_posting
from app.portal.service import Portal, agent_data
from app.profile.text_extraction import MAX_FILE_BYTES, ResumeRejected, extract_text
from app.schemas.agent_io import SupervisorInput

router = APIRouter(prefix="/company", tags=["company dashboard"])
_portal: Portal | None = None
Company = Depends(require_role("recruiter"))


def configure(portal: Portal) -> None:
    global _portal
    _portal = portal


def portal() -> Portal:
    if _portal is None:
        raise HTTPException(503, "portal not configured")
    return _portal


def _own_job(user: dict, job_id: str) -> dict:
    row = portal().one("SELECT job_id, job_title, posting, confirmed_at, posted_by, created_at, posting_updated_at "
                       "FROM jobs WHERE job_id = %s", (job_id,))
    if row is None:
        raise HTTPException(404, "Job not found")
    if row["posted_by"] != user["user_id"]:
        raise HTTPException(403, "This job belongs to another account")
    return row


def _known(value) -> str | None:
    return value if isinstance(value, str) and value not in ("Not detected", "Not yet evaluated") else None


def _company(user: dict) -> dict:
    row = portal().one("SELECT company_id, name FROM companies WHERE company_id = %s", (user.get("company_id"),))
    if row is None:
        raise HTTPException(409, "This account has no company record")
    return row


# ------------------------------------------------------------------------------------------------------- jobs
class JobForm(BaseModel):
    """The job creation form. It is written out as a plain job description and parsed by the Job Intelligence
    Agent like any uploaded JD, so a form-created job and an uploaded one take the same path."""

    job_title: str = Field(min_length=2, max_length=200)
    location: str | None = Field(None, max_length=200)
    work_type: str | None = Field(None, max_length=50)
    description: str = Field(min_length=20, max_length=20_000)
    responsibilities: list[str] = Field(default_factory=list, max_length=60)
    required_skills: list[str] = Field(default_factory=list, max_length=80)
    preferred_skills: list[str] = Field(default_factory=list, max_length=80)
    qualification: str | None = Field(None, max_length=500)
    experience: str | None = Field(None, max_length=200)

    def as_text(self, company: str) -> str:
        lines = [self.job_title, f"Company: {company}"]
        if self.location or self.work_type:
            lines.append(" | ".join(x for x in (f"Location: {self.location}" if self.location else "",
                                                f"Work type: {self.work_type}" if self.work_type else "") if x))
        lines += ["", "Job Description", self.description.strip()]
        if self.responsibilities:
            lines += ["", "Responsibilities", *[f"- {r}" for r in self.responsibilities if r.strip()]]
        if self.required_skills:
            lines += ["", "Required Skills", ", ".join(self.required_skills)]
        if self.preferred_skills:
            lines += ["", "Preferred Skills", ", ".join(self.preferred_skills)]
        if self.qualification:
            lines += ["", "Qualifications", self.qualification]
        if self.experience:
            lines += ["", "Experience", self.experience]
        return "\n".join(lines)


async def _ingest(user: dict, jd_text: str, company_name: str | None = None) -> dict:
    """company_name: the name the company typed with the JD; it wins over a name parsed from the text, and the
    account's company is the fallback."""
    p = portal()
    company = _company(user)
    request = SupervisorInput(actor="company", request_kind="jd_upload", task_type="ingest_job", jd_text=jd_text)
    run = await p.run(request, user_id=user["user_id"])
    posting = agent_data(run, "job_intelligence_agent", "posting")
    if not isinstance(posting, dict):
        errors = "; ".join(f"{e.get('code')}: {e.get('message')}" for e in run.get("errors") or [] if isinstance(e, dict))
        raise HTTPException(422, f"The job description could not be parsed. {errors}".strip())
    job_id = posting["job_id"]
    saved = agent_data(run, "job_intelligence_agent", "saved")
    dup = posting.get("duplicate_check") if isinstance(posting.get("duplicate_check"), dict) else {}
    if not saved and dup.get("near_duplicate_of"):
        raise HTTPException(409, f"This description duplicates a job already in the system ({job_id}); "
                                 "it was not saved again.")
    if not saved:
        raise HTTPException(422, "The parsed job failed validation and was not saved: "
                                 + "; ".join(str(e.get("message")) for e in run.get("errors") or [] if isinstance(e, dict)))
    name = " ".join((company_name or "").split()) or company["name"]
    posting["company_name"] = name
    if posting.get("missing_fields") and "company_name" in posting["missing_fields"]:
        posting["missing_fields"] = [f for f in posting["missing_fields"] if f != "company_name"]
    with p.connect() as conn:
        conn.execute("UPDATE jobs SET posted_by = %s, company_id = %s, company_name = %s, posting = %s, "
                     "role = %s, location = %s, work_type = %s, salary_range = %s, updated_at = now() "
                     "WHERE job_id = %s",
                     (user["user_id"], company["company_id"], name, Jsonb(posting),
                      *[_known(posting.get(k)) for k in ("role", "location", "work_type", "salary_range")], job_id))
    p.audit(user["user_id"], "job_posted", "job", job_id, {"request_id": run["request_id"]}, run["request_id"])
    return {"job_id": job_id, "posting": posting, "run": run,
            "embedding": agent_data(run, "job_intelligence_agent", "embedding")}


@router.post("/jobs")
async def create_job(form: JobForm, user: dict = Company):
    return await _ingest(user, form.as_text(_company(user)["name"]))


@router.post("/jobs/upload")
async def upload_jd(file: UploadFile | None = File(None), jd_text: str | None = Form(None),
                    company_name: str | None = Form(None, max_length=200), user: dict = Company):
    """The company sends only its name and the JD (a file or pasted text). Title, role, location, work type,
    salary and the skill requirements are parsed from the JD, then shown back for confirmation."""
    if file is not None:
        data = await file.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(413, f"File larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
        try:
            text, _ = extract_text(None, base64.b64encode(data).decode(), file.filename)
        except ResumeRejected as exc:
            raise HTTPException(422, "Job description rejected: " + str(exc).replace("resume", "job description"))
    elif jd_text and jd_text.strip():
        text = jd_text
    else:
        raise HTTPException(422, "Send a .pdf/.docx/.txt file or paste the job description text")
    return await _ingest(user, text, company_name)


@router.get("/jobs")
def list_jobs(user: dict = Company):
    rows = portal().query(
        "SELECT job_id, job_title, created_at, confirmed_at, posting->'required_skills' AS required_skills, "
        "posting->>'location' AS location, posting->>'work_type' AS work_type, "
        "posting->>'salary_range' AS salary_range, posting->>'company_name' AS company_name FROM jobs WHERE posted_by = %s "
        "ORDER BY created_at DESC", (user["user_id"],))
    return [{**r, "created_at": r["created_at"].isoformat(),
             "confirmed_at": r["confirmed_at"].isoformat() if r["confirmed_at"] else None} for r in rows]


@router.get("/jobs/{job_id}")
def get_job(job_id: str, user: dict = Company):
    row = _own_job(user, job_id)
    emb = portal().query("SELECT model_name, model_version, section FROM embeddings WHERE entity_type = 'job' "
                         "AND entity_id = %s ORDER BY model_name, section", (job_id,))
    return {"job_id": job_id, "posting": row["posting"], "embeddings": emb,
            "confirmed_at": row["confirmed_at"].isoformat() if row["confirmed_at"] else None}


@router.put("/jobs/{job_id}")
async def confirm_job(job_id: str, body: JobCorrection, user: dict = Company):
    row = _own_job(user, job_id)
    p = portal()
    registry = p.quiet.agents["job_intelligence_agent"].registry
    posting, changed, saved = await correct_posting(registry, row["posting"], body)
    with p.connect() as conn:
        if changed:  # the raw columns mirror the confirmed posting (company lists and eligibility read them)
            conn.execute("UPDATE jobs SET job_title = %s, role = %s, location = %s, work_type = %s, company_name = %s, "
                         "salary_range = %s WHERE job_id = %s",
                         tuple(_known(posting.get(k)) for k in ("job_title", "role", "location", "work_type",
                                                                "company_name", "salary_range")) + (job_id,))
        conn.execute("UPDATE jobs SET confirmed_at = now() WHERE job_id = %s", (job_id,))
    p.audit(user["user_id"], "job_confirmed", "job", job_id, {"changed_fields": changed, **saved})
    return {"posting": posting, "changed_fields": changed, **saved}


# ------------------------------------------------------------------------------------------------- candidates
def _shortlist(job_ids: list[str]) -> dict[tuple[str, str], dict]:
    """The latest shortlist decision per (job, student): the shortlist is the feedback_events log, nothing else."""
    if not job_ids:
        return {}
    rows = portal().query(
        "SELECT DISTINCT ON (job_id, student_id) job_id, student_id, event_type, created_at, payload, id "
        "FROM feedback_events WHERE job_id = ANY(%s) AND event_type IN ('shortlist', 'unshortlist') "
        "ORDER BY job_id, student_id, created_at DESC, id DESC", (job_ids,))
    return {(r["job_id"], r["student_id"]): r for r in rows if r["event_type"] == "shortlist"}


@router.get("/jobs/{job_id}/candidates")
async def candidates(job_id: str, refresh: bool = False, user: dict = Company):
    row = _own_job(user, job_id)
    if row["confirmed_at"] is None:
        raise HTTPException(409, "Confirm the parsed job first: candidates are ranked against the confirmed posting.")
    run = await portal().result("rank_candidates_for_job", actor="company", job_id=job_id, refresh=refresh,
                                also_accept=("ingest_job",), user_id=user["user_id"])
    listed = _shortlist([job_id])
    return {"run": run, "shortlisted": sorted(sid for (_, sid) in listed)}


@router.get("/jobs/{job_id}/candidates/{student_id}")
async def candidate(job_id: str, student_id: str, refresh: bool = False, user: dict = Company):
    _own_job(user, job_id)
    p = portal()
    run = await p.result("explain_match", actor="company", student_id=student_id, job_id=job_id, refresh=refresh,
                         user_id=user["user_id"])
    matches = agent_data(run, "matching_agent", "matches")
    versions = matches[0].get("model_versions") if isinstance(matches, list) and matches else None
    return {"run": run, "ai_summary": p.cached_ai(student_id, job_id, versions)}


@router.post("/jobs/{job_id}/candidates/{student_id}/ask-ai")
async def ask_ai(job_id: str, student_id: str, user: dict = Company):
    _own_job(user, job_id)
    return await portal().ask_ai(actor="company", student_id=student_id, job_id=job_id, user_id=user["user_id"])


class ShortlistIn(BaseModel):
    student_id: str = Field(min_length=1, max_length=64)
    action: Literal["shortlist", "unshortlist"] = "shortlist"
    request_id: str | None = None  # the ranking run the decision was made on
    rank: int | None = None
    suitability_score: float | None = None
    model_versions: dict | None = None


@router.post("/jobs/{job_id}/shortlist")
def shortlist(job_id: str, body: ShortlistIn, user: dict = Company):
    """One decision = one feedback_events row (Phase 15's feedback loop reads them). Never frontend-only state."""
    _own_job(user, job_id)
    p = portal()
    if p.one("SELECT 1 AS ok FROM student_profiles WHERE student_id = %s", (body.student_id,)) is None:
        raise HTTPException(404, "Unknown candidate")
    payload = {"request_id": body.request_id, "rank": body.rank, "suitability_score": body.suitability_score,
               "model_versions": body.model_versions or {}, "surface": "company_dashboard"}
    row = p.one(
        "INSERT INTO feedback_events (source, event_type, user_id, student_id, job_id, payload) "
        "VALUES ('real_deployment', %s, %s, %s, %s, %s) RETURNING id, source, event_type, user_id, student_id, "
        "job_id, payload, created_at", (body.action, user["user_id"], body.student_id, job_id, Jsonb(payload)))
    return {**row, "user_id": str(row["user_id"]), "created_at": row["created_at"].isoformat()}


@router.get("/shortlist")
def shortlist_all(user: dict = Company):
    jobs = portal().query("SELECT job_id, job_title FROM jobs WHERE posted_by = %s", (user["user_id"],))
    titles = {j["job_id"]: j["job_title"] for j in jobs}
    listed = _shortlist(list(titles))
    return [{"job_id": j, "job_title": titles[j], "student_id": s, "shortlisted_at": r["created_at"].isoformat(),
             "event_id": r["id"], "payload": r["payload"]} for (j, s), r in
            sorted(listed.items(), key=lambda kv: kv[1]["created_at"], reverse=True)]


@router.get("/overview")
def overview(user: dict = Company):
    p = portal()
    jobs = p.query("SELECT job_id, job_title, confirmed_at, created_at FROM jobs WHERE posted_by = %s "
                   "ORDER BY created_at", (user["user_id"],))
    per_job = []
    for j in jobs:
        found = p.stored(("rank_candidates_for_job", "ingest_job"), None, j["job_id"], current_only=False)
        matches = agent_data(found, "matching_agent", "matches") if found else None
        scores = [m["suitability_score"] for m in matches or [] if isinstance(m.get("suitability_score"), (int, float))]
        per_job.append({"job_id": j["job_id"], "job_title": j["job_title"],
                        "confirmed": j["confirmed_at"] is not None, "ranked_at": found["finished_at"] if found else None,
                        "candidates": len(matches) if isinstance(matches, list) else None,
                        "top5_mean_suitability": mean(sorted(scores, reverse=True)[:5]) if scores else None,
                        "scores": sorted(scores, reverse=True)})
    listed = _shortlist([j["job_id"] for j in jobs])
    ranked = [x for x in per_job if x["candidates"] is not None]
    top = [x["top5_mean_suitability"] for x in ranked if x["top5_mean_suitability"] is not None]
    return {"jobs_posted": len(jobs), "candidates_matched": sum(x["candidates"] for x in ranked) if ranked else None,
            "shortlisted": len(listed), "avg_top_suitability": mean(top) if top else None, "jobs": per_job}
