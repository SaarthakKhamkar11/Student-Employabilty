"""Labelling page and API (Phase 6, human relevance benchmark).

    GET  /labelling                          the page (local only: ports bind to 127.0.0.1)
    GET  /labelling/api/status?labeller=AB   progress, rubric, and this labeller's next student
    GET  /labelling/api/students/{id}?labeller=AB   the student card and the jobs to grade
    PUT  /labelling/api/judgments            {student_id, job_id, labeller, relevance_label}
    GET  /labelling/api/export.csv           student_id,job_id,relevance_label,label_source,split
    GET  /labelling/api/export.manifest.json provenance + agreement, for the circular-evaluation check
"""

import csv
import io
from pathlib import Path

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from app.labelling import service
from app.labelling.repository import LabellingRepository

PAGE = Path(__file__).with_name("page.html")
router = APIRouter(prefix="/labelling", tags=["labelling"])
_repo: LabellingRepository | None = None


def configure(repo: LabellingRepository | None) -> None:
    global _repo
    _repo = repo


def get_repo() -> LabellingRepository:
    if _repo is None:
        raise HTTPException(503, "labelling needs the database (DATABASE_URL is not set)")
    return _repo


class JudgmentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student_id: str = Field(pattern=r"^S\d{5}$")
    job_id: str = Field(pattern=r"^J\d{7}$")
    labeller: str
    relevance_label: int


def _guard(fn, *args):
    try:
        return fn(*args)
    except service.LabellingError as exc:
        raise HTTPException(404 if "not in the labelling pool" in str(exc) else 400, str(exc)) from None
    except psycopg.Error:
        raise HTTPException(503, "database unavailable") from None


@router.get("", response_class=HTMLResponse)
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


@router.get("/api/status")
def api_status(labeller: str | None = None, repo: LabellingRepository = Depends(get_repo)) -> dict:
    return _guard(service.status, repo, labeller)


@router.get("/api/students/{student_id}")
def api_task(student_id: str, labeller: str, repo: LabellingRepository = Depends(get_repo)) -> dict:
    return _guard(service.task, repo, student_id, labeller)


@router.put("/api/judgments")
def api_save(body: JudgmentIn, repo: LabellingRepository = Depends(get_repo)) -> dict:
    return _guard(service.save, repo, body.student_id, body.job_id, body.labeller, body.relevance_label)


@router.get("/api/export.csv")
def api_export_csv(repo: LabellingRepository = Depends(get_repo)) -> Response:
    rows, _ = _guard(service.export, repo)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["student_id", "job_id", "relevance_label", "label_source", "split"],
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="human_labels.csv"'})


@router.get("/api/export.manifest.json")
def api_export_manifest(repo: LabellingRepository = Depends(get_repo)) -> dict:
    return _guard(service.export, repo)[1]
