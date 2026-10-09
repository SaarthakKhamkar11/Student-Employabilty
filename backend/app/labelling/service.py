"""Labelling-page logic: who labels which student next, what is shown, and the export.

Every pair needs one judgment. A fixed 4 students (chosen by hash, so the same ones every
time) are also given to a second person, which measures how much labellers agree. The
export keeps only students whose pool is fully judged. Where two people graded a pair it
takes the median, rounding ties down: a conservative choice that never inflates relevance.
"""

import hashlib
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from app.labelling.repository import JOB_FIELDS, LabellingRepository
from app.labelling.rubric import GRADES, IGNORE, LABEL_SOURCE, QUESTION, RUBRIC_VERSION
from app.profile.pii import DETECTOR_VERSION, detect, mask

OVERLAP_STUDENTS = 4
_INITIALS = re.compile(r"^[A-Za-z]{2,8}$")
_SHOWN_STUDENT_FIELDS = {
    "degree": "Degree", "branch": "Branch", "domain": "Career domain (the student's stated interest)",
    "skills_raw": "Skills", "internship_raw": "Experience / internships", "technical_projects": "Projects",
    "certifications": "Certifications", "achievements": "Achievements",
}


class LabellingError(ValueError):
    pass


def normalise_labeller(value: str) -> str:
    if not _INITIALS.match(value or ""):
        raise LabellingError("labeller must be your initials: 2 to 8 letters")
    return value.upper()


def overlap_students(pool_version: str, student_ids) -> set[str]:
    ranked = sorted(set(student_ids), key=lambda s: hashlib.sha256(f"{pool_version}:{s}".encode()).hexdigest())
    return set(ranked[:OVERLAP_STUDENTS])


def _by_student(rows) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["student_id"]].append(r)
    return grouped


def _coverage(pool: list[dict], judgments: list[dict]):
    jobs_of = {s: {r["job_id"] for r in rows} for s, rows in _by_student(pool).items()}
    judged = defaultdict(lambda: defaultdict(set))  # student -> labeller -> job ids
    for j in judgments:
        judged[j["student_id"]][j["labeller"]].add(j["job_id"])
    complete = {s: {lab for lab, done in judged[s].items() if done >= jobs} for s, jobs in jobs_of.items()}
    return jobs_of, judged, complete


def status(repo: LabellingRepository, labeller: str | None = None) -> dict[str, Any]:
    version = repo.active_pool()
    if version is None:
        return {"pool_version": None, "message": "No labelling pool loaded. Run: python -m app.labelling.load_pool"}
    pool, judgments = repo.pool(version), repo.judgments(version)
    jobs_of, judged, complete = _coverage(pool, judgments)
    overlap = overlap_students(version, jobs_of)
    target = {s: 2 if s in overlap else 1 for s in jobs_of}
    out = {
        "pool_version": version, "rubric_version": RUBRIC_VERSION,
        "students_total": len(jobs_of), "pairs_total": len(pool),
        "students_with_one_complete_judgment": sum(1 for s in jobs_of if complete[s]),
        "students_at_target": sum(1 for s in jobs_of if len(complete[s]) >= target[s]),
        "judgments_total": len(judgments),
        "labellers": sorted({j["labeller"] for j in judgments}),
        "overlap_students": len(overlap),
        "rubric": {"question": QUESTION, "grades": {g: {"name": n, "meaning": m} for g, (n, m) in GRADES.items()},
                   "ignore": IGNORE},
    }
    if labeller:
        me = normalise_labeller(labeller)
        todo = [s for s in jobs_of if me not in complete[s]]
        # most-needed student first; a student this labeller already started comes before a fresh one
        todo.sort(key=lambda s: (-(target[s] - len(complete[s])), -len(judged[s][me]), s))
        out["you"] = {"labeller": me, "judgments": sum(len(v[me]) for v in judged.values()),
                      "students_completed": sum(1 for s in jobs_of if me in complete[s]),
                      "next_student": todo[0] if todo else None}
    return out


def _masked(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return mask(text, detect(text, field="profile")) if text else ""


def task(repo: LabellingRepository, student_id: str, labeller: str) -> dict[str, Any]:
    me = normalise_labeller(labeller)
    version = repo.active_pool()
    rows = [r for r in repo.pool(version or "") if r["student_id"] == student_id]
    if not rows:
        raise LabellingError(f"{student_id} is not in the labelling pool")
    record = repo.student(student_id) or {}
    jobs = repo.jobs([r["job_id"] for r in rows])
    mine = {j["job_id"]: j["relevance_label"] for j in repo.judgments(version)
            if j["student_id"] == student_id and j["labeller"] == me}
    return {
        "pool_version": version, "student_id": student_id, "split": rows[0]["split"],
        # free text is PII-masked (pii-regex-v2); gender, CGPA, source and plans are never shown
        "student": [{"label": label, "text": _masked(record.get(field))} for field, label in
                    _SHOWN_STUDENT_FIELDS.items() if _masked(record.get(field))],
        "pii_detector": DETECTOR_VERSION,
        "jobs": [{"job_id": r["job_id"], "display_order": r["display_order"],
                  **{f: str(jobs.get(r["job_id"], {}).get(f) or "") for f in JOB_FIELDS if f != "job_id"},
                  "your_label": mine.get(r["job_id"])}
                 for r in sorted(rows, key=lambda r: r["display_order"])],
    }


def save(repo: LabellingRepository, student_id: str, job_id: str, labeller: str, label: int) -> dict[str, Any]:
    me = normalise_labeller(labeller)
    if label not in GRADES:
        raise LabellingError("relevance_label must be 0, 1, 2 or 3")
    version = repo.active_pool()
    if not any(r["student_id"] == student_id and r["job_id"] == job_id for r in repo.pool(version or "")):
        raise LabellingError(f"({student_id}, {job_id}) is not in the labelling pool")
    repo.save_judgment(version, student_id, job_id, me, label, RUBRIC_VERSION)
    return {"saved": True, "student_id": student_id, "job_id": job_id, "relevance_label": label, "labeller": me}


def quadratic_kappa(a: list[int], b: list[int]) -> float | None:
    """Cohen's kappa with quadratic weights (ordinal grades 0-3)."""
    if len(a) < 2:
        return None
    from sklearn.metrics import cohen_kappa_score

    if len(set(a) | set(b)) < 2:
        return None
    return float(cohen_kappa_score(a, b, labels=list(GRADES), weights="quadratic"))


def export(repo: LabellingRepository) -> tuple[list[dict], dict]:
    """(rows for the labels CSV, provenance manifest). Only fully judged students are exported."""
    version = repo.active_pool()
    if version is None:
        raise LabellingError("no labelling pool loaded")
    pool, judgments = repo.pool(version), repo.judgments(version)
    jobs_of, _, complete = _coverage(pool, judgments)
    split_of = {r["student_id"]: r["split"] for r in pool}
    grades = defaultdict(dict)  # (student, job) -> labeller -> grade
    for j in judgments:
        grades[(j["student_id"], j["job_id"])][j["labeller"]] = int(j["relevance_label"])
    done = sorted(s for s in jobs_of if complete[s])
    rows, pairs_a, pairs_b, multi = [], [], [], 0
    for s in done:
        for job in sorted(jobs_of[s]):
            by = grades[(s, job)]
            values = [by[k] for k in sorted(by)]
            if len(values) >= 2:
                multi += 1
                pairs_a.append(values[0])
                pairs_b.append(values[1])
            rows.append({"student_id": s, "job_id": job,
                         "relevance_label": math.floor(statistics.median(values)),
                         "label_source": LABEL_SOURCE, "split": split_of[s]})
    exact = sum(x == y for x, y in zip(pairs_a, pairs_b))
    manifest = {
        "label_source": LABEL_SOURCE, "generated_by": "human",
        "method": "graded 0-3 by project team members on the local labelling page (/labelling), blind to which "
                  "model nominated each job and to every model score",
        "models_used_for_labels": [],
        "rubric_version": RUBRIC_VERSION, "pool_version": version,
        "pool": {"note": "models chose which pairs were shown (TREC-style pooling), never the grades; "
                         "see data/benchmark/labelling_pool.manifest.json"},
        "students_exported": len(done), "students_incomplete": len(jobs_of) - len(done),
        "pairs": len(rows), "labellers": len({j["labeller"] for j in judgments}),
        "aggregation": "median of the labellers' grades, ties rounded down",
        "agreement": {"pairs_with_two_or_more_labels": multi,
                      "exact_agreement": round(exact / multi, 3) if multi else None,
                      "quadratic_weighted_kappa": quadratic_kappa(pairs_a, pairs_b)},
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return rows, manifest
