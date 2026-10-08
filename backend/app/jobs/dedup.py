"""Near-duplicate job detection at ingest time (docs/adr/0010).

Two postings are near-duplicates when ALL of these hold:
- the same company (case- and space-normalised),
- the same job title (case- and space-normalised), and
- a Jaccard similarity >= 0.9 between their word 3-shingles of
  description + responsibilities + skills.

Company and title are required, not just similar text, because the job
dataset is templated: its 4,990 postings have only 376 distinct
descriptions, one per role. Text similarity alone would call every
same-role posting a duplicate.

The canonical posting of a group is the one with the smallest job_id;
the others get near_duplicate_of = that job_id.
"""

import re
from collections import defaultdict
from typing import Any

SIMILARITY_THRESHOLD = 0.9
DETECTOR_VERSION = "job-dedup-v1"


def norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def shingles(text: str, k: int = 3) -> set[str]:
    words = re.findall(r"[a-z0-9+#.]+", text.lower())
    return {" ".join(words[i:i + k]) for i in range(max(len(words) - k + 1, 1))} if words else set()


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def content(job: dict[str, Any]) -> str:
    return " ".join(str(job.get(f) or "") for f in ("job_description", "responsibilities", "skills"))


def find_near_duplicates(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """{job_id: {"near_duplicate_of": canonical_id, "similarity": s}} for every non-canonical duplicate."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        if norm(job.get("company_name")) and norm(job.get("job_title")):
            groups[(norm(job["company_name"]), norm(job["job_title"]))].append(job)
    flagged: dict[str, dict[str, Any]] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda j: j["job_id"])
        cached = {j["job_id"]: shingles(content(j)) for j in members}
        for i, job in enumerate(members):
            if job["job_id"] in flagged:
                continue
            for other in members[i + 1:]:
                if other["job_id"] in flagged:
                    continue
                sim = jaccard(cached[job["job_id"]], cached[other["job_id"]])
                if sim >= SIMILARITY_THRESHOLD:
                    flagged[other["job_id"]] = {"near_duplicate_of": job["job_id"], "similarity": round(sim, 4)}
    return flagged


def match_existing(candidate: dict[str, Any], existing: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return {"near_duplicate_of", "similarity"} for the earliest existing near-duplicate of `candidate`."""
    if not (norm(candidate.get("company_name")) and norm(candidate.get("job_title"))):
        return None
    mine = shingles(content(candidate))
    for job in sorted(existing, key=lambda j: j["job_id"]):
        if job["job_id"] == candidate.get("job_id"):
            continue
        if (norm(job.get("company_name")), norm(job.get("job_title"))) != (norm(candidate.get("company_name")),
                                                                           norm(candidate.get("job_title"))):
            continue
        sim = jaccard(mine, shingles(content(job)))
        if sim >= SIMILARITY_THRESHOLD and (candidate.get("job_id") is None or job["job_id"] < candidate["job_id"]):
            return {"near_duplicate_of": job["job_id"], "similarity": round(sim, 4)}
    return None
