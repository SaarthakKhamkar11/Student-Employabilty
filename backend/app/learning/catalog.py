"""The course catalogue (spec §17, docs/adr/0018): real courses only, every field traced to its source page.

    python -m app.learning.catalog build   fetch each course page in data/learning/curated_courses.csv and extract
                                           its fields -> data/learning/course_catalog.json (committed)
    python -m app.learning.catalog load    upsert data/learning/course_catalog.json into learning_resources

The owner supplied the course names and URLs (data/learning/curated_courses.csv). Every other field comes from
that course's own page, by fixed text patterns, never by a model and never by guesswork:
  title        the page's og:title
  provider     the text after "Offered by"
  difficulty   "<Beginner|Intermediate|Advanced|Mixed> level"
  prerequisites the line right after the level when it states the requirement itself ("No prior experience
               required", "Some related experience required"). A bare "Recommended experience" heading is NOT
               a statement (its detail is not in the page source), so it stays None
  estimated_effort the course-level duration as written: "N weeks|months at H hours a week" (also when the page
               splits it as "N weeks to complete" + "at H hours a week"), else "N weeks|months to complete",
               else "H hours to complete" as one phrase (per-module "8 hours" + "to complete" is not used)
  skills       the "Skills you'll gain" list
  description  the "What you'll learn" bullets, else the og:description
A field a page doesn't state is None: "Not available in source data". The page's SHA-256 and fetch time are kept
with every row (source_reference), so each value can be checked against what the page said.
"""

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import REPO_ROOT

CURATED = REPO_ROOT / "data" / "learning" / "curated_courses.csv"
CATALOG = REPO_ROOT / "data" / "learning" / "course_catalog.json"
USER_AGENT = "Mozilla/5.0 (student-employability catalogue builder; one request per owner-listed course)"
LEVELS = ("Beginner", "Intermediate", "Advanced", "Mixed")
SECTION_ENDS = ("Details to know", "Skills you'll gain", "Tools you'll learn", "Why people choose", "Build your",
                "Earn a career certificate", "Specialization -", "Professional Certificate -", "There are",
                "Course series", "Taught in", "Instructor", "Offered by", "Modules", "About", "Outcomes")


def page_text(raw_html: str) -> list[str]:
    """The visible text of a page as a list of non-empty fragments (tags split fragments)."""
    body = re.sub(r"<script.*?</script>|<style.*?</style>|<svg.*?</svg>", " ", raw_html, flags=re.S)
    parts = (html.unescape(re.sub(r"\s+", " ", p)).strip() for p in re.split(r"<[^>]+>", body))
    return [p for p in parts if p]


def _meta(raw_html: str, prop: str) -> str | None:
    m = re.search(rf'property="og:{prop}" content="([^"]*)"', raw_html)
    return html.unescape(m.group(1)).strip() if m else None


def _after(parts: list[str], label: str) -> list[str]:
    """Fragments after the first fragment equal to (or starting with) `label`, up to the next section heading."""
    for i, p in enumerate(parts):
        if p == label or p.startswith(label):
            out = []
            for q in parts[i + 1:]:
                if any(q.startswith(end) for end in SECTION_ENDS) or q in ("Learn more",):
                    break
                out.append(q)
            return out
    return []


def extract(url: str, raw_html: str) -> dict:
    parts = page_text(raw_html)
    level_i = next((i for i, p in enumerate(parts) if re.fullmatch(rf"({'|'.join(LEVELS)}) level", p)), None)
    difficulty = parts[level_i].replace(" level", "") if level_i is not None else None
    prerequisites = None
    if level_i is not None and level_i + 1 < len(parts):
        nxt = parts[level_i + 1]
        if re.search(r"experience|prerequisite", nxt, re.I) and nxt != "Recommended experience":
            prerequisites = nxt
    text = " | ".join(parts)
    n = r"\d+(?:\.\d+)?"
    effort = None
    for pattern in (rf"{n} (?:weeks?|months?)(?: to complete \| | )at {n} hours? (?:a|per) week",
                    rf"{n} (?:weeks?|months?) to complete", rf"{n} hours? to complete"):
        m = re.search(pattern, text)
        if m:
            effort = m.group(0).replace(" to complete | at ", " at ")
            break
    provider = (_after(parts, "Offered by") or [None])[0]
    skills = list(dict.fromkeys(s for s in _after(parts, "Skills you'll gain") if len(s) <= 60 and s != "Category:"))
    learn = [s for s in _after(parts, "What you'll learn") if len(s) > 20]
    return {
        "url": url, "title": _meta(raw_html, "title"), "provider": provider, "difficulty": difficulty,
        "prerequisites": prerequisites, "estimated_effort": effort,
        "skills": skills, "learning_outcomes": learn,
        "description": " ".join(learn) if learn else _meta(raw_html, "description"),
    }


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def resource_id(url: str) -> str:
    return "coursera:" + url.rstrip("/").rsplit("/", 2)[-2] + "/" + url.rstrip("/").rsplit("/", 1)[-1]


def build() -> dict:
    rows = list(csv.DictReader(open(CURATED, encoding="utf-8")))
    courses, failures = [], []
    for row in rows:
        url = row["url"].strip()
        try:
            raw = fetch(url)
        except Exception as exc:  # noqa: BLE001 - a failed page is reported, never filled in
            failures.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
            continue
        fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
        item = extract(url, raw.decode("utf-8", errors="replace"))
        item.update(resource_id=resource_id(url), title_as_given=row["title_as_given"], supplied_by=row["supplied_by"],
                    supplied_on=row["supplied_on"], page_sha256=hashlib.sha256(raw).hexdigest(), fetched_at=fetched,
                    source_reference=(f"owner-curated list ({row['supplied_on']}) -> {url} fetched {fetched}, "
                                      f"page sha256 {hashlib.sha256(raw).hexdigest()[:12]}"))
        if not item["title"]:
            item["title"] = row["title_as_given"]
            item["title_basis"] = "owner-supplied (the page had no og:title)"
        courses.append(item)
        time.sleep(1.0)  # one polite request per second
    return {"version": "catalog-v1", "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": str(CURATED.relative_to(REPO_ROOT)), "courses": courses, "failures": failures}


def load(dsn: str, catalog: dict) -> int:
    import psycopg

    with psycopg.connect(dsn, connect_timeout=5) as conn, conn.transaction():
        for c in catalog["courses"]:
            conn.execute(
                "INSERT INTO learning_resources (resource_id, title, url, provider, description, skills_covered, "
                "difficulty, prerequisites, estimated_effort, source_reference) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (resource_id) DO UPDATE SET title = EXCLUDED.title, url = EXCLUDED.url, "
                "provider = EXCLUDED.provider, description = EXCLUDED.description, "
                "skills_covered = EXCLUDED.skills_covered, difficulty = EXCLUDED.difficulty, "
                "prerequisites = EXCLUDED.prerequisites, estimated_effort = EXCLUDED.estimated_effort, "
                "source_reference = EXCLUDED.source_reference, updated_at = now()",
                (c["resource_id"], c["title"], c["url"], c["provider"], c["description"], c["skills"],
                 c["difficulty"], c["prerequisites"], c["estimated_effort"], c["source_reference"]))
    return len(catalog["courses"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["build", "load"])
    args = ap.parse_args(argv)
    if args.command == "build":
        catalog = build()
        CATALOG.parent.mkdir(parents=True, exist_ok=True)
        CATALOG.write_text(json.dumps(catalog, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{len(catalog['courses'])} courses extracted, {len(catalog['failures'])} failed -> {CATALOG}")
        for f in catalog["failures"]:
            print("  FAILED", f, file=sys.stderr)
        return
    from app.core.config import get_settings

    settings = get_settings()
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is not set")
    n = load(settings.database_url.get_secret_value(), json.loads(CATALOG.read_text(encoding="utf-8")))
    print(f"upserted {n} courses into learning_resources")


if __name__ == "__main__":
    main()
