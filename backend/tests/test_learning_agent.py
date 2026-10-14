"""Phase 10: the Learning Agent never invents a resource (docs/adr/0018)."""

import asyncio

import pytest

from app.agents import build_agents
from app.core.config import Settings
from app.learning import catalog, ranking as R
from app.learning.repository import InMemoryLearningRepository
from app.learning.youtube import YouTubeUnavailable, iso_duration
from app.schemas.agent_io import LearningAgentInput, LearningAgentOutput, SkillGap, SkillGapReport
from app.schemas.agent_state import AgentState
from app.tools import learning_tools

COURSES = [
    {"resource_id": "coursera:learn/sql", "title": "SQL for Data Science", "url": "https://example.org/sql",
     "provider": "UC Davis", "description": "Write SQL queries to filter, sort and summarize data.",
     "skills": ["SQL", "Data Analysis"], "difficulty": "Beginner", "prerequisites": "No prior experience required",
     "estimated_effort": "2 weeks at 10 hours a week", "source_reference": "test catalogue row 1"},
    {"resource_id": "coursera:learn/python", "title": "Programming in Python", "url": "https://example.org/python",
     "provider": "Meta", "description": "Python syntax, objects, classes and unit testing.",
     "skills": ["Python", "Unit Testing"], "difficulty": None, "prerequisites": None, "estimated_effort": None,
     "source_reference": "test catalogue row 2"},
]
ABSENT = "Underwater basket weaving"


def video(vid, title, desc):
    return {"resource_id": f"youtube:{vid}", "title": title, "url": f"https://www.youtube.com/watch?v={vid}",
            "provider": "YouTube: Test", "description": desc, "skills": [], "difficulty": None, "prerequisites": None,
            "estimated_effort": "video length 12 min", "source_reference": f"test API videos.list id={vid}", "kind": "video"}


class FakeYouTube:
    def __init__(self, results=None, fail=False):
        self.results, self.fail, self.queries = results or {}, fail, []

    def __call__(self, query, key, max_results):
        self.queries.append(query)
        if self.fail:
            raise YouTubeUnavailable("YouTube API search HTTP 403: quota exceeded")
        return self.results.get(query, []), {"endpoint": "search.list", "q": query}


@pytest.fixture(autouse=True)
def cfg(monkeypatch):
    conf = {"ranking": {"top_k": 3, "min_relevance": {"tfidf_baseline": 0.1}},
            "youtube": {"enabled": True, "query": "{skill} tutorial", "max_results": 5, "cache_days": 7,
                        "max_searches_per_request": 10}}
    monkeypatch.setattr(learning_tools, "learning_config", lambda: conf)
    return conf


def report(required, preferred=(), importance=None):
    imp = importance or {}
    gap = lambda s: SkillGap(skill=s, importance_score=imp.get(s, 0.1), transferable_candidate=None,  # noqa: E731
                             transferable_similarity=None)
    return SkillGapReport(matched_skills=["Excel"], missing_required=[gap(s) for s in required],
                          missing_preferred=[gap(s) for s in preferred], partially_related=[],
                          importance_model="weighted_baseline_v1+test")


def run(embedder, rep, youtube=None, repo=None):
    repo = repo or InMemoryLearningRepository(COURSES)
    settings = Settings(_env_file=None, environment="test", tool_retry_backoff_seconds=0.0, youtube_api_key="k")
    agents = build_agents(settings=settings, profile_repo=None, job_repo=None, embedder=embedder, matching_repo=None,
                          learning_repo=repo, youtube_search=youtube or FakeYouTube())
    state = AgentState(request_id="req-p10", task_type="learning_recommendations")
    inp = LearningAgentInput(student_id="S1", job_id="J1", skill_gaps=[g.skill for g in rep.missing_required] or ["x"],
                             skill_gap_report=rep, job_title="Data Analyst")
    out = asyncio.run(agents["learning_agent"].run(state, inp.model_dump()))
    return out, repo


# -- the hard rule ----------------------------------------------------------------------------------------------
def test_a_skill_absent_from_the_catalog_gets_the_explicit_no_resources_message(embedder):
    out, repo = run(embedder, report([ABSENT, "SQL"]))
    assert out.status == "success", out.errors
    recs = {r.skill: r for r in LearningAgentOutput.model_validate(out.data).recommendations}
    assert recs[ABSENT].resources == []
    assert recs[ABSENT].message == "No resources found for this skill in the current catalog"
    assert "No resources found" in recs[ABSENT].explanation
    assert recs["SQL"].resources[0].resource_url == "https://example.org/sql"
    assert any(row["skill"] == ABSENT and row["resource_id"] is None for row in repo.saved)


def test_every_recommended_resource_traces_to_a_catalogue_row_or_an_api_result(embedder):
    yt = FakeYouTube({"Python tutorial": [video("abc123", "Python for beginners", "Learn Python programming")]})
    out, _ = run(embedder, report(["Python", "SQL", ABSENT]), youtube=yt)
    known = {c["url"]: c["resource_id"] for c in COURSES} | {"https://www.youtube.com/watch?v=abc123": "youtube:abc123"}
    for rec in LearningAgentOutput.model_validate(out.data).recommendations:
        for res in rec.resources:
            assert res.resource_url in known
            assert res.source.startswith(known[res.resource_url] + " | ")


def test_resources_are_ranked_by_computed_relevance(embedder):
    yt = FakeYouTube({"Python tutorial": [video("v1", "Python programming full course", "Python programming")]})
    out, _ = run(embedder, report(["Python"]), youtube=yt)
    rec = LearningAgentOutput.model_validate(out.data).recommendations[0]
    scores = [r.relevance_score for r in rec.resources]
    assert scores == sorted(scores, reverse=True) and all(s >= 0.1 for s in scores)


def test_missing_optional_fields_say_so(embedder):
    out, _ = run(embedder, report(["Python"]))
    py = next(r for r in LearningAgentOutput.model_validate(out.data).recommendations[0].resources
              if r.resource_url == "https://example.org/python")
    assert (py.difficulty, py.prerequisites, py.estimated_effort) == ("Not available in source data",) * 3


def test_youtube_failure_falls_back_to_the_catalogue_and_says_why(embedder):
    out, _ = run(embedder, report(["SQL"]), youtube=FakeYouTube(fail=True))
    data = LearningAgentOutput.model_validate(out.data)
    assert "quota exceeded" in data.sources["youtube"]["SQL"]
    assert data.recommendations[0].resources[0].kind == "course"


def test_cached_searches_spend_no_quota(embedder):
    yt = FakeYouTube()
    repo = InMemoryLearningRepository(COURSES)
    run(embedder, report(["SQL"]), youtube=yt, repo=repo)
    run(embedder, report(["SQL"]), youtube=yt, repo=repo)
    assert yt.queries == ["SQL tutorial"]


def test_quota_guard_caps_searches_per_request(embedder, cfg):
    cfg["youtube"]["max_searches_per_request"] = 1
    yt = FakeYouTube()
    out, _ = run(embedder, report(["SQL", "Python"]), youtube=yt)
    assert len(yt.queries) == 1
    assert any("quota guard" in s for s in LearningAgentOutput.model_validate(out.data).sources["youtube"].values())


# -- prioritisation ----------------------------------------------------------------------------------------------
def test_gaps_follow_importance_and_ties_are_reported_not_ranked():
    rep = report(["A", "B", "C"], ["D"], importance={"A": 0.09, "B": 0.09, "C": 0.2, "D": 0.03}).model_dump()
    gaps = R.prioritise(rep)
    assert [(g["skill"], g["priority"]) for g in gaps] == [("C", 1), ("A", 2), ("B", 2), ("D", 3)]
    assert next(g for g in gaps if g["skill"] == "A")["tied_with"] == ["B"]


def test_no_report_means_no_recommendations(embedder):
    settings = Settings(_env_file=None, environment="test", tool_retry_backoff_seconds=0.0)
    agents = build_agents(settings=settings, profile_repo=None, job_repo=None, embedder=embedder, matching_repo=None,
                          learning_repo=InMemoryLearningRepository(COURSES), youtube_search=FakeYouTube())
    state = AgentState(request_id="req-p10b", task_type="learning_recommendations")
    out = asyncio.run(agents["learning_agent"].run(state, {"student_id": "S1", "job_id": "J1", "skill_gaps": ["SQL"]}))
    assert LearningAgentOutput.model_validate(out.data).recommendations == "Not yet evaluated"


# -- the catalogue builder and the API helpers -------------------------------------------------------------------
PAGE = """<html><head><meta property="og:title" content="SQL for Data Science"></head><body>
<span>Beginner level</span><span>No prior experience required</span><span>2 weeks at 10 hours a week</span>
<h2>What you'll learn</h2><li>Write SQL queries to filter, sort, and summarize data.</li>
<h2>Skills you'll gain</h2><span>SQL</span><span>Data Analysis</span><h2>Details to know</h2>
<span>Offered by</span><span>University of California, Davis</span><span>Learn more</span></body></html>"""


def test_catalogue_fields_come_only_from_the_page():
    row = catalog.extract("https://www.coursera.org/learn/sql-for-data-science", PAGE)
    assert row["title"] == "SQL for Data Science" and row["provider"] == "University of California, Davis"
    assert (row["difficulty"], row["prerequisites"], row["estimated_effort"]) == (
        "Beginner", "No prior experience required", "2 weeks at 10 hours a week")
    assert row["skills"] == ["SQL", "Data Analysis"]


def test_a_page_that_states_nothing_gives_nothing():
    row = catalog.extract("https://example.org/x", "<html><body><p>Hello</p></body></html>")
    assert all(row[k] is None for k in ("title", "provider", "difficulty", "prerequisites", "estimated_effort"))
    assert row["skills"] == []


def test_a_bare_recommended_experience_heading_is_not_a_prerequisite():
    page = PAGE.replace("No prior experience required", "Recommended experience")
    assert catalog.extract("https://example.org/x", page)["prerequisites"] is None


def test_iso_durations_are_reformatted_not_estimated():
    assert iso_duration("PT2H24M25S") == "2 h 24 min 25 s" and iso_duration("PT56M") == "56 min"
    assert iso_duration(None) is None and iso_duration("garbage") is None
