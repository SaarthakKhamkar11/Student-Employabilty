"""Job Intelligence components: JD sections, skill phrases, inferred requirements, degree/experience, dedup."""

import pytest

from app.jobs.dedup import find_near_duplicates, match_existing
from app.jobs.requirements import classify_requirements, parse_degree, parse_experience
from app.jobs.skills_split import split_skill_phrases
from src.parsing.resume_text_parser import guess_degree
from src.parsing.section_parser import _clean, _slice_section, parse_job_sections


def test_job_sections_use_line_anchored_headers():
    jd = ("Data Analyst Intern\nAbout the role\nExperience with dashboards is useful.\nResponsibilities:\n- Build "
          "dashboards\nRequirements: Python, SQL\nNice to have\nTableau\nExperience: 0-2 years\nEducation: BCA")
    s = parse_job_sections(jd)
    assert s["title"] == "Data Analyst Intern" and s["required"] == "Python, SQL" and s["preferred"] == "Tableau"
    assert s["experience"] == "0-2 years" and "Experience with dashboards" in s["description"]


def test_resume_slicing_is_unchanged_by_the_extension():
    text = _clean("Education Details\nB.E\nSkills\nPython\nExperience\nX")
    assert _slice_section(text.lower(), text, "skills") == "Python"


@pytest.mark.parametrize("text, phrases", [
    ("Event planning Conference logistics Budget management",
     ["Event planning", "Conference logistics", "Budget management"]),
    ("Social media platforms (e.g., Facebook, Twitter) Content creation",
     ["Social media platforms (e.g., Facebook, Twitter)", "Content creation"]),
    ("Proficiency in Adobe Creative Suite Typography", ["Proficiency in Adobe Creative Suite Typography"]),
    ("Problem-solving Communication", ["Problem-solving", "Communication"]),
    ("Python, SQL\nPower BI", ["Python", "SQL", "Power BI"]),
])
def test_skill_phrases_are_split(text, phrases):
    assert split_skill_phrases(text) == phrases


def _skills(*raws, origin=None):
    return [{"raw": r, "canonical": r, "category": "unclassified", "method": "unmapped", "evidence": None,
             "origin": origin} for r in raws]


def test_requirement_rules_and_their_recorded_basis():
    out = classify_requirements(_skills("Event planning", "Vendor coordination", "Social media"),
                                "Handle event planning and vendor coordination.", "Plan events.")
    assert [(s["requirement"], s["requirement_basis"]) for s in out] == [
        ("required", "named_in_role_text"), ("required", "named_in_role_text"),
        ("preferred", "not_named_in_role_text")]
    explicit = classify_requirements(_skills("Python", origin="required") + _skills("Tableau", origin="preferred"),
                                     "", "")
    assert [s["requirement_basis"] for s in explicit] == ["explicit_required_section", "explicit_preferred_section"]
    language = classify_requirements(_skills("Tableau"), "Tableau experience is a plus.", "")
    assert language[0]["requirement_basis"] == "explicit_preferred_language"


def test_related_word_forms_count_as_named_v2():
    # v1 missed these: "planning"/"plan" and "coordination"/"coordinate" (seen on real postings, ADR 0008)
    out = classify_requirements(_skills("Event planning", "Vendor coordination"),
                                "Plan and execute events.", "Coordinate with each vendor.")
    assert [s["requirement_basis"] for s in out] == ["named_in_role_text", "named_in_role_text"]


def test_unrelated_words_with_a_short_common_start_do_not_match():
    out = classify_requirements(_skills("Origami", "Communication skills"), "Serve the community.", "")
    assert out[1]["requirement_basis"] == "not_named_in_role_text"  # "community" is not "communication"


def test_generic_words_are_not_requirement_cues():
    # These produced only false positives on jobs.csv and were removed from the cue lists.
    for sentence in ("Provide essential healthcare services.", "Infrastructure required for web applications.",
                     "Recommend ways to gain a competitive advantage."):
        (skill,) = classify_requirements(_skills("Healthcare services"), sentence, "")
        assert not skill["requirement_basis"].startswith("explicit")


def test_a_posting_always_has_at_least_one_required_skill():
    out = classify_requirements(_skills("Origami", "Juggling"), "Manage accounts.", "")
    assert out[0]["requirement_basis"] == "first_listed_fallback" and out[1]["requirement"] == "preferred"


@pytest.mark.parametrize("text, degree", [
    ("B.Tech", ("B.TECH", 2)), ("MBA", ("MBA", 3)), ("PhD", ("PHD", 4)), ("B.Com", ("B.COM", 2)),
    ("M.Com", ("M.COM", 3)), ("BBA", ("BBA", 2)), ("BA", ("BA", 2)), ("Diploma in IT", ("DIPLOMA", 1)),
    ("candidates should be proficient", (None, None)),
])
def test_degrees_reuse_guess_degree_then_job_extras(text, degree):
    assert parse_degree(text) == degree


def test_guess_degree_no_longer_reads_the_word_be_as_a_degree():
    assert guess_degree("should be proficient") == "" and guess_degree("Must be an MBA") == "MBA"
    assert guess_degree("BE Civil") == "BE" and guess_degree("b.e mechanical") == "B.E"


@pytest.mark.parametrize("text, years", [("2 to 8 Years", (2.0, 8.0)), ("0-2 years", (0.0, 2.0)),
                                         ("3+ years", (3.0, None)), ("fresher", (None, None))])
def test_experience_is_parsed_to_numbers(text, years):
    assert parse_experience(text) == years


def _job(job_id, company, title, text):
    return {"job_id": job_id, "company_name": company, "job_title": title, "job_description": text,
            "responsibilities": "", "skills": ""}


def test_near_duplicates_need_same_company_title_and_text():
    text = "Analyse sales data and build weekly dashboards for regional managers"
    jobs = [_job("J1", "Acme", "Analyst", text), _job("J2", " ACME ", "analyst", text),
            _job("J3", "Other Co", "Analyst", text), _job("J4", "Acme", "Analyst", "Completely different work here")]
    assert find_near_duplicates(jobs) == {"J2": {"near_duplicate_of": "J1", "similarity": 1.0}}
    assert match_existing(_job(None, "Acme", "Analyst", text), jobs)["near_duplicate_of"] == "J1"
    assert match_existing(_job(None, "Acme", "Analyst", "Brand new role"), jobs) is None
