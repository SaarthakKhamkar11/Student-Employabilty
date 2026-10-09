"""A small in-memory world for Matching Agent tests: real TF-IDF vectors, no database.

Profiles and postings use the same structure the Profile and Job Intelligence agents store, and the
vector index holds the same sections those agents embed.
"""

from dataclasses import replace

from app.agents.job_intelligence_agent import job_section_texts
from app.agents.profile_agent import section_texts
from app.db.matching_repository import InMemoryMatchingRepository
from app.matching.calibration import Calibration, build_calibration
from app.matching.config import load_matching_config
from app.matching.entities import job_view, student_view


def skill(name: str, evidence: str | None = None) -> dict:
    return {"raw": name, "canonical": name, "category": "technical", "method": "exact", "evidence": evidence}


def profile(sid: str, skills, *, degree="B.E", branch="Computer Engineering", domain="Data Analyst and Machine Learning",
            experience=None, projects=None, certifications=None, interests="Placements") -> dict:
    return {"student_id": sid, "source": "stored_record", "degree": degree, "branch": branch, "graduation_year": 2027,
            "avg_cgpa_sem5": 8.0, "education": f"{degree} {branch}", "skills": [skill(*s) if isinstance(s, tuple)
                                                                                 else skill(s) for s in skills],
            "projects": projects or "Not detected", "experience": experience or "Not detected",
            "certifications": certifications or "Not detected", "achievements": "Not detected",
            "career_interests": interests, "domain": domain, "pii": "Not yet evaluated", "missing_fields": [],
            "extractor_version": "test"}


def posting(jid: str, title: str, role: str, required, preferred=(), *, level=2, degree="B.TECH", min_years=0.0,
            country="India", visible="both", description="", duties=()) -> dict:
    skills = [{**skill(s), "requirement": "required", "requirement_basis": "test"} for s in required]
    skills += [{**skill(s), "requirement": "preferred", "requirement_basis": "test"} for s in preferred]
    return {"job_id": jid, "source": "stored_record", "job_title": title, "role": role, "company_name": "Example Co",
            "location": "City", "country": country, "work_type": "Full-Time", "salary_range": "$1",
            "job_portal": "Test", "job_description": description or f"{role} work.",
            "qualification_required": degree, "experience_required": f"{min_years} to 9 Years",
            "responsibilities": list(duties) or [f"Do {role} tasks."], "min_degree": degree, "min_degree_level": level,
            "min_experience_years": min_years, "max_experience_years": 9.0, "skills": skills,
            "required_skills": list(required), "preferred_skills": list(preferred),
            "skill_split": "Not yet evaluated", "visibility": {"gender_preference": visible},
            "duplicate_check": "Not yet evaluated", "pii": "Not yet evaluated", "missing_fields": [],
            "extractor_version": "test"}


STUDENTS = {
    "S1": (profile("S1", [("Python", "18 months"), "SQL", "Power BI", "Machine learning"],
                   experience=["Data Analyst Intern at Example Analytics, 6 months"],
                   projects=["Sales dashboard in Power BI", "Student placement predictor using machine learning"],
                   certifications=["Google Data Analytics Certificate"]), {"country": "India"}, "female"),
    "S2": (profile("S2", ["AutoCAD", "SolidWorks", "Thermodynamics"], branch="Mechanical Engineering",
                   domain="Core Mechanical", projects=["Gearbox design in SolidWorks"]), {"country": None}, None),
    "S3": (profile("S3", ["Java", "Spring Boot", "SQL"], degree="DIPLOMA", domain="Software Developer Engineer (SDE)"),
           {"country": "India"}, "male"),
}
JOBS = {
    "J1": posting("J1", "Data Analyst", "Data Analyst", ["Python", "SQL"], ["Tableau", "Power BI"],
                  description="Analyse data with Python and SQL; build Power BI dashboards.",
                  duties=["Build dashboards in Power BI.", "Write SQL queries."]),
    "J2": posting("J2", "Data Scientist", "Machine Learning Engineer", ["Python", "Deep learning"], ["SQL"],
                  level=3, degree="M.TECH", description="Build machine learning models in Python."),
    "J3": posting("J3", "Mechanical Engineer", "Mechanical Design Engineer", ["AutoCAD", "SolidWorks"],
                  country="Germany", description="Design mechanical parts in SolidWorks."),
    "J4": posting("J4", "Data Analyst", "Business Intelligence Analyst", ["SQL", "Power BI"], ["Excel"],
                  min_years=3.0, description="Business intelligence reporting with SQL and Power BI."),
    "J5": posting("J5", "Software Engineer", "Backend Developer", ["Java", "Spring Boot"], ["Docker"],
                  visible="male", description="Backend services in Java with Spring Boot."),
    "J6": posting("J6", "Data Analyst", "Data Quality Analyst", ["SQL", "Data cleaning"], [],
                  country="Germany", description="Check data quality with SQL."),
}


def build_world(embedder, *, registered: bool = True):
    repo = InMemoryMatchingRepository()
    version, _ = embedder.describe()
    for sid, (prof, record, gender) in STUDENTS.items():
        view = student_view(sid, prof, record)
        repo.add_student(view, gender)
        texts = section_texts(prof)
        for section, vec in zip(texts, embedder.embed(list(texts.values()))):
            if vec is not None:
                repo.index.add("student", sid, section, vec, embedder.model_name, version)
    for jid, post in JOBS.items():
        repo.add_job(job_view(jid, post, {"country": post["country"], "near_duplicate_of": None,
                                          "source_dataset": "kaggle_job_descriptions"}))
        texts = job_section_texts(post)
        for section, vec in zip(texts, embedder.embed(list(texts.values()))):
            if vec is not None:
                repo.index.add("job", jid, section, vec, embedder.model_name, version)
    cfg = load_matching_config()
    cfg = replace(cfg, registration={"experiment_id": "test-experiment", "config_sha256": cfg.sha256}
                  if registered else {})
    data = build_calibration([s for s, _ in repo.students()], repo.jobs(), embedder, cfg.features["transferable"],
                             [n for s, _ in repo.students() for n in s.skills] + [n for j in repo.jobs() for n, _ in j.skills])
    calibration = Calibration(model_name=data["model_name"], model_version=data["model_version"],
                              quantiles={k: tuple(v) for k, v in data["quantiles"].items()},
                              transferable_threshold=data["transferable"]["threshold"],
                              reference_pairs=data["reference_pairs"])
    return repo, cfg, calibration
