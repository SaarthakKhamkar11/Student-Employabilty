"""Build backend/app/profile/data/skill_aliases.csv, the auditable skill-alias table.

The canonical skills and their aliases below were chosen by inspecting the real
skill text in data/processed/students_unified.csv (student skill lists) and
data/processed/jobs.csv (job skill phrases), plus common abbreviations. For
every row the script counts how often the alias actually occurs in each
dataset and writes that into the `evidence` column. A row with no occurrences
in either dataset is a common abbreviation, labelled as such.

Matching is case-insensitive (see skill_key), so case variants such as
JAVA/Java/java need no rows of their own.

To extend: add an alias to ALIASES and re-run (from the repo root)
    PYTHONPATH=backend:. .venv/bin/python -m app.profile.build_skill_aliases
"""

import csv
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from app.profile.skills import ALIAS_TABLE_PATH, clean_skill_item, skill_key

REPO_ROOT = Path(__file__).resolve().parents[3]

# canonical -> (category, [aliases]). "technical" = tools, methods and domain skills;
# "soft" = interpersonal and professional skills.
ALIASES: dict[str, tuple[str, list[str]]] = {
    # programming languages
    "Python": ("technical", ["python3", "python 3"]),
    "Java": ("technical", ["core java", "java se"]),
    "C": ("technical", ["c language", "c programming"]),
    "C++": ("technical", ["cpp", "c plus plus"]),
    "C#": ("technical", ["c sharp", "csharp"]),
    "JavaScript": ("technical", ["js", "java script", "es6", "es6+"]),
    "TypeScript": ("technical", ["ts"]),
    "SQL": ("technical", ["structured query language"]),
    "R": ("technical", ["r programming", "r language"]),
    "PHP": ("technical", []),
    "Go": ("technical", ["golang"]),
    "Bash": ("technical", ["shell scripting", "shell script", "unix shell"]),
    "MATLAB": ("technical", []),
    "HTML": ("technical", ["html5"]),
    "CSS": ("technical", ["css3"]),
    # web frameworks and stacks
    "React": ("technical", ["react.js", "reactjs", "react js"]),
    "Node.js": ("technical", ["node", "nodejs", "node js"]),
    "Express.js": ("technical", ["express", "expressjs", "express js"]),
    "Next.js": ("technical", ["nextjs", "next js"]),
    "Angular": ("technical", ["angularjs", "angular.js", "angular js"]),
    "Vue.js": ("technical", ["vue", "vuejs", "vue js"]),
    "Django": ("technical", []),
    "Flask": ("technical", []),
    "FastAPI": ("technical", ["fast api"]),
    "Spring Boot": ("technical", ["springboot", "spring-boot"]),
    "Spring": ("technical", ["spring framework", "spring mvc"]),
    "Hibernate": ("technical", []),
    "jQuery": ("technical", []),
    "Bootstrap": ("technical", []),
    "Tailwind CSS": ("technical", ["tailwind", "tailwindcss"]),
    "ASP.NET": ("technical", ["asp .net", "asp.net mvc"]),
    ".NET": ("technical", ["dot net", "dotnet", ".net framework"]),
    "Java EE": ("technical", ["j2ee", "jee"]),
    "Servlets": ("technical", ["servlet"]),
    "JSP": ("technical", []),
    "AJAX": ("technical", []),
    "REST APIs": ("technical", ["rest api", "rest", "restful api", "restful apis", "restful services"]),
    "MERN Stack": ("technical", ["mern"]),
    "Web Development": ("technical", ["web designing", "web design"]),
    # data and ML libraries
    "Pandas": ("technical", []),
    "NumPy": ("technical", []),
    "SciPy": ("technical", []),
    "Matplotlib": ("technical", []),
    "Seaborn": ("technical", []),
    "Scikit-learn": ("technical", ["sklearn", "scikit learn", "scikitlearn"]),
    "TensorFlow": ("technical", []),
    "Keras": ("technical", []),
    "PyTorch": ("technical", ["torch"]),
    "OpenCV": ("technical", ["open cv"]),
    "LangChain": ("technical", []),
    "LangGraph": ("technical", []),
    "Hugging Face": ("technical", ["huggingface"]),
    "Transformers": ("technical", []),
    "Statsmodels": ("technical", []),
    # data / AI concepts
    "Machine Learning": ("technical", ["ml"]),
    "Deep Learning": ("technical", ["dl"]),
    "Natural Language Processing": ("technical", ["nlp"]),
    "Computer Vision": ("technical", []),  # "CV" is deliberately not an alias: it also means curriculum vitae
    "Data Analysis": ("technical", ["data analytics"]),
    "Data Visualization": ("technical", ["data visualisation"]),
    "Data Cleaning & Preprocessing": ("technical", ["data cleaning", "data preprocessing",
                                                    "data cleaning and preprocessing"]),
    "Feature Engineering": ("technical", []),
    "Statistics": ("technical", ["statistics & probability basics", "statistics and probability",
                                 "statistics & probability", "statistical analysis"]),
    "Generative AI": ("technical", ["genai", "gen ai"]),
    "Large Language Models": ("technical", ["llm", "llms"]),
    "Retrieval-Augmented Generation": ("technical", ["rag", "rag pipelines"]),
    "Agentic AI": ("technical", []),
    "Exploratory Data Analysis": ("technical", ["eda"]),
    "Artificial Intelligence": ("technical", ["ai"]),
    "Convolutional Neural Networks": ("technical", ["cnn", "cnns"]),
    "Multi-Agent Systems": ("technical", []),
    "Model Context Protocol": ("technical", ["mcp"]),
    "Data Engineering": ("technical", []),
    "Data Modeling": ("technical", ["data modelling"]),
    "Time Series Analysis": ("technical", ["time series", "time series forecasting"]),
    "Model Evaluation": ("technical", []),
    "Predictive Modeling": ("technical", ["predictive modelling"]),
    "Big Data": ("technical", []),
    # computer-science fundamentals
    "Data Structures & Algorithms": ("technical", ["dsa", "data structures and algorithms", "data structures",
                                                   "data structure"]),
    "Object-Oriented Programming": ("technical", ["oop", "oops", "object oriented programming"]),
    "Database Management Systems": ("technical", ["dbms", "database management system", "database management"]),
    "Databases": ("technical", ["database"]),
    "Operating Systems": ("technical", []),
    "Computer Networks": ("technical", ["computer networking", "networking"]),
    # tools, platforms, databases
    "Git": ("technical", []),
    "GitHub": ("technical", []),
    "Docker": ("technical", []),
    "Kubernetes": ("technical", ["k8s"]),
    "AWS": ("technical", ["amazon web services"]),
    "Microsoft Azure": ("technical", ["azure"]),
    "Google Cloud Platform": ("technical", ["gcp", "google cloud"]),
    "Linux": ("technical", []),
    "Power BI": ("technical", ["powerbi", "power-bi", "ms power bi"]),
    "Tableau": ("technical", []),
    "Excel": ("technical", ["ms excel", "microsoft excel", "advanced excel", "vlookup"]),
    "MS Office": ("technical", ["microsoft office", "ms-office"]),
    "Jira": ("technical", []),
    "Figma": ("technical", []),
    "AutoCAD": ("technical", ["auto cad"]),
    "CATIA": ("technical", []),
    "STAAD Pro": ("technical", ["staad.pro", "staad"]),
    "Revit": ("technical", []),
    "Photoshop": ("technical", ["adobe photoshop"]),
    "SAP": ("technical", []),
    "SAP BusinessObjects": ("technical", ["sap bo"]),
    "Salesforce": ("technical", []),
    "Hadoop": ("technical", []),
    "Apache Spark": ("technical", ["spark"]),
    "PySpark": ("technical", []),
    "Kafka": ("technical", ["apache kafka"]),
    "Airflow": ("technical", ["apache airflow"]),
    "Selenium": ("technical", []),
    "MySQL": ("technical", ["my sql"]),
    "PostgreSQL": ("technical", ["postgres", "postgre sql"]),
    "MongoDB": ("technical", ["mongo", "mongo db"]),
    "Oracle": ("technical", ["oracle database", "oracle db"]),
    "Microsoft SQL Server": ("technical", ["ms sql server", "sql server", "mssql"]),
    "Firebase": ("technical", []),
    "Redis": ("technical", []),
    "Vercel": ("technical", []),
    "CI/CD": ("technical", ["ci-cd", "ci cd", "cicd"]),
    "DevOps": ("technical", []),
    "ETL": ("technical", []),
    "Enterprise Resource Planning": ("technical", ["erp"]),
    "UI/UX Design": ("technical", ["ui/ux", "ux/ui", "ui ux", "ui-ux"]),
    "SEO": ("technical", ["search engine optimization"]),
    "Software Testing": ("technical", ["testing"]),
    "Manual Testing": ("technical", []),
    "Automation Testing": ("technical", ["test automation"]),
    "Drafting": ("technical", []),
    # further items seen at least twice in students_unified.csv (Phase 4 coverage review)
    "Backend Development": ("technical", ["backend"]),
    "APIs": ("technical", ["api", "api integration"]),
    "Version Control": ("technical", []),
    "NoSQL": ("technical", []),
    "Supabase": ("technical", []),
    "VS Code": ("technical", ["vscode", "visual studio code"]),
    "Google Sheets": ("technical", []),
    "Streamlit": ("technical", []),
    "WebSockets": ("technical", ["websocket"]),
    "Socket.IO": ("technical", []),
    "Postman": ("technical", []),
    "Ollama": ("technical", []),
    "YOLO": ("technical", []),
    "CrewAI": ("technical", []),
    "Neo4j": ("technical", []),
    "Rust": ("technical", []),
    "Dart": ("technical", []),
    "Flutter": ("technical", []),
    "Android": ("technical", ["android development"]),
    "XML": ("technical", []),
    "Blockchain": ("technical", []),
    "Agile": ("technical", ["agile methodology", "scrum"]),
    "Eclipse": ("technical", []),
    "Crystal Reports": ("technical", ["crystal report"]),
    "SAP ABAP": ("technical", []),
    "SharePoint": ("technical", ["microsoft sharepoint"]),
    "Informatica": ("technical", []),
    "Entity Framework": ("technical", []),
    "Apache Hive": ("technical", ["hive"]),
    "Sqoop": ("technical", ["apache hadoop sqoop"]),
    "Network Security": ("technical", []),
    "Cisco": ("technical", []),
    "VPN": ("technical", []),
    "LAN": ("technical", []),
    "RMAN": ("technical", []),
    "Functional Testing": ("technical", []),
    "Sales": ("technical", []),
    "Marketing": ("technical", []),
    "Operations Management": ("technical", ["operations"]),
    # soft and professional skills
    "Communication": ("soft", ["communication skills"]),
    "Teamwork": ("soft", ["team work", "team player"]),
    "Leadership": ("soft", []),
    "Problem Solving": ("soft", ["problem-solving"]),
    "Time Management": ("soft", []),
    "Attention to Detail": ("soft", []),
    "Critical Thinking": ("soft", []),
    "Project Management": ("soft", []),
    "Presentation Skills": ("soft", ["presentation"]),
    "Documentation": ("soft", []),
}


def student_skill_items(students_csv: Path) -> Counter:
    """Skill items as they appear in student data: Kaggle rows are '; '-joined, placement rows comma/newline lists."""
    df = pd.read_csv(students_csv, dtype=str, keep_default_na=False)
    items: Counter = Counter()
    for value in df["skills_raw"]:
        for line in re.split(r"[\n;]", value):
            line = re.sub(r"^[A-Za-z /&]+:\s*", "", line)  # category prefixes such as "Programming Languages:"
            for item in line.split(","):
                for piece in clean_skill_item(item)[0]:
                    items[skill_key(piece)] += 1
    return items


def job_phrase_counts(jobs_csv: Path, keys: set[str]) -> Counter:
    text = " " + " ".join(pd.read_csv(jobs_csv, dtype=str, keep_default_na=False)["skills"]).lower() + " "
    return Counter({k: len(re.findall(r"(?<![\w.+#])" + re.escape(k) + r"(?![\w+#])", text)) for k in keys})


def build(students_csv: Path, jobs_csv: Path, out_path: Path = ALIAS_TABLE_PATH) -> int:
    rows = [(canonical, alias, category)
            for canonical, (category, aliases) in ALIASES.items()
            for alias in [canonical, *aliases]]
    keys = {skill_key(alias) for _, alias, _ in rows}
    if len(keys) != len(rows):
        raise ValueError("an alias is listed twice (case-insensitively)")
    students, jobs = student_skill_items(students_csv), job_phrase_counts(jobs_csv, keys)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["alias", "canonical", "category", "evidence"])
        for canonical, alias, category in rows:
            k = skill_key(alias)
            found = [f"students_unified.csv:{students[k]}" if students[k] else "",
                     f"jobs.csv:{jobs[k]}" if jobs[k] else ""]
            evidence = "; ".join(x for x in found if x) or "common abbreviation (not in data)"
            writer.writerow([alias, canonical, category, evidence])
    return len(rows)


if __name__ == "__main__":
    n = build(REPO_ROOT / "data/processed/students_unified.csv", REPO_ROOT / "data/processed/jobs.csv")
    print(f"wrote {n} alias rows ({len(ALIASES)} canonical skills) to {ALIAS_TABLE_PATH}")
