"""Generate rule-based weak relevance labels (config/weak_label_rules.yaml) for every student x distinct job.

The labels come from two fields no model sees: the student's stated domain and the job's role
name. They are independent of all embedding models (no circular evaluation), and are written
with label_source = weakly_labelled and a provenance manifest saying so.

Usage:  python -m src.evaluation.weak_labels     (writes data/benchmark/weak_labels.csv + .manifest.json)
"""

import hashlib
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd
import yaml

from src.embeddings.embedding_model_benchmark import load_benchmark_data
from src.evaluation.splits import SPLIT_VERSION, split_for

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RULES = os.path.join(ROOT, "config", "weak_label_rules.yaml")
OUT = os.path.join(ROOT, "data", "benchmark", "weak_labels.csv")


def load_rules(path: str = RULES) -> dict:
    with open(path, encoding="utf-8") as fh:
        rules = yaml.safe_load(fh)
    families = {name: re.compile(pattern, re.I) for name, pattern in rules["families"].items()}
    for domain, spec in rules["domains"].items():
        unknown = {f for f in spec["primary"] + spec["related"] if f not in families}
        if unknown:
            raise ValueError(f"domain {domain!r} refers to unknown families {sorted(unknown)}")
    return {"version": rules["version"], "families": families, "domains": rules["domains"]}


def role_families(role: str, families: dict) -> set[str]:
    return {name for name, pattern in families.items() if pattern.search(role)}


def grade(student_domain: str, families_of_job: set[str], rules: dict) -> int:
    spec = rules["domains"][student_domain]
    if families_of_job & set(spec["primary"]):
        return 3
    if families_of_job & set(spec["related"]):
        return 1
    return 0


def generate(students: pd.DataFrame, jobs: pd.DataFrame, rules: dict) -> pd.DataFrame:
    unknown = sorted(set(students["domain"]) - set(rules["domains"]))
    if unknown:
        raise ValueError(f"students have domains with no rule: {unknown}")
    fam = {j: role_families(r, rules["families"]) for j, r in zip(jobs["job_id"], jobs["role"])}
    rows = [{"student_id": s, "job_id": j, "relevance_label": grade(d, fam[j], rules),
             "label_source": "weakly_labelled", "split": split_for(s)}
            for s, d in zip(students["student_id"], students["domain"]) for j in jobs["job_id"]]
    return pd.DataFrame(rows)


def write(out: str = OUT) -> dict:
    rules = load_rules()
    students, jobs, _ = load_benchmark_data()
    labels = generate(students, jobs, rules)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    labels.to_csv(out, index=False)
    fam = {r: role_families(r, rules["families"]) for r in jobs["role"]}
    per_student_relevant = labels[labels["relevance_label"] == 3].groupby("student_id").size()
    with open(RULES, "rb") as fh:
        rules_sha = hashlib.sha256(fh.read()).hexdigest()
    manifest = {
        "label_source": "weakly_labelled", "generated_by": "rules",
        "method": "job role name -> job families (regex); student's stated domain -> primary/related families; "
                  "3 = primary family, 1 = related family, 0 = otherwise",
        "models_used_for_labels": [], "rules_version": rules["version"], "rules_sha256": rules_sha,
        "split_version": SPLIT_VERSION, "pairs": int(len(labels)),
        "students": int(labels["student_id"].nunique()), "jobs": int(labels["job_id"].nunique()),
        "label_distribution": {str(k): int(v) for k, v in labels["relevance_label"].value_counts().sort_index().items()},
        "roles_in_no_family": sum(1 for f in fam.values() if not f),
        "students_with_no_strong_job": int(labels["student_id"].nunique() - len(per_student_relevant)),
        "strong_jobs_per_student_median": float(per_student_relevant.median()),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.splitext(out)[0] + ".manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


if __name__ == "__main__":
    print(json.dumps(write(), indent=2))
