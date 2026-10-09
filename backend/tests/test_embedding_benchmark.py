"""Phase 6 benchmark, labelling pool, splits and weak labels.

Covers the audit fixes (I-3 empty sections, I-4 double-counted description, I-5 noisy timing), the model list,
the fixed per-student splits and the rules behind the weak labels. No model weights are loaded here: a fake
encoder records what it was asked to embed.
"""

import numpy as np
import pandas as pd
import pytest

from src.embeddings import embedding_model_benchmark as bench
from src.evaluation.labelling_pool import build_pool, select_students
from src.evaluation.splits import RATIOS, SPLITS, assign_splits, split_for
from src.evaluation.weak_labels import generate, grade, load_rules, role_families


class FakeEncoder:
    """Embeds text as a normalised bag of three keywords; remembers every text it was given."""
    seen: list[str]

    def __init__(self):
        self.seen = []

    def encode(self, texts, role):
        self.seen += list(texts)
        rows = np.array([[t.lower().count(w) + 0.01 for w in ("python", "sql", "nurse")] for t in texts], float)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def data():
    return bench.load_benchmark_data()


def test_model_list_covers_the_spec_slots_with_tfidf_as_the_floor():
    assert {bench.MODEL_CONFIGS[k]["spec_slot"] for k in bench.CANDIDATES} == {"A", "B", "C", "D"}
    assert bench.MODEL_CONFIGS["minilm"]["hf_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert bench.FLOOR == "tfidf" and bench.MODEL_CONFIGS["tfidf"]["spec_slot"] == "Floor"
    assert bench.NOT_RUN_MODELS["custom"]["status"] == "Not feasible — insufficient labeled data"
    assert "Phase 7" in bench.NOT_RUN_MODELS["hybrid"]["status"]


def test_job_text_uses_each_field_once_and_drops_the_randomly_assigned_fields(data):
    _, reps, _ = data
    row = reps.iloc[0].to_dict()
    sections = bench.job_section_texts(row)
    text = bench.job_text(row)
    assert text.count(row["job_description"]) == 1  # audit I-4: the description was counted twice
    for field in bench.RANDOMLY_ASSIGNED_JOB_FIELDS:
        assert row[field] not in sections.values()
    assert row["role"] in sections["title"]


def test_the_two_fields_left_out_really_are_random_within_a_role(data):
    _, _, jobs = data
    per_role = jobs.groupby("role")[list(bench.RANDOMLY_ASSIGNED_JOB_FIELDS)].nunique().median()
    assert (per_role >= 5).all()  # one role shows many degrees and experience ranges


def test_role_representatives_are_the_distinct_jobs(data):
    _, reps, jobs = data
    assert len(reps) == jobs["role"].nunique() == 376 and reps["role"].is_unique


def test_empty_sections_are_never_embedded_and_weights_renormalise(data):
    students = pd.DataFrame([{"student_id": "S1", "skills_raw": "Python SQL", "internship_raw": "",
                              "degree": "", "branch": "", "technical_projects": "", "certifications": "", "achievements": ""},
                             {"student_id": "S2", **{c: "" for c in ("skills_raw", "internship_raw", "degree", "branch",
                                                                     "technical_projects", "certifications", "achievements")}}])
    jobs = pd.DataFrame([{"job_id": "J1", "job_title": "Analyst", "role": "Data Analyst", "skills": "Python SQL",
                          "job_description": "Nurse care", "responsibilities": ""}])
    enc = FakeEncoder()
    sids, jids, m = bench.score_matrix(enc, students, jobs, "section_weighted")
    assert "" not in enc.seen  # audit I-3
    # S1 has only skills on its side and the job has no responsibilities: the score is the skills cosine alone
    assert m[0, 0] == pytest.approx(1.0) and np.isnan(m[1, 0])
    with pytest.raises(ValueError, match="I-3"):
        bench.SentenceTransformerEncoder.encode(type("E", (), {"cfg": bench.MODEL_CONFIGS["e5"]})(), ["", "x"], "query")


def test_full_text_scores_match_the_section_free_cosine():
    students = pd.DataFrame([{"student_id": "S1", "skills_raw": "python", "internship_raw": "sql"}])
    jobs = pd.DataFrame([{"job_id": "J1", "job_title": "", "role": "", "skills": "python sql", "job_description": "",
                          "responsibilities": ""}])
    _, _, m = bench.score_matrix(FakeEncoder(), students, jobs, "full_text")
    assert m[0, 0] == pytest.approx(1.0)


def test_timing_uses_warm_repeated_trials_not_one_call(data):
    students, reps, _ = data
    pairs, batch, _, _ = bench._timing_inputs(students, reps)
    assert len(pairs) == bench.N_TRIALS >= 10 and len(batch) == bench.BATCH_TEXTS  # audit I-5
    timing = bench._time_device(FakeEncoder(), pairs, batch)
    assert timing["single_pair_latency_ms"]["trials"] == bench.N_TRIALS and "stdev" in timing["single_pair_latency_ms"]


def test_complexity_points_have_reasons():
    assert bench.complexity("e5")["points"] == 3  # two prefixes + the PyTorch runtime
    assert bench.complexity("sbert")["points"] == 1
    assert bench.complexity("tfidf")["reasons"] == ["vocabulary is fitted on the corpus: refit and re-embed "
                                                    "everything when the data changes"]


def test_the_deployed_tfidf_version_is_unchanged_by_the_benchmark_rewrite():
    from src.embeddings.tfidf_baseline import TfidfBaseline

    assert TfidfBaseline.fit_from_processed().version == "tfidf-3ec097a5876e"


def test_splits_are_fixed_per_student_and_roughly_follow_the_ratios(data):
    students, _, _ = data
    table = assign_splits(students["student_id"])
    assert set(table["split"]) == set(SPLITS) and table["student_id"].is_unique
    assert split_for("S00019") == split_for("S00019")
    shares = table["split"].value_counts(normalize=True)
    for name, share in RATIOS:
        assert abs(shares[name] - share) < 0.08


def test_labelling_students_come_only_from_evaluation_splits_and_engineering(data):
    students, _, _ = data
    chosen = select_students(students, 30)
    assert len(chosen) == 30 and set(chosen["split"]) == {"validation", "test"}
    assert (chosen["branch"] != "Non-Engineering / Management").all()
    assert chosen["source_dataset"].value_counts().to_dict() == {"placement_cell": 20, "kaggle_resume_dataset": 10}


def test_round_robin_pool_gives_every_model_its_top_choices_plus_random_jobs():
    selected = pd.DataFrame({"student_id": ["S1"], "split": ["validation"]})
    jobs = pd.DataFrame({"job_id": [f"J{i}" for i in range(20)]})
    rankings = {m: {"S1": [f"J{(i + 3 * n) % 20}" for i in range(20)]} for n, m in enumerate(("a", "b", "c"))}
    pool, sources = build_pool(selected, jobs, rankings, n_pooled=6, n_random=2)
    assert len(pool) == 8 and pool["job_id"].is_unique and sorted(pool["display_order"]) == list(range(1, 9))
    pooled = set(sources.loc[~sources["random_fill"], "job_id"])
    assert all(rankings[m]["S1"][0] in pooled and rankings[m]["S1"][1] in pooled for m in rankings)
    assert sources["random_fill"].sum() == 2


def test_weak_label_rules_cover_every_domain_and_need_no_model(data):
    students, reps, _ = data
    rules = load_rules()
    assert set(students["domain"]) <= set(rules["domains"])
    fam = role_families("Machine Learning Engineer", rules["families"])
    assert fam == {"data_ml"}
    assert grade("Data Analyst and Machine Learning", fam, rules) == 3
    assert grade("Data Engineer", fam, rules) == 1
    assert grade("Legal", fam, rules) == 0
    assert role_families("Security Operations Center (SOC) Analyst", rules["families"]) == {"networking_security"}
    assert "civil" not in role_families("Aircraft Design Engineer", rules["families"])
    labels = generate(students.head(3), reps.head(10), rules)
    assert len(labels) == 30 and set(labels["label_source"]) == {"weakly_labelled"}
    assert set(labels["relevance_label"]) <= {0, 1, 3}
