"""Phase 6 ranking metrics (src/evaluation/retrieval_metrics.py), checked against hand-computed values."""

import math

import pandas as pd
import pytest

from src.evaluation.retrieval_metrics import (bootstrap_ci, evaluate, mean_kendall_tau, paired_difference,
                                              query_metrics, rank)


def test_rank_orders_by_score_and_breaks_ties_by_job_id():
    assert rank({"J2": 0.5, "J1": 0.5, "J3": 0.9}) == ["J3", "J1", "J2"]


def test_query_metrics_match_hand_computed_values():
    labels = {"a": 0, "b": 3, "c": 2, "d": 0}
    m = query_metrics(["a", "b", "c", "d"], labels, ks=(2,), threshold=2)
    assert m["mrr"] == 0.5
    assert m["precision@2"] == 0.5 and m["recall@2"] == 0.5 and m["hit_rate@2"] == 1.0
    dcg = 7 / math.log2(3)                 # grades [0, 3] at ranks 1-2, gain 2^g - 1
    ideal = 7 / math.log2(2) + 3 / math.log2(3)  # grades [3, 2]
    assert m["ndcg@2"] == pytest.approx(dcg / ideal)


def test_a_perfect_ranking_scores_one_and_a_student_with_nothing_relevant_is_skipped():
    labels = {"a": 3, "b": 2, "c": 0}
    m = query_metrics(["a", "b", "c"], labels, ks=(2,), threshold=2)
    assert m["mrr"] == m["ndcg@2"] == m["recall@2"] == 1.0
    assert query_metrics(["a", "b"], {"a": 1, "b": 0}, ks=(2,), threshold=2) is None


def test_evaluate_ranks_unscored_pairs_last_and_flags_degenerate_k():
    labels = pd.DataFrame({"student_id": ["S1"] * 3 + ["S2"] * 3, "job_id": ["a", "b", "c"] * 2,
                           "relevance_label": [3, 0, 0, 0, 0, 1]})
    scores = {("S1", "a"): 0.1, ("S1", "b"): 0.9}  # ("S1", "c") unscored -> ranked last
    r = evaluate(scores, labels, ks=(1, 5), threshold=2)
    assert r["students_evaluated"] == 1 and r["students_excluded_no_relevant"] == 1  # S2 has no grade >= 2
    assert r["unscored_pairs"] == 4  # S1/c and all of S2
    assert r["mean"]["mrr"] == 0.5 and r["mean"]["precision@1"] == 0.0
    assert r["degenerate_k"] == [5]  # 3 judged jobs per student <= 5


def test_kendall_tau_is_rank_based_so_score_scale_does_not_matter():
    labels = pd.DataFrame({"student_id": ["S1"] * 4, "job_id": list("abcd"), "relevance_label": [3, 2, 1, 0]})
    small = {("S1", j): s for j, s in zip("abcd", (0.04, 0.03, 0.02, 0.01))}
    large = {("S1", j): s for j, s in zip("abcd", (0.99, 0.98, 0.80, 0.75))}
    assert mean_kendall_tau(labels, small)[0] == pytest.approx(1.0)
    assert mean_kendall_tau(labels, large)[0] == pytest.approx(1.0)


def test_paired_difference_separates_clear_and_unclear_differences():
    a = {f"S{i}": 0.9 for i in range(20)}
    b = {f"S{i}": 0.1 for i in range(20)}
    assert paired_difference(a, b)["distinguishable"] is True
    noisy = {f"S{i}": (0.9 if i % 2 else 0.1) for i in range(20)}
    flipped = {f"S{i}": (0.1 if i % 2 else 0.9) for i in range(20)}
    assert paired_difference(noisy, flipped)["distinguishable"] is False
    assert bootstrap_ci([0.5]) is None
