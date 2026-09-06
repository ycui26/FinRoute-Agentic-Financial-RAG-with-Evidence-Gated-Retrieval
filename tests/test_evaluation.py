import pytest

from finroute.evaluation import (
    all_gold_at_k,
    bootstrap_mean_ci,
    ndcg_at_k,
    page_hit_at_k,
    page_precision_at_k,
    page_recall_at_k,
    reciprocal_rank_at_k,
    token_f1,
)


def test_page_metrics():
    retrieved = [("a", 1), ("a", 2), ("b", 1)]
    gold = {("a", 2), ("b", 1)}
    assert page_hit_at_k(retrieved, gold, 1) == 0
    assert page_hit_at_k(retrieved, gold, 2) == 1
    assert page_recall_at_k(retrieved, gold, 2) == 0.5
    assert page_precision_at_k(retrieved, gold, 2) == 0.5
    assert all_gold_at_k(retrieved, gold, 3) == 1
    assert reciprocal_rank_at_k(retrieved, gold, 3) == 0.5
    assert 0 < ndcg_at_k(retrieved, gold, 3) <= 1


def test_text_and_bootstrap_metrics():
    assert token_f1("revenue was 10 million", "10 million revenue") > 0.7
    low, high = bootstrap_mean_ci([0, 1, 1, 1], repeats=500, seed=7)
    assert 0 <= low <= high <= 1
