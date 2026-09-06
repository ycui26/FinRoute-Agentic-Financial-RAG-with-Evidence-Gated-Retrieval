from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np


PagePair = tuple[str, int]
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?", re.I)


def page_hit_at_k(retrieved: list[PagePair], gold: set[PagePair], k: int) -> int:
    return int(bool(set(retrieved[:k]) & gold))


def page_recall_at_k(retrieved: list[PagePair], gold: set[PagePair], k: int) -> float:
    return len(set(retrieved[:k]) & gold) / len(gold) if gold else 0.0


def page_precision_at_k(retrieved: list[PagePair], gold: set[PagePair], k: int) -> float:
    denominator = min(k, len(retrieved))
    return len(set(retrieved[:k]) & gold) / denominator if denominator else 0.0


def all_gold_at_k(retrieved: list[PagePair], gold: set[PagePair], k: int) -> int:
    return int(bool(gold) and gold.issubset(set(retrieved[:k])))


def reciprocal_rank_at_k(retrieved: list[PagePair], gold: set[PagePair], k: int) -> float:
    for rank, pair in enumerate(retrieved[:k], 1):
        if pair in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[PagePair], gold: set[PagePair], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, pair in enumerate(retrieved[:k], 1) if pair in gold)
    ideal_hits = min(k, len(gold))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred = TOKEN_RE.findall(str(prediction).lower())
    gold = TOKEN_RE.findall(str(reference).lower())
    if not pred or not gold:
        return float(pred == gold)
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def bootstrap_mean_ci(
    values: Iterable[float], repeats: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=float)
    for start in range(0, repeats, 1000):
        size = min(1000, repeats - start)
        means[start:start + size] = rng.choice(
            array, size=(size, len(array)), replace=True
        ).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def retrieval_metrics(
    retrieved: list[PagePair], gold: set[PagePair], k: int = 5
) -> dict[str, float]:
    return {
        f"hit@{k}": page_hit_at_k(retrieved, gold, k),
        f"recall@{k}": page_recall_at_k(retrieved, gold, k),
        f"precision@{k}": page_precision_at_k(retrieved, gold, k),
        f"all_gold@{k}": all_gold_at_k(retrieved, gold, k),
        f"mrr@{k}": reciprocal_rank_at_k(retrieved, gold, k),
        f"ndcg@{k}": ndcg_at_k(retrieved, gold, k),
    }
