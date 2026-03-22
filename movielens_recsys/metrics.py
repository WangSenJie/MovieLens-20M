from __future__ import annotations

import math
from typing import Iterable, Sequence


def reciprocal_rank(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / rank


def hit_rate_at_k(rank: int | None, k: int) -> float:
    return float(rank is not None and rank <= k)


def recall_at_k(rank: int | None, k: int) -> float:
    return hit_rate_at_k(rank, k)


def ndcg_at_k(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1.0)


def average_precision_at_k(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / rank


def item_coverage_at_k(recommendation_lists: Sequence[Sequence[int]], total_items: int, k: int) -> float:
    if total_items <= 0:
        return 0.0
    recommended_items = set()
    for items in recommendation_lists:
        recommended_items.update(int(item_id) for item_id in items[:k])
    return round(len(recommended_items) / total_items, 6)


def summarize_ranking_metrics(rows: Iterable[dict], total_items: int) -> dict:
    rows = list(rows)
    if not rows:
        return {
            "users_evaluated": 0,
            "hit_rate@10": 0.0,
            "hit_rate@20": 0.0,
            "recall@10": 0.0,
            "recall@20": 0.0,
            "ndcg@10": 0.0,
            "ndcg@20": 0.0,
            "map@10": 0.0,
            "mrr": 0.0,
            "item_coverage@10": 0.0,
        }

    users_evaluated = len(rows)
    recommendation_lists = [row["recommended_items"] for row in rows]
    metrics = {
        "users_evaluated": users_evaluated,
        "hit_rate@10": round(sum(row["hit_rate@10"] for row in rows) / users_evaluated, 6),
        "hit_rate@20": round(sum(row["hit_rate@20"] for row in rows) / users_evaluated, 6),
        "recall@10": round(sum(row["recall@10"] for row in rows) / users_evaluated, 6),
        "recall@20": round(sum(row["recall@20"] for row in rows) / users_evaluated, 6),
        "ndcg@10": round(sum(row["ndcg@10"] for row in rows) / users_evaluated, 6),
        "ndcg@20": round(sum(row["ndcg@20"] for row in rows) / users_evaluated, 6),
        "map@10": round(sum(row["map@10"] for row in rows) / users_evaluated, 6),
        "mrr": round(sum(row["mrr"] for row in rows) / users_evaluated, 6),
        "item_coverage@10": item_coverage_at_k(recommendation_lists, total_items=total_items, k=10),
    }
    return metrics
