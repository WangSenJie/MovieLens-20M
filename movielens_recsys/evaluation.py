from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize_ranking_metrics,
)


def compute_target_rank(scores: np.ndarray, target_index: int) -> int | None:
    if target_index < 0 or target_index >= len(scores):
        return None
    target_score = scores[target_index]
    if not np.isfinite(target_score):
        return None
    return int(1 + np.sum(scores > target_score))


def evaluate_model(model, user_history: dict[int, set[int]], test_targets: dict[int, int], item_to_index: dict[int, int]) -> tuple[dict, list[dict]]:
    rows = []
    recommendation_lists = []

    for user_id, target_item in test_targets.items():
        seen_items = user_history.get(user_id, set())
        scores = model.score_all_items(user_id=user_id, seen_items=seen_items)
        if scores is None:
            continue

        target_index = item_to_index.get(target_item)
        rank = compute_target_rank(scores, target_index) if target_index is not None else None
        recommendations = model.recommend(user_id=user_id, seen_items=seen_items, top_k=20)
        recommendation_lists.append([entry["movieId"] for entry in recommendations])

        rows.append(
            {
                "user_id": int(user_id),
                "target_item": int(target_item),
                "rank": rank,
                "hit_rate@10": hit_rate_at_k(rank, 10),
                "hit_rate@20": hit_rate_at_k(rank, 20),
                "recall@10": recall_at_k(rank, 10),
                "recall@20": recall_at_k(rank, 20),
                "ndcg@10": ndcg_at_k(rank, 10),
                "ndcg@20": ndcg_at_k(rank, 20),
                "map@10": average_precision_at_k(rank, 10),
                "mrr": reciprocal_rank(rank),
                "recommended_items": recommendation_lists[-1],
            }
        )

    metrics = summarize_ranking_metrics(rows, total_items=len(item_to_index))
    return metrics, rows


def metrics_to_frame(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty and "ndcg@10" in frame.columns:
        frame = frame.sort_values(["ndcg@10", "hit_rate@10"], ascending=False).reset_index(drop=True)
    return frame


def sample_recommendations(model, user_ids: Iterable[int], user_history: dict[int, set[int]], top_k: int) -> list[dict]:
    samples = []
    for user_id in user_ids:
        recommendations = model.recommend(user_id=user_id, seen_items=user_history.get(user_id, set()), top_k=top_k)
        samples.append({"user_id": int(user_id), "recommendations": recommendations})
    return samples
