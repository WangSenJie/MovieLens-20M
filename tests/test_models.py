import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from movielens_recsys.data import DatasetBundle
from movielens_recsys.models import ContentRecommender, ItemCFRecommender, PopularityRecommender, SVDRecommender, TwoStageRecommender, UserCFRecommender


def build_toy_dataset() -> DatasetBundle:
    movies = pd.DataFrame(
        [
            {"movieId": 1, "title": "Movie 1", "genres": "Action"},
            {"movieId": 2, "title": "Movie 2", "genres": "Comedy"},
            {"movieId": 3, "title": "Movie 3", "genres": "Action"},
            {"movieId": 4, "title": "Movie 4", "genres": "Comedy"},
        ]
    )
    train = pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": 4.0, "timestamp": pd.Timestamp("2024-01-01")},
            {"userId": 1, "movieId": 2, "rating": 4.0, "timestamp": pd.Timestamp("2024-01-02")},
            {"userId": 2, "movieId": 1, "rating": 4.0, "timestamp": pd.Timestamp("2024-01-01")},
            {"userId": 2, "movieId": 3, "rating": 5.0, "timestamp": pd.Timestamp("2024-01-02")},
            {"userId": 3, "movieId": 2, "rating": 4.0, "timestamp": pd.Timestamp("2024-01-01")},
            {"userId": 3, "movieId": 4, "rating": 5.0, "timestamp": pd.Timestamp("2024-01-02")},
        ]
    )
    test = pd.DataFrame(
        [
            {"userId": 1, "movieId": 3, "rating": 5.0, "timestamp": pd.Timestamp("2024-01-03")},
            {"userId": 2, "movieId": 4, "rating": 4.0, "timestamp": pd.Timestamp("2024-01-03")},
            {"userId": 3, "movieId": 1, "rating": 4.0, "timestamp": pd.Timestamp("2024-01-03")},
        ]
    )

    user_to_index = {1: 0, 2: 1, 3: 2}
    item_to_index = {1: 0, 2: 1, 3: 2, 4: 3}
    matrix = csr_matrix(
        np.array(
            [
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
            dtype=np.float32,
        )
    )
    content_matrix = csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.1],
                [0.1, 1.0],
            ],
            dtype=np.float32,
        )
    )
    user_history = {1: {1, 2}, 2: {1, 3}, 3: {2, 4}}
    test_targets = {1: 3, 2: 4, 3: 1}

    return DatasetBundle(
        movies=movies,
        tags=pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"]),
        interactions=pd.concat([train, test], ignore_index=True),
        train=train,
        test=test,
        matrix=matrix,
        content_matrix=content_matrix,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        index_to_user={0: 1, 1: 2, 2: 3},
        index_to_item={0: 1, 1: 2, 2: 3, 3: 4},
        user_history=user_history,
        test_targets=test_targets,
        available_user_ids=[1, 2, 3],
        available_item_ids=[1, 2, 3, 4],
    )


class ModelSmokeTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = build_toy_dataset()

    def test_popularity_recommendation_excludes_seen_items(self):
        model = PopularityRecommender.fit(self.dataset)
        recommendations = model.recommend(user_id=1, seen_items={1, 2}, top_k=2)
        self.assertEqual(len(recommendations), 2)
        self.assertTrue(all(entry["movieId"] not in {1, 2} for entry in recommendations))

    def test_content_supports_recommendation_and_similarity(self):
        model = ContentRecommender.fit(self.dataset)
        recommendations = model.recommend(user_id=1, seen_items={1, 2}, top_k=2)
        similar_items = model.similar_items(item_id=1, top_k=2)
        self.assertTrue(all(entry["movieId"] not in {1, 2} for entry in recommendations))
        self.assertEqual(similar_items[0]["movieId"], 3)

    def test_neighbor_models_return_unseen_items(self):
        itemcf = ItemCFRecommender.fit(self.dataset, neighbors=2)
        usercf = UserCFRecommender.fit(self.dataset, neighbors=2)
        itemcf_recs = itemcf.recommend(user_id=1, seen_items={1, 2}, top_k=2)
        usercf_recs = usercf.recommend(user_id=1, seen_items={1, 2}, top_k=2)
        self.assertTrue(all(entry["movieId"] not in {1, 2} for entry in itemcf_recs))
        self.assertTrue(all(entry["movieId"] not in {1, 2} for entry in usercf_recs))

    def test_two_stage_recommender_returns_candidate_reranked_items(self):
        recall_models = {
            "content": ContentRecommender.fit(self.dataset),
            "itemcf": ItemCFRecommender.fit(self.dataset, neighbors=2),
            "popularity": PopularityRecommender.fit(self.dataset),
        }
        reranker = SVDRecommender.fit(self.dataset, factors=2, random_state=42)
        model = TwoStageRecommender.fit(
            self.dataset,
            recall_models=recall_models,
            reranker=reranker,
            recall_k_per_model=2,
            candidate_limit=4,
        )
        recommendations = model.recommend(user_id=1, seen_items={1, 2}, top_k=2)
        self.assertTrue(all(entry["movieId"] not in {1, 2} for entry in recommendations))
        self.assertGreaterEqual(len(recommendations), 1)


if __name__ == "__main__":
    unittest.main()
