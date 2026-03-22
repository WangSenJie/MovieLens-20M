import unittest

from movielens_recsys.metrics import summarize_ranking_metrics


class MetricsTestCase(unittest.TestCase):
    def test_summarize_ranking_metrics(self):
        rows = [
            {
                "hit_rate@10": 1.0,
                "hit_rate@20": 1.0,
                "recall@10": 1.0,
                "recall@20": 1.0,
                "ndcg@10": 1.0,
                "ndcg@20": 1.0,
                "map@10": 1.0,
                "mrr": 1.0,
                "recommended_items": [1, 2, 3],
            },
            {
                "hit_rate@10": 0.0,
                "hit_rate@20": 1.0,
                "recall@10": 0.0,
                "recall@20": 1.0,
                "ndcg@10": 0.0,
                "ndcg@20": 0.5,
                "map@10": 0.0,
                "mrr": 0.5,
                "recommended_items": [3, 4, 5],
            },
        ]

        metrics = summarize_ranking_metrics(rows, total_items=10)
        self.assertEqual(metrics["users_evaluated"], 2)
        self.assertAlmostEqual(metrics["hit_rate@10"], 0.5)
        self.assertAlmostEqual(metrics["hit_rate@20"], 1.0)
        self.assertAlmostEqual(metrics["item_coverage@10"], 0.5)


if __name__ == "__main__":
    unittest.main()
