from __future__ import annotations

import pickle
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Sequence

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from .data import DatasetBundle


class MissingOptionalDependency(RuntimeError):
    """Raised when an optional recommender dependency is not installed."""


def trim_sparse_rows(matrix: csr_matrix, top_k: int) -> csr_matrix:
    matrix = matrix.tocsr()
    trimmed_data: list[float] = []
    trimmed_indices: list[int] = []
    trimmed_indptr = [0]

    for row_idx in range(matrix.shape[0]):
        start, end = matrix.indptr[row_idx], matrix.indptr[row_idx + 1]
        row_data = matrix.data[start:end]
        row_indices = matrix.indices[start:end]

        if len(row_data) > top_k:
            best = np.argpartition(-row_data, top_k - 1)[:top_k]
            best = best[np.argsort(-row_data[best])]
            row_data = row_data[best]
            row_indices = row_indices[best]
        elif len(row_data) > 1:
            order = np.argsort(-row_data)
            row_data = row_data[order]
            row_indices = row_indices[order]

        trimmed_data.extend(row_data.tolist())
        trimmed_indices.extend(row_indices.tolist())
        trimmed_indptr.append(len(trimmed_data))

    return csr_matrix(
        (np.array(trimmed_data, dtype=np.float32), np.array(trimmed_indices, dtype=np.int32), np.array(trimmed_indptr)),
        shape=matrix.shape,
    )


def compute_top_k_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    finite_mask = np.isfinite(scores)
    finite_indices = np.flatnonzero(finite_mask)
    if len(finite_indices) == 0:
        return np.array([], dtype=np.int64)

    k = min(top_k, len(finite_indices))
    candidate_scores = scores[finite_indices]
    candidate_idx = np.argpartition(-candidate_scores, k - 1)[:k]
    ordered = candidate_idx[np.argsort(-candidate_scores[candidate_idx])]
    return finite_indices[ordered]


def align_popularity_scores(train_matrix: csr_matrix) -> np.ndarray:
    popularity = np.asarray(train_matrix.sum(axis=0)).ravel().astype(np.float32)
    if popularity.max(initial=0.0) > 0:
        popularity /= popularity.max()
    return popularity


@dataclass
class BaseRecommender(ABC):
    model_name: str
    item_ids: np.ndarray
    item_to_index: Dict[int, int]
    user_to_index: Dict[int, int]
    popularity_scores: np.ndarray
    optional_dependency: ClassVar[str | None] = None
    supports_item_similarity: ClassVar[bool] = False

    @property
    def available_user_ids(self) -> list[int]:
        return sorted(int(user_id) for user_id in self.user_to_index)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path):
        with Path(path).open("rb") as file:
            return pickle.load(file)

    def recommend(self, user_id: int, seen_items: set[int], top_k: int) -> list[dict]:
        scores = self.score_all_items(user_id=user_id, seen_items=seen_items)
        if scores is None:
            return []

        top_indices = compute_top_k_indices(scores, top_k)
        return [
            {
                "movieId": int(self.item_ids[idx]),
                "score": float(scores[idx]),
            }
            for idx in top_indices
        ]

    def similar_items(self, item_id: int, top_k: int) -> list[dict]:
        raise NotImplementedError(f"{self.model_name} does not support item-to-item similarity.")

    @abstractmethod
    def score_all_items(self, user_id: int, seen_items: set[int]) -> np.ndarray | None:
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "model_name": self.model_name,
            "optional_dependency": self.optional_dependency,
            "supports_item_similarity": self.supports_item_similarity,
            "available_users": len(self.user_to_index),
            "available_items": len(self.item_ids),
        }

    def _apply_seen_mask(self, scores: np.ndarray, seen_items: set[int]) -> np.ndarray:
        masked_scores = scores.astype(np.float32, copy=True)
        for item_id in seen_items:
            item_index = self.item_to_index.get(int(item_id))
            if item_index is not None:
                masked_scores[item_index] = -np.inf
        return masked_scores


@dataclass
class PopularityRecommender(BaseRecommender):
    @classmethod
    def fit(cls, dataset: DatasetBundle) -> "PopularityRecommender":
        popularity = align_popularity_scores(dataset.matrix)
        return cls(
            model_name="popularity",
            item_ids=np.array(dataset.available_item_ids, dtype=np.int64),
            item_to_index=dataset.item_to_index,
            user_to_index=dataset.user_to_index,
            popularity_scores=popularity,
        )

    def score_all_items(self, user_id: int, seen_items: set[int]) -> np.ndarray | None:
        if user_id not in self.user_to_index:
            return None
        return self._apply_seen_mask(self.popularity_scores, seen_items)


@dataclass
class SVDRecommender(BaseRecommender):
    user_factors: np.ndarray
    item_factors: np.ndarray
    explained_variance: float

    @classmethod
    def fit(cls, dataset: DatasetBundle, factors: int, random_state: int) -> "SVDRecommender":
        max_components = max(2, min(dataset.matrix.shape[0] - 1, dataset.matrix.shape[1] - 1, factors))
        svd = TruncatedSVD(n_components=max_components, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            user_factors = svd.fit_transform(dataset.matrix)
            item_factors = svd.components_.T
        user_factors = np.nan_to_num(user_factors, nan=0.0, posinf=0.0, neginf=0.0)
        item_factors = np.nan_to_num(item_factors, nan=0.0, posinf=0.0, neginf=0.0)
        popularity = align_popularity_scores(dataset.matrix)
        return cls(
            model_name="svd",
            item_ids=np.array(dataset.available_item_ids, dtype=np.int64),
            item_to_index=dataset.item_to_index,
            user_to_index=dataset.user_to_index,
            popularity_scores=popularity,
            user_factors=user_factors.astype(np.float32),
            item_factors=item_factors.astype(np.float32),
            explained_variance=float(svd.explained_variance_ratio_.sum()),
        )

    def score_all_items(self, user_id: int, seen_items: set[int]) -> np.ndarray | None:
        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            return None
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = self.user_factors[user_index] @ self.item_factors.T
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) + self.popularity_scores * 1e-6
        return self._apply_seen_mask(scores, seen_items)


@dataclass
class ContentRecommender(BaseRecommender):
    content_matrix: csr_matrix
    user_profiles: csr_matrix
    supports_item_similarity: ClassVar[bool] = True

    @classmethod
    def fit(cls, dataset: DatasetBundle) -> "ContentRecommender":
        content_matrix = normalize(dataset.content_matrix, norm="l2", axis=1).tocsr()
        user_profiles = normalize(dataset.matrix @ content_matrix, norm="l2", axis=1).tocsr()
        popularity = align_popularity_scores(dataset.matrix)
        return cls(
            model_name="content",
            item_ids=np.array(dataset.available_item_ids, dtype=np.int64),
            item_to_index=dataset.item_to_index,
            user_to_index=dataset.user_to_index,
            popularity_scores=popularity,
            content_matrix=content_matrix,
            user_profiles=user_profiles,
        )

    def score_all_items(self, user_id: int, seen_items: set[int]) -> np.ndarray | None:
        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            return None
        scores = self.user_profiles.getrow(user_index).dot(self.content_matrix.T).toarray().ravel().astype(np.float32)
        scores += self.popularity_scores * 1e-6
        return self._apply_seen_mask(scores, seen_items)

    def similar_items(self, item_id: int, top_k: int) -> list[dict]:
        item_index = self.item_to_index.get(item_id)
        if item_index is None:
            return []
        scores = self.content_matrix.getrow(item_index).dot(self.content_matrix.T).toarray().ravel().astype(np.float32)
        scores[item_index] = -np.inf
        top_indices = compute_top_k_indices(scores, top_k)
        return [{"movieId": int(self.item_ids[idx]), "score": float(scores[idx])} for idx in top_indices]


@dataclass
class ItemCFRecommender(BaseRecommender):
    similarity_matrix: csr_matrix
    interaction_matrix: csr_matrix
    supports_item_similarity: ClassVar[bool] = True

    @classmethod
    def fit(cls, dataset: DatasetBundle, neighbors: int) -> "ItemCFRecommender":
        item_popularity = np.asarray(dataset.matrix.sum(axis=0)).ravel().astype(np.float32)
        item_norms = np.sqrt(np.maximum(item_popularity, 1.0))
        normalization = diags(1.0 / item_norms)
        normalized_items = normalization @ dataset.matrix.T
        similarity = (normalized_items @ normalized_items.T).tocsr()
        similarity.setdiag(0.0)
        similarity.eliminate_zeros()
        similarity = trim_sparse_rows(similarity, top_k=neighbors)
        popularity = align_popularity_scores(dataset.matrix)
        return cls(
            model_name="itemcf",
            item_ids=np.array(dataset.available_item_ids, dtype=np.int64),
            item_to_index=dataset.item_to_index,
            user_to_index=dataset.user_to_index,
            popularity_scores=popularity,
            similarity_matrix=similarity,
            interaction_matrix=dataset.matrix.tocsr(),
        )

    def score_all_items(self, user_id: int, seen_items: set[int]) -> np.ndarray | None:
        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            return None
        user_vector = self.interaction_matrix.getrow(user_index)
        scores = user_vector.dot(self.similarity_matrix).toarray().ravel().astype(np.float32)
        scores += self.popularity_scores * 1e-6
        return self._apply_seen_mask(scores, seen_items)

    def similar_items(self, item_id: int, top_k: int) -> list[dict]:
        item_index = self.item_to_index.get(item_id)
        if item_index is None:
            return []
        row = self.similarity_matrix.getrow(item_index)
        scores = np.full(len(self.item_ids), -np.inf, dtype=np.float32)
        if row.nnz:
            scores[row.indices] = row.data.astype(np.float32)
        top_indices = compute_top_k_indices(scores, top_k)
        return [{"movieId": int(self.item_ids[idx]), "score": float(scores[idx])} for idx in top_indices]


@dataclass
class UserCFRecommender(BaseRecommender):
    interaction_matrix: csr_matrix
    user_norms: np.ndarray
    neighbors: int

    @classmethod
    def fit(cls, dataset: DatasetBundle, neighbors: int) -> "UserCFRecommender":
        user_activity = np.asarray(dataset.matrix.sum(axis=1)).ravel().astype(np.float32)
        popularity = align_popularity_scores(dataset.matrix)
        return cls(
            model_name="usercf",
            item_ids=np.array(dataset.available_item_ids, dtype=np.int64),
            item_to_index=dataset.item_to_index,
            user_to_index=dataset.user_to_index,
            popularity_scores=popularity,
            interaction_matrix=dataset.matrix.tocsr(),
            user_norms=np.sqrt(np.maximum(user_activity, 1.0)),
            neighbors=neighbors,
        )

    def score_all_items(self, user_id: int, seen_items: set[int]) -> np.ndarray | None:
        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            return None

        user_row = self.interaction_matrix.getrow(user_index)
        overlaps = self.interaction_matrix @ user_row.T
        similarities = overlaps.toarray().ravel().astype(np.float32)
        denominator = self.user_norms * self.user_norms[user_index]
        similarities = np.divide(similarities, denominator, out=np.zeros_like(similarities), where=denominator > 0)
        similarities[user_index] = 0.0

        non_zero = np.flatnonzero(similarities > 0)
        if len(non_zero) == 0:
            scores = self.popularity_scores.copy()
            return self._apply_seen_mask(scores, seen_items)

        if len(non_zero) > self.neighbors:
            best = np.argpartition(-similarities[non_zero], self.neighbors - 1)[: self.neighbors]
            non_zero = non_zero[best]

        weights = similarities[non_zero]
        scores = self.interaction_matrix[non_zero].T.dot(weights).astype(np.float32)
        scores = np.asarray(scores).ravel()
        scores += self.popularity_scores * 1e-6
        return self._apply_seen_mask(scores, seen_items)


@dataclass
class ALSRecommender(BaseRecommender):
    user_factors: np.ndarray
    item_factors: np.ndarray
    optional_dependency: ClassVar[str | None] = "implicit"

    @classmethod
    def fit(
        cls,
        dataset: DatasetBundle,
        factors: int,
        regularization: float,
        alpha: float,
        iterations: int,
        random_state: int,
    ) -> "ALSRecommender":
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as exc:
            raise MissingOptionalDependency("implicit is required for ALS. Install requirements-optional.txt.") from exc

        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
        )
        confidence = (dataset.matrix * alpha).astype(np.float32).tocsr()
        model.fit(confidence)
        popularity = align_popularity_scores(dataset.matrix)
        return cls(
            model_name="als",
            item_ids=np.array(dataset.available_item_ids, dtype=np.int64),
            item_to_index=dataset.item_to_index,
            user_to_index=dataset.user_to_index,
            popularity_scores=popularity,
            user_factors=np.asarray(model.user_factors, dtype=np.float32),
            item_factors=np.asarray(model.item_factors, dtype=np.float32),
        )

    def score_all_items(self, user_id: int, seen_items: set[int]) -> np.ndarray | None:
        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            return None
        scores = self.user_factors[user_index] @ self.item_factors.T
        scores = scores.astype(np.float32) + self.popularity_scores * 1e-6
        return self._apply_seen_mask(scores, seen_items)


@dataclass
class BPRRecommender(BaseRecommender):
    user_factors: np.ndarray
    item_factors: np.ndarray
    optional_dependency: ClassVar[str | None] = "implicit"

    @classmethod
    def fit(
        cls,
        dataset: DatasetBundle,
        factors: int,
        regularization: float,
        learning_rate: float,
        iterations: int,
        random_state: int,
    ) -> "BPRRecommender":
        try:
            from implicit.bpr import BayesianPersonalizedRanking
        except ImportError as exc:
            raise MissingOptionalDependency("implicit is required for BPR. Install requirements-optional.txt.") from exc

        model = BayesianPersonalizedRanking(
            factors=factors,
            regularization=regularization,
            learning_rate=learning_rate,
            iterations=iterations,
            random_state=random_state,
        )
        model.fit(dataset.matrix.tocsr())
        popularity = align_popularity_scores(dataset.matrix)
        return cls(
            model_name="bpr",
            item_ids=np.array(dataset.available_item_ids, dtype=np.int64),
            item_to_index=dataset.item_to_index,
            user_to_index=dataset.user_to_index,
            popularity_scores=popularity,
            user_factors=np.asarray(model.user_factors, dtype=np.float32),
            item_factors=np.asarray(model.item_factors, dtype=np.float32),
        )

    def score_all_items(self, user_id: int, seen_items: set[int]) -> np.ndarray | None:
        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            return None
        scores = self.user_factors[user_index] @ self.item_factors.T
        scores = scores.astype(np.float32) + self.popularity_scores * 1e-6
        return self._apply_seen_mask(scores, seen_items)


@dataclass
class LightFMRecommender(BaseRecommender):
    model: object
    item_features: csr_matrix
    num_threads: int
    optional_dependency: ClassVar[str | None] = "lightfm"

    @classmethod
    def fit(
        cls,
        dataset: DatasetBundle,
        factors: int,
        epochs: int,
        learning_rate: float,
        num_threads: int,
        random_state: int,
    ) -> "LightFMRecommender":
        try:
            from lightfm import LightFM
        except ImportError as exc:
            raise MissingOptionalDependency("lightfm is required for LightFM. Install requirements-optional.txt.") from exc

        model = LightFM(
            no_components=factors,
            loss="warp",
            learning_rate=learning_rate,
            random_state=np.random.RandomState(random_state),
        )
        item_features = dataset.content_matrix.tocsr()
        model.fit(
            dataset.matrix.tocoo(),
            item_features=item_features,
            epochs=epochs,
            num_threads=num_threads,
            verbose=False,
        )
        popularity = align_popularity_scores(dataset.matrix)
        return cls(
            model_name="lightfm",
            item_ids=np.array(dataset.available_item_ids, dtype=np.int64),
            item_to_index=dataset.item_to_index,
            user_to_index=dataset.user_to_index,
            popularity_scores=popularity,
            model=model,
            item_features=item_features,
            num_threads=num_threads,
        )

    def score_all_items(self, user_id: int, seen_items: set[int]) -> np.ndarray | None:
        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            return None
        item_indices = np.arange(len(self.item_ids), dtype=np.int32)
        user_indices = np.full(len(self.item_ids), user_index, dtype=np.int32)
        scores = self.model.predict(
            user_ids=user_indices,
            item_ids=item_indices,
            item_features=self.item_features,
            num_threads=self.num_threads,
        ).astype(np.float32)
        scores += self.popularity_scores * 1e-6
        return self._apply_seen_mask(scores, seen_items)


@dataclass
class TwoStageRecommender(BaseRecommender):
    recall_models: dict[str, BaseRecommender]
    reranker: BaseRecommender
    recall_model_names: list[str]
    reranker_name: str
    recall_k_per_model: int
    candidate_limit: int

    @classmethod
    def fit(
        cls,
        dataset: DatasetBundle,
        recall_models: dict[str, BaseRecommender],
        reranker: BaseRecommender,
        recall_k_per_model: int,
        candidate_limit: int,
    ) -> "TwoStageRecommender":
        return cls(
            model_name="two_stage",
            item_ids=np.array(dataset.available_item_ids, dtype=np.int64),
            item_to_index=dataset.item_to_index,
            user_to_index=dataset.user_to_index,
            popularity_scores=align_popularity_scores(dataset.matrix),
            recall_models=recall_models,
            reranker=reranker,
            recall_model_names=list(recall_models),
            reranker_name=reranker.model_name,
            recall_k_per_model=recall_k_per_model,
            candidate_limit=candidate_limit,
        )

    def _candidate_indices(self, user_id: int, seen_items: set[int]) -> list[int]:
        candidate_indices: set[int] = set()
        for model in self.recall_models.values():
            for entry in model.recommend(user_id=user_id, seen_items=seen_items, top_k=self.recall_k_per_model):
                item_index = self.item_to_index.get(int(entry["movieId"]))
                if item_index is not None:
                    candidate_indices.add(item_index)

        if len(candidate_indices) < self.candidate_limit:
            fallback_scores = self.reranker.score_all_items(user_id=user_id, seen_items=seen_items)
            if fallback_scores is not None:
                fallback_indices = compute_top_k_indices(fallback_scores, self.candidate_limit)
                candidate_indices.update(int(idx) for idx in fallback_indices.tolist())

        if not candidate_indices:
            return []
        return sorted(candidate_indices)[: self.candidate_limit]

    def score_all_items(self, user_id: int, seen_items: set[int]) -> np.ndarray | None:
        rerank_scores = self.reranker.score_all_items(user_id=user_id, seen_items=seen_items)
        if rerank_scores is None:
            return None

        scores = np.full(len(self.item_ids), -np.inf, dtype=np.float32)
        candidate_indices = self._candidate_indices(user_id=user_id, seen_items=seen_items)
        if not candidate_indices:
            return scores
        scores[candidate_indices] = rerank_scores[candidate_indices]
        return scores

    def describe(self) -> dict:
        payload = super().describe()
        payload.update(
            {
                "recall_model_names": self.recall_model_names,
                "reranker_name": self.reranker_name,
                "recall_k_per_model": self.recall_k_per_model,
                "candidate_limit": self.candidate_limit,
            }
        )
        return payload


MODEL_REGISTRY = {
    "popularity": PopularityRecommender,
    "content": ContentRecommender,
    "usercf": UserCFRecommender,
    "itemcf": ItemCFRecommender,
    "svd": SVDRecommender,
    "als": ALSRecommender,
    "bpr": BPRRecommender,
    "lightfm": LightFMRecommender,
    "two_stage": TwoStageRecommender,
}

ALL_MODELS = list(MODEL_REGISTRY)
