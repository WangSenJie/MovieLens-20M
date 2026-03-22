from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize

GENRES_PLACEHOLDER = "(no genres listed)"
PERSON_STOPWORDS = {
    "based",
    "best",
    "big",
    "book",
    "classic",
    "comedy",
    "cult",
    "dark",
    "drama",
    "family",
    "film",
    "funny",
    "great",
    "love",
    "movie",
    "new",
    "old",
    "original",
    "romance",
    "seen",
    "story",
    "thriller",
    "war",
}


@dataclass
class DatasetBundle:
    movies: pd.DataFrame
    tags: pd.DataFrame
    interactions: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    matrix: csr_matrix
    content_matrix: csr_matrix
    user_to_index: Dict[int, int]
    item_to_index: Dict[int, int]
    index_to_user: Dict[int, int]
    index_to_item: Dict[int, int]
    user_history: Dict[int, set[int]]
    test_targets: Dict[int, int]
    available_user_ids: list[int]
    available_item_ids: list[int]
    feature_summary: dict[str, int] = field(default_factory=dict)


def load_movielens(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)
    movies = pd.read_csv(data_path / "movies.csv")
    ratings = pd.read_csv(data_path / "ratings.csv", usecols=["userId", "movieId", "rating", "timestamp"])
    tags = pd.read_csv(data_path / "tags.csv", usecols=["userId", "movieId", "tag", "timestamp"])
    return movies, ratings, tags


def parse_comma_separated_ints(raw_value: str | None) -> list[int]:
    if not raw_value:
        return []
    values = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    return values


def parse_comma_separated_strings(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [chunk.strip() for chunk in raw_value.split(",") if chunk.strip()]


def normalize_tag_text(value: str) -> str:
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    tokens = [token for token in text.split() if len(token) > 1]
    return " ".join(tokens)


def prepare_interactions(
    ratings: pd.DataFrame,
    min_rating: float,
    min_user_interactions: int,
    min_item_interactions: int,
    sample_users: int | None,
    random_state: int,
    include_user_ids: Sequence[int] | None = None,
) -> pd.DataFrame:
    interactions = ratings.loc[ratings["rating"] >= min_rating].copy()
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    interactions = interactions.sort_values(["userId", "movieId", "timestamp"])
    interactions = interactions.drop_duplicates(subset=["userId", "movieId"], keep="last")

    include_user_ids = list(include_user_ids or [])
    if sample_users is not None:
        unique_users = interactions["userId"].drop_duplicates()
        sampled_users = unique_users.sample(n=min(sample_users, unique_users.nunique()), random_state=random_state)
        sampled_set = set(sampled_users.tolist())
        sampled_set.update(user_id for user_id in include_user_ids if user_id in set(unique_users.tolist()))
        interactions = interactions.loc[interactions["userId"].isin(sampled_set)].copy()

    while True:
        user_counts = interactions["userId"].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        filtered = interactions.loc[interactions["userId"].isin(valid_users)].copy()

        item_counts = filtered["movieId"].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        filtered = filtered.loc[filtered["movieId"].isin(valid_items)].copy()

        if len(filtered) == len(interactions):
            break
        interactions = filtered

    return interactions.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)


def leave_one_out_split(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = interactions.sort_values(["userId", "timestamp", "movieId"]).copy()
    test = ordered.groupby("userId", group_keys=False).tail(1).copy()
    train = ordered.drop(index=test.index).copy()

    remaining = train["userId"].value_counts()
    valid_users = remaining[remaining >= 1].index

    train = train.loc[train["userId"].isin(valid_users)].reset_index(drop=True)
    test = test.loc[test["userId"].isin(valid_users)].reset_index(drop=True)
    return train, test


def build_interaction_matrix(train: pd.DataFrame, user_to_index: Dict[int, int], item_to_index: Dict[int, int]) -> csr_matrix:
    row_index = train["userId"].map(user_to_index).to_numpy()
    col_index = train["movieId"].map(item_to_index).to_numpy()
    values = np.ones(len(train), dtype=np.float32)
    return csr_matrix((values, (row_index, col_index)), shape=(len(user_to_index), len(item_to_index)))


def _genres_to_tokens(raw_value: str) -> list[str]:
    genres = str(raw_value).split("|")
    return [genre.strip().lower() for genre in genres if genre and genre != GENRES_PLACEHOLDER]


def extract_year(title: str) -> int | None:
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    if not match:
        return None
    return int(match.group(1))


def year_bucket(year: int | None) -> str:
    if year is None:
        return "year_unknown"
    return f"decade_{year // 10 * 10}s"


def load_genome_annotations(
    data_dir: str | Path,
    item_ids: Sequence[int],
    top_n: int,
    min_relevance: float,
) -> tuple[dict[int, list[str]], dict[str, int]]:
    data_path = Path(data_dir)
    scores_path = data_path / "genome_scores.csv"
    tags_path = data_path / "genome_tags.csv"
    if not scores_path.exists() or not tags_path.exists():
        return {}, {"genome_movies": 0, "genome_rows": 0}

    tag_lookup = pd.read_csv(tags_path).set_index("tagId")["tag"].to_dict()
    item_set = set(int(item_id) for item_id in item_ids)
    relevant_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(scores_path, usecols=["movieId", "tagId", "relevance"], chunksize=250_000):
        chunk = chunk.loc[chunk["movieId"].isin(item_set) & (chunk["relevance"] >= min_relevance)]
        if not chunk.empty:
            relevant_chunks.append(chunk)

    if not relevant_chunks:
        return {}, {"genome_movies": 0, "genome_rows": 0}

    relevant = pd.concat(relevant_chunks, ignore_index=True)
    relevant["tag"] = relevant["tagId"].map(tag_lookup)
    relevant = relevant.dropna(subset=["tag"])
    relevant = relevant.sort_values(["movieId", "relevance"], ascending=[True, False])
    top = relevant.groupby("movieId", group_keys=False).head(top_n).copy()
    genome_map = top.groupby("movieId")["tag"].apply(list).to_dict()
    return genome_map, {"genome_movies": len(genome_map), "genome_rows": len(top)}


def is_person_like_tag(value: str) -> bool:
    cleaned = str(value).strip().lower()
    if not cleaned or any(char.isdigit() for char in cleaned):
        return False
    if not re.fullmatch(r"[a-z][a-z .'\-]+", cleaned):
        return False
    tokens = [token for token in cleaned.split() if token]
    if len(tokens) < 2 or len(tokens) > 3:
        return False
    if any(token in PERSON_STOPWORDS for token in tokens):
        return False
    return True


def infer_people_metadata(
    item_ids: Sequence[int],
    tags: pd.DataFrame,
    genome_map: dict[int, list[str]],
) -> pd.DataFrame:
    tag_candidates = {}
    if not tags.empty:
        grouped = tags.groupby("movieId")["tag"].apply(list)
        tag_candidates = grouped.to_dict()

    rows = []
    for item_id in item_ids:
        counter: Counter[str] = Counter()
        for raw_tag in tag_candidates.get(int(item_id), []):
            if is_person_like_tag(raw_tag):
                counter[raw_tag.strip().title()] += 2
        for genome_tag in genome_map.get(int(item_id), []):
            if is_person_like_tag(genome_tag):
                counter[genome_tag.strip().title()] += 1

        ranked_people = [name for name, _ in counter.most_common(4)]
        director = ranked_people[0] if ranked_people else ""
        actors = " | ".join(ranked_people[1:4]) if len(ranked_people) > 1 else ""
        rows.append({"movieId": int(item_id), "director": director, "actors": actors})

    return pd.DataFrame(rows, columns=["movieId", "director", "actors"])


def load_people_metadata(
    data_dir: str | Path,
    item_ids: Sequence[int],
    tags: pd.DataFrame,
    genome_map: dict[int, list[str]],
) -> pd.DataFrame:
    data_path = Path(data_dir)
    metadata_path = data_path / "movie_people.csv"
    if metadata_path.exists():
        people = pd.read_csv(metadata_path)
        required = {"movieId", "director", "actors"}
        missing = required - set(people.columns)
        if missing:
            raise ValueError(f"movie_people.csv is missing required columns: {', '.join(sorted(missing))}")
        people = people.loc[people["movieId"].isin(item_ids), ["movieId", "director", "actors"]].copy()
        return people
    return infer_people_metadata(item_ids=item_ids, tags=tags, genome_map=genome_map)


def build_people_documents(movie_frame: pd.DataFrame) -> list[str]:
    documents = []
    for row in movie_frame.itertuples(index=False):
        values = []
        if getattr(row, "director", ""):
            values.append(str(row.director))
        if getattr(row, "actors", ""):
            values.append(str(row.actors).replace("|", " "))
        normalized = " ".join(normalize_tag_text(value) for value in values if str(value).strip())
        documents.append(normalized)
    return documents


def build_content_features(
    data_dir: str | Path,
    movies: pd.DataFrame,
    tags: pd.DataFrame,
    item_ids: Iterable[int],
    max_tag_features: int = 3000,
    max_genome_features: int = 1000,
    genome_top_n: int = 8,
    genome_min_relevance: float = 0.6,
) -> tuple[pd.DataFrame, csr_matrix, dict[str, int]]:
    item_ids = np.array(list(item_ids), dtype=np.int64)
    movie_frame = movies.set_index("movieId").reindex(item_ids).reset_index()
    movie_frame["genres_list"] = movie_frame["genres"].map(_genres_to_tokens)
    movie_frame["year"] = movie_frame["title"].map(extract_year)
    movie_frame["decade_token"] = movie_frame["year"].map(year_bucket)

    mlb = MultiLabelBinarizer(sparse_output=True)
    genre_matrix = mlb.fit_transform(movie_frame["genres_list"]).astype(np.float32)

    decade_binarizer = MultiLabelBinarizer(sparse_output=True)
    decade_matrix = decade_binarizer.fit_transform(movie_frame["decade_token"].map(lambda token: [token])).astype(np.float32)

    year_values = movie_frame["year"].fillna(movie_frame["year"].median()).fillna(0).astype(np.float32).to_numpy()
    if year_values.max(initial=0.0) > year_values.min(initial=0.0):
        year_scaled = (year_values - year_values.min()) / (year_values.max() - year_values.min())
    else:
        year_scaled = np.zeros_like(year_values, dtype=np.float32)
    year_matrix = csr_matrix(year_scaled.reshape(-1, 1), dtype=np.float32)

    relevant_tags = tags.loc[tags["movieId"].isin(item_ids)].copy()
    if not relevant_tags.empty:
        relevant_tags["clean_tag"] = relevant_tags["tag"].map(normalize_tag_text)
        relevant_tags = relevant_tags.loc[relevant_tags["clean_tag"].str.len() > 0]
        tag_docs = relevant_tags.groupby("movieId")["clean_tag"].apply(lambda values: " ".join(values))
        tag_texts = [tag_docs.get(int(item_id), "") for item_id in item_ids]
    else:
        tag_texts = [""] * len(item_ids)

    if any(text.strip() for text in tag_texts):
        tag_vectorizer = TfidfVectorizer(max_features=max_tag_features, min_df=2)
        tag_matrix = tag_vectorizer.fit_transform(tag_texts).astype(np.float32)
        tag_features = len(tag_vectorizer.get_feature_names_out())
    else:
        tag_matrix = csr_matrix((len(item_ids), 0), dtype=np.float32)
        tag_features = 0

    genome_map, genome_summary = load_genome_annotations(
        data_dir=data_dir,
        item_ids=item_ids,
        top_n=genome_top_n,
        min_relevance=genome_min_relevance,
    )
    movie_frame["genome_tags"] = movie_frame["movieId"].map(lambda movie_id: " | ".join(genome_map.get(int(movie_id), [])))
    genome_docs = [" ".join(normalize_tag_text(tag) for tag in genome_map.get(int(item_id), [])) for item_id in item_ids]
    if any(text.strip() for text in genome_docs):
        genome_vectorizer = TfidfVectorizer(max_features=max_genome_features, min_df=2)
        genome_matrix = genome_vectorizer.fit_transform(genome_docs).astype(np.float32)
        genome_features = len(genome_vectorizer.get_feature_names_out())
    else:
        genome_matrix = csr_matrix((len(item_ids), 0), dtype=np.float32)
        genome_features = 0

    people_frame = load_people_metadata(data_dir=data_dir, item_ids=item_ids, tags=relevant_tags, genome_map=genome_map)
    movie_frame = movie_frame.merge(people_frame, on="movieId", how="left")
    movie_frame["director"] = movie_frame["director"].fillna("")
    movie_frame["actors"] = movie_frame["actors"].fillna("")
    people_docs = build_people_documents(movie_frame)
    if any(text.strip() for text in people_docs):
        people_vectorizer = TfidfVectorizer(max_features=512, min_df=1)
        people_matrix = people_vectorizer.fit_transform(people_docs).astype(np.float32)
        people_features = len(people_vectorizer.get_feature_names_out())
    else:
        people_matrix = csr_matrix((len(item_ids), 0), dtype=np.float32)
        people_features = 0

    content_matrix = hstack(
        [genre_matrix, decade_matrix, year_matrix, tag_matrix, genome_matrix, people_matrix],
        format="csr",
        dtype=np.float32,
    )
    content_matrix = normalize(content_matrix, norm="l2", axis=1)
    movie_frame = movie_frame.drop(columns=["genres_list"])

    feature_summary = {
        "genre_features": genre_matrix.shape[1],
        "decade_features": decade_matrix.shape[1],
        "year_features": year_matrix.shape[1],
        "tag_features": tag_features,
        "genome_features": genome_features,
        "people_features": people_features,
    }
    feature_summary.update(genome_summary)
    return movie_frame, content_matrix, feature_summary


def build_dataset(
    data_dir: str | Path,
    min_rating: float,
    min_user_interactions: int,
    min_item_interactions: int,
    sample_users: int | None,
    random_state: int,
    include_user_ids: Sequence[int] | None = None,
    max_tag_features: int = 3000,
    max_genome_features: int = 1000,
    genome_top_n: int = 8,
    genome_min_relevance: float = 0.6,
) -> DatasetBundle:
    movies, ratings, tags = load_movielens(data_dir)
    interactions = prepare_interactions(
        ratings=ratings,
        min_rating=min_rating,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        sample_users=sample_users,
        random_state=random_state,
        include_user_ids=include_user_ids,
    )
    train, test = leave_one_out_split(interactions)
    if train.empty or test.empty:
        raise ValueError(
            "No training data remains after filtering. Increase --sample-users or lower the interaction thresholds."
        )

    item_ids = np.sort(train["movieId"].unique())
    user_ids = np.sort(train["userId"].unique())

    user_to_index = {int(user_id): idx for idx, user_id in enumerate(user_ids)}
    item_to_index = {int(item_id): idx for idx, item_id in enumerate(item_ids)}
    index_to_user = {idx: int(user_id) for user_id, idx in user_to_index.items()}
    index_to_item = {idx: int(item_id) for item_id, idx in item_to_index.items()}

    matrix = build_interaction_matrix(train, user_to_index, item_to_index)
    filtered_movies, content_matrix, feature_summary = build_content_features(
        data_dir=data_dir,
        movies=movies,
        tags=tags,
        item_ids=item_ids,
        max_tag_features=max_tag_features,
        max_genome_features=max_genome_features,
        genome_top_n=genome_top_n,
        genome_min_relevance=genome_min_relevance,
    )

    user_history = {
        int(user_id): set(int(movie_id) for movie_id in movie_ids)
        for user_id, movie_ids in train.groupby("userId")["movieId"]
    }
    test_targets = {int(user_id): int(movie_id) for user_id, movie_id in test[["userId", "movieId"]].itertuples(index=False)}

    return DatasetBundle(
        movies=filtered_movies,
        tags=tags,
        interactions=interactions,
        train=train,
        test=test,
        matrix=matrix,
        content_matrix=content_matrix,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        index_to_user=index_to_user,
        index_to_item=index_to_item,
        user_history=user_history,
        test_targets=test_targets,
        available_user_ids=sorted(user_history),
        available_item_ids=[int(item_id) for item_id in item_ids.tolist()],
        feature_summary=feature_summary,
    )
