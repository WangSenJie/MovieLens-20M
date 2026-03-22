from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SHARED_PICKLE = "shared.pkl"
MANIFEST_JSON = "manifest.json"
METRICS_JSON = "metrics_summary.json"
METRICS_CSV = "metrics_summary.csv"


@dataclass
class SharedArtifacts:
    movies: pd.DataFrame
    user_history: dict[int, set[int]]
    test_targets: dict[int, int]
    available_user_ids: list[int]
    available_item_ids: list[int]
    dataset_stats: dict[str, Any]
    config: dict[str, Any]


def ensure_artifact_layout(artifacts_dir: str | Path) -> Path:
    base_dir = Path(artifacts_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "models").mkdir(parents=True, exist_ok=True)
    return base_dir


def model_dir(artifacts_dir: str | Path, model_name: str) -> Path:
    path = ensure_artifact_layout(artifacts_dir) / "models" / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj: Any, path: str | Path) -> None:
    with Path(path).open("wb") as file:
        pickle.dump(obj, file)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def save_json(obj: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_shared_artifacts(artifacts_dir: str | Path, shared: SharedArtifacts) -> None:
    base_dir = ensure_artifact_layout(artifacts_dir)
    save_pickle(shared, base_dir / SHARED_PICKLE)
    shared.movies.to_csv(base_dir / "movies_filtered.csv", index=False)


def load_shared_artifacts(artifacts_dir: str | Path) -> SharedArtifacts:
    return load_pickle(Path(artifacts_dir) / SHARED_PICKLE)


def save_model_artifact(artifacts_dir: str | Path, model_name: str, model: Any, metadata: dict[str, Any]) -> None:
    path = model_dir(artifacts_dir, model_name)
    save_pickle(model, path / "model.pkl")
    save_json(metadata, path / "metadata.json")


def load_model_artifact(artifacts_dir: str | Path, model_name: str) -> Any:
    return load_pickle(Path(artifacts_dir) / "models" / model_name / "model.pkl")


def save_manifest(artifacts_dir: str | Path, manifest: dict[str, Any]) -> None:
    base_dir = ensure_artifact_layout(artifacts_dir)
    save_json(manifest, base_dir / MANIFEST_JSON)
    save_json(manifest, base_dir / "metadata.json")


def load_manifest(artifacts_dir: str | Path) -> dict[str, Any]:
    return load_json(Path(artifacts_dir) / MANIFEST_JSON)
