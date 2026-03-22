from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .artifacts import load_json, load_manifest, load_model_artifact, load_shared_artifacts

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class ArtifactStore:
    def __init__(self, artifacts_dir: str | Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.manifest = load_manifest(self.artifacts_dir)
        self.shared = load_shared_artifacts(self.artifacts_dir)
        self.movies = self._load_movies()
        self.models = {}
        self.model_errors = {}
        self.metrics = self._load_metrics()
        self._load_models()

    def _load_movies(self):
        csv_path = self.artifacts_dir / "movies_filtered.csv"
        if csv_path.exists():
            movie_lookup = {}
            with csv_path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    try:
                        movie_id = int(row.get("movieId", ""))
                    except (TypeError, ValueError):
                        continue
                    movie_lookup[movie_id] = row
            return movie_lookup

        movie_lookup = {}
        movies = self.shared.movies
        if hasattr(movies, "to_dict"):
            try:
                for row in movies.to_dict(orient="records"):
                    movie_id = int(row.get("movieId"))
                    movie_lookup[movie_id] = row
            except Exception:
                pass
        return movie_lookup

    def _load_metrics(self):
        metrics_path = self.artifacts_dir / "metrics_summary.json"
        if metrics_path.exists():
            return load_json(metrics_path)
        return []

    def _load_models(self) -> None:
        for model_name, metadata in self.manifest.get("models", {}).items():
            if metadata.get("status") != "trained":
                self.model_errors[model_name] = metadata.get("reason", "model is not trained")
                continue
            try:
                self.models[model_name] = load_model_artifact(self.artifacts_dir, model_name)
            except Exception as exc:
                self.model_errors[model_name] = repr(exc)

    def get_model(self, model_name: str):
        model = self.models.get(model_name)
        if model is not None:
            return model
        if model_name in self.manifest.get("models", {}):
            reason = self.model_errors.get(model_name, "model is unavailable")
            raise HTTPException(status_code=400, detail={"message": f"Model '{model_name}' is unavailable.", "reason": reason})
        raise HTTPException(status_code=404, detail={"message": f"Model '{model_name}' does not exist."})

    def get_movie_payload(self, movie_id: int, score: float) -> dict:
        if movie_id not in self.movies:
            return {
                "movieId": int(movie_id),
                "title": f"movieId={movie_id}",
                "genres": "",
                "year": None,
                "director": "",
                "actors": "",
                "genome_tags": "",
                "score": float(score),
            }

        row = self.movies[movie_id]
        year_value = row.get("year")
        if year_value in (None, "", "nan", "NaN"):
            parsed_year = None
        else:
            try:
                parsed_year = int(float(year_value))
            except (TypeError, ValueError):
                parsed_year = None
        return {
            "movieId": int(movie_id),
            "title": str(row.get("title", f"movieId={movie_id}")),
            "genres": str(row.get("genres", "")),
            "year": parsed_year,
            "director": str(row.get("director", "")),
            "actors": str(row.get("actors", "")),
            "genome_tags": str(row.get("genome_tags", "")),
            "score": float(score),
        }

    def recommend(self, model_name: str, user_id: int, top_k: int) -> dict:
        if user_id not in self.shared.user_history:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"User {user_id} is not available in the trained artifacts.",
                    "example_user_ids": self.shared.available_user_ids[:10],
                },
            )

        model = self.get_model(model_name)
        try:
            recommendations = model.recommend(
                user_id=user_id,
                seen_items=self.shared.user_history.get(user_id, set()),
                top_k=top_k,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"message": f"Failed to generate recommendations for user {user_id}.", "reason": repr(exc)},
            ) from exc
        return {
            "model": model_name,
            "userId": int(user_id),
            "top_k": int(top_k),
            "results": [self.get_movie_payload(entry["movieId"], entry["score"]) for entry in recommendations],
        }

    def similar_items(self, model_name: str, movie_id: int, top_k: int) -> dict:
        if movie_id not in self.movies:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"Movie {movie_id} is not available in the trained artifacts.",
                    "example_movie_ids": self.shared.available_item_ids[:10],
                },
            )

        model = self.get_model(model_name)
        if not getattr(model, "supports_item_similarity", False):
            raise HTTPException(status_code=400, detail={"message": f"Model '{model_name}' does not support item similarity."})

        try:
            results = model.similar_items(movie_id=movie_id, top_k=top_k)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"message": f"Failed to generate similar items for movie {movie_id}.", "reason": repr(exc)},
            ) from exc
        return {
            "model": model_name,
            "movieId": int(movie_id),
            "top_k": int(top_k),
            "results": [self.get_movie_payload(entry["movieId"], entry["score"]) for entry in results],
        }

    def list_models(self) -> list[dict]:
        output = []
        for model_name, metadata in self.manifest.get("models", {}).items():
            output.append(
                {
                    "model": model_name,
                    "status": metadata.get("status"),
                    "loaded": model_name in self.models,
                    "reason": self.model_errors.get(model_name) or metadata.get("reason"),
                    "optional_dependency": metadata.get("optional_dependency"),
                    "supports_item_similarity": metadata.get("description", {}).get("supports_item_similarity", False),
                }
            )
        return output


def create_app(artifacts_dir: str | Path) -> FastAPI:
    store = ArtifactStore(artifacts_dir)
    app = FastAPI(title="MovieLens Recommendation API", version="1.0.0")
    app.state.store = store

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        if not STATIC_DIR.exists():
            raise HTTPException(status_code=404, detail={"message": "Static frontend is not available."})
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "artifacts_dir": str(store.artifacts_dir.resolve()),
            "loaded_models": sorted(store.models),
            "dataset_stats": store.shared.dataset_stats,
        }

    @app.get("/models")
    def list_models() -> dict:
        return {"models": store.list_models()}

    @app.get("/users/{user_id}/recommendations")
    def user_recommendations(user_id: int, model: str = "svd", top_k: int = 10) -> dict:
        return store.recommend(model_name=model.lower(), user_id=user_id, top_k=top_k)

    @app.get("/items/{movie_id}/similar")
    def item_similarity(movie_id: int, model: str = "content", top_k: int = 10) -> dict:
        return store.similar_items(model_name=model.lower(), movie_id=movie_id, top_k=top_k)

    @app.get("/metrics/latest")
    def latest_metrics() -> dict:
        return {"metrics": store.metrics}

    @app.get("/ab-test/plan")
    def ab_test_plan() -> dict:
        return {
            "title": "MovieLens 推荐系统 A/B Test 方案",
            "goal": "验证两阶段推荐与单阶段推荐在点击率、播放转化和长尾曝光上的差异。",
            "variants": [
                {"name": "control", "description": "现有单阶段排序模型，例如 svd 或 lightfm。"},
                {"name": "treatment", "description": "两阶段架构：content/itemcf 召回，svd/lightfm 重排。"},
            ],
            "north_star_metrics": ["CTR", "watch_rate", "add_to_favorite_rate"],
            "guardrail_metrics": ["latency_p95", "bounce_rate", "long_tail_coverage", "complaint_rate"],
            "segmentation": ["new_users", "active_users", "heavy_movie_watchers"],
            "duration_hint": "至少持续 7-14 天，并保证每组样本量足够。",
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve trained MovieLens recommenders with FastAPI.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import uvicorn

    app = create_app(args.artifacts_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
