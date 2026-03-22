from __future__ import annotations

import argparse

import pandas as pd

from .artifacts import load_manifest, load_model_artifact, load_shared_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate top-k recommendations for a user.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--model", type=str, default="svd")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.artifacts_dir)
    shared = load_shared_artifacts(args.artifacts_dir)
    model_name = args.model.lower()

    model_status = manifest["models"].get(model_name)
    if not model_status:
        print(f"Model '{model_name}' is not registered. Available models: {', '.join(sorted(manifest['models']))}")
        return
    if model_status.get("status") != "trained":
        reason = model_status.get("reason", "model is not available")
        print(f"Model '{model_name}' is not ready: {reason}")
        return

    if args.user_id not in shared.user_history:
        sample_ids = ", ".join(str(user_id) for user_id in shared.available_user_ids[:10])
        print(f"User {args.user_id} is not available in the trained model. Example user ids: {sample_ids}")
        return

    model = load_model_artifact(args.artifacts_dir, model_name)
    movie_lookup = shared.movies.set_index("movieId")[["title", "genres"]]
    recommendations = model.recommend(
        user_id=args.user_id,
        seen_items=shared.user_history.get(args.user_id, set()),
        top_k=args.top_k,
    )

    if not recommendations:
        print(f"No recommendations generated for user {args.user_id} with model {model_name}.")
        return

    result = pd.DataFrame(recommendations)
    result["title"] = result["movieId"].map(movie_lookup["title"])
    result["genres"] = result["movieId"].map(movie_lookup["genres"])
    result.insert(0, "rank", range(1, len(result) + 1))

    print(f"Top-{args.top_k} recommendations for user {args.user_id} with model {model_name}:")
    print(result[["rank", "movieId", "title", "genres", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
