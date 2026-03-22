from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .artifacts import SharedArtifacts, ensure_artifact_layout, model_dir, save_json, save_manifest, save_model_artifact, save_shared_artifacts
from .data import build_dataset, parse_comma_separated_ints, parse_comma_separated_strings
from .evaluation import evaluate_model, metrics_to_frame, sample_recommendations
from .models import ALL_MODELS, MODEL_REGISTRY, MissingOptionalDependency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate MovieLens recommenders.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--models", type=str, default="all")
    parser.add_argument("--sample-users", type=int, default=None)
    parser.add_argument("--include-user-ids", type=str, default="")
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--min-user-interactions", type=int, default=10)
    parser.add_argument("--min-item-interactions", type=int, default=20)
    parser.add_argument("--max-tag-features", type=int, default=3000)
    parser.add_argument("--max-genome-features", type=int, default=1000)
    parser.add_argument("--genome-top-n", type=int, default=8)
    parser.add_argument("--genome-min-relevance", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--als-iterations", type=int, default=15)
    parser.add_argument("--bpr-iterations", type=int, default=50)
    parser.add_argument("--lightfm-epochs", type=int, default=20)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--two-stage-recall-models", type=str, default="content,itemcf,popularity")
    parser.add_argument("--two-stage-reranker", type=str, default="svd")
    parser.add_argument("--recall-k-per-model", type=int, default=100)
    parser.add_argument("--candidate-limit", type=int, default=300)
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "quick" and args.sample_users is None:
        args.sample_users = 5000
    return args


def resolve_models(raw_models: str) -> list[str]:
    if raw_models == "all":
        return ALL_MODELS
    models = [name.strip().lower() for name in raw_models.split(",") if name.strip()]
    unknown_models = [name for name in models if name not in MODEL_REGISTRY]
    if unknown_models:
        raise ValueError(f"Unknown models: {', '.join(unknown_models)}")
    return models


def fit_model(model_name: str, dataset, args: argparse.Namespace, trained_models: dict[str, object]):
    if model_name == "popularity":
        return MODEL_REGISTRY[model_name].fit(dataset)
    if model_name == "content":
        return MODEL_REGISTRY[model_name].fit(dataset)
    if model_name == "usercf":
        return MODEL_REGISTRY[model_name].fit(dataset, neighbors=args.neighbors)
    if model_name == "itemcf":
        return MODEL_REGISTRY[model_name].fit(dataset, neighbors=args.neighbors)
    if model_name == "svd":
        return MODEL_REGISTRY[model_name].fit(dataset, factors=args.factors, random_state=args.random_state)
    if model_name == "als":
        return MODEL_REGISTRY[model_name].fit(
            dataset,
            factors=args.factors,
            regularization=args.regularization,
            alpha=args.alpha,
            iterations=args.als_iterations,
            random_state=args.random_state,
        )
    if model_name == "bpr":
        return MODEL_REGISTRY[model_name].fit(
            dataset,
            factors=args.factors,
            regularization=args.regularization,
            learning_rate=args.learning_rate,
            iterations=args.bpr_iterations,
            random_state=args.random_state,
        )
    if model_name == "lightfm":
        return MODEL_REGISTRY[model_name].fit(
            dataset,
            factors=args.factors,
            epochs=args.lightfm_epochs,
            learning_rate=args.learning_rate,
            num_threads=args.num_threads,
            random_state=args.random_state,
        )
    if model_name == "two_stage":
        recall_model_names = parse_comma_separated_strings(args.two_stage_recall_models)
        if not recall_model_names:
            recall_model_names = ["content", "itemcf", "popularity"]
        recall_models = {}
        for recall_model_name in recall_model_names:
            if recall_model_name == "two_stage":
                continue
            if recall_model_name not in trained_models:
                trained_models[recall_model_name] = fit_model(recall_model_name, dataset, args, trained_models)
            recall_models[recall_model_name] = trained_models[recall_model_name]

        reranker_name = args.two_stage_reranker.lower()
        if reranker_name not in trained_models:
            trained_models[reranker_name] = fit_model(reranker_name, dataset, args, trained_models)
        reranker = trained_models[reranker_name]
        return MODEL_REGISTRY[model_name].fit(
            dataset,
            recall_models=recall_models,
            reranker=reranker,
            recall_k_per_model=args.recall_k_per_model,
            candidate_limit=args.candidate_limit,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    args = apply_preset(parse_args())
    artifacts_dir = ensure_artifact_layout(args.artifacts_dir)
    include_user_ids = parse_comma_separated_ints(args.include_user_ids)
    model_names = resolve_models(args.models)

    dataset = build_dataset(
        data_dir=args.data_dir,
        min_rating=args.min_rating,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        sample_users=args.sample_users,
        random_state=args.random_state,
        include_user_ids=include_user_ids,
        max_tag_features=args.max_tag_features,
        max_genome_features=args.max_genome_features,
        genome_top_n=args.genome_top_n,
        genome_min_relevance=args.genome_min_relevance,
    )

    config = {
        "data_dir": args.data_dir,
        "preset": args.preset,
        "models": model_names,
        "sample_users": args.sample_users,
        "include_user_ids": include_user_ids,
        "min_rating": args.min_rating,
        "min_user_interactions": args.min_user_interactions,
        "min_item_interactions": args.min_item_interactions,
        "max_tag_features": args.max_tag_features,
        "max_genome_features": args.max_genome_features,
        "genome_top_n": args.genome_top_n,
        "genome_min_relevance": args.genome_min_relevance,
        "top_k": args.top_k,
        "random_state": args.random_state,
        "factors": args.factors,
        "neighbors": args.neighbors,
        "als_iterations": args.als_iterations,
        "bpr_iterations": args.bpr_iterations,
        "lightfm_epochs": args.lightfm_epochs,
        "regularization": args.regularization,
        "alpha": args.alpha,
        "learning_rate": args.learning_rate,
        "num_threads": args.num_threads,
        "two_stage_recall_models": parse_comma_separated_strings(args.two_stage_recall_models),
        "two_stage_reranker": args.two_stage_reranker.lower(),
        "recall_k_per_model": args.recall_k_per_model,
        "candidate_limit": args.candidate_limit,
    }
    dataset_stats = {
        "users": len(dataset.available_user_ids),
        "items": len(dataset.available_item_ids),
        "train_interactions": len(dataset.train),
        "test_interactions": len(dataset.test),
        "interactions": len(dataset.interactions),
        "feature_summary": dataset.feature_summary,
    }

    shared = SharedArtifacts(
        movies=dataset.movies,
        user_history=dataset.user_history,
        test_targets=dataset.test_targets,
        available_user_ids=dataset.available_user_ids,
        available_item_ids=dataset.available_item_ids,
        dataset_stats=dataset_stats,
        config=config,
    )
    save_shared_artifacts(artifacts_dir, shared)

    manifest = {
        "config": config,
        "dataset": dataset_stats,
        "available_user_ids_sample": dataset.available_user_ids[:10],
        "available_item_ids_sample": dataset.available_item_ids[:10],
        "models": {},
    }
    trained_models = {}
    metrics_summary = {}

    for model_name in model_names:
        started = time.time()
        try:
            model = fit_model(model_name, dataset, args, trained_models)
            trained_models[model_name] = model
            metrics, _ = evaluate_model(
                model=model,
                user_history=dataset.user_history,
                test_targets=dataset.test_targets,
                item_to_index=dataset.item_to_index,
            )
            metrics_summary[model_name] = metrics
            metadata = {
                "status": "trained",
                "training_seconds": round(time.time() - started, 3),
                "description": model.describe(),
                "metrics": metrics,
                "sample_recommendations": sample_recommendations(
                    model=model,
                    user_ids=dataset.available_user_ids[:3],
                    user_history=dataset.user_history,
                    top_k=args.top_k,
                ),
            }
            save_model_artifact(artifacts_dir, model_name, model, metadata)
            manifest["models"][model_name] = metadata
        except MissingOptionalDependency as exc:
            metadata = {
                "status": "skipped",
                "reason": str(exc),
                "optional_dependency": getattr(MODEL_REGISTRY[model_name], "optional_dependency", None),
            }
            save_json(metadata, model_dir(artifacts_dir, model_name) / "metadata.json")
            manifest["models"][model_name] = metadata
        except Exception as exc:
            metadata = {
                "status": "failed",
                "reason": repr(exc),
            }
            save_json(metadata, model_dir(artifacts_dir, model_name) / "metadata.json")
            manifest["models"][model_name] = metadata

    metrics_frame = metrics_to_frame(metrics_summary)
    if not metrics_frame.empty:
        metrics_frame.to_csv(artifacts_dir / "metrics_summary.csv", index=False)
        save_json(metrics_frame.to_dict(orient="records"), artifacts_dir / "metrics_summary.json")
    else:
        save_json([], artifacts_dir / "metrics_summary.json")

    save_manifest(artifacts_dir, manifest)

    output = {
        "config": config,
        "dataset": dataset_stats,
        "trained_models": list(trained_models),
        "metrics": metrics_summary,
        "artifacts_dir": str(Path(artifacts_dir).resolve()),
    }
    print("Training completed.")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
