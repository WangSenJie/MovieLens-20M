from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import ensure_artifact_layout, load_manifest, load_model_artifact, load_shared_artifacts, save_json
from .evaluation import evaluate_model, metrics_to_frame
from .models import ALL_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained MovieLens recommenders.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--models", type=str, default="all")
    return parser.parse_args()


def resolve_models(raw_models: str, manifest: dict) -> list[str]:
    if raw_models == "all":
        return [name for name in ALL_MODELS if manifest["models"].get(name, {}).get("status") == "trained"]
    return [name.strip().lower() for name in raw_models.split(",") if name.strip()]


def main() -> None:
    args = parse_args()
    artifacts_dir = ensure_artifact_layout(args.artifacts_dir)
    manifest = load_manifest(artifacts_dir)
    shared = load_shared_artifacts(artifacts_dir)
    model_names = resolve_models(args.models, manifest)

    results = {}
    for model_name in model_names:
        status = manifest["models"].get(model_name, {}).get("status")
        if status != "trained":
            continue
        model = load_model_artifact(artifacts_dir, model_name)
        metrics, _ = evaluate_model(
            model=model,
            user_history=shared.user_history,
            test_targets=shared.test_targets,
            item_to_index=model.item_to_index,
        )
        results[model_name] = metrics

    frame = metrics_to_frame(results)
    frame.to_csv(artifacts_dir / "metrics_summary.csv", index=False)
    save_json(frame.to_dict(orient="records"), artifacts_dir / "metrics_summary.json")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
