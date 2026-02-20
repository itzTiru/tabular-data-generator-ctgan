"""End-to-end pipeline orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.constraints import apply_constraints_and_reconstruct
from src.eval_privacy import evaluate_privacy
from src.eval_similarity import evaluate_similarity
from src.eval_utility import evaluate_utility
from src.generate import generate_for_scales
from src.preprocess import (
    load_and_validate_data,
    preprocess_and_split,
    save_preprocess_artifacts,
    save_processed_outputs,
)
from src.report import generate_figures, save_metrics
from src.train import train_models


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load pipeline config from YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config path does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a mapping at top level.")

    return config


def _ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    processed_dir = Path(config["data"]["processed_dir"])
    synthetic_dir = Path(config["data"]["synthetic_dir"])
    models_dir = Path(config["artifacts"]["models_dir"])
    figures_dir = Path(config["reports"]["figures_dir"])

    for path in (processed_dir, synthetic_dir, models_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "processed_dir": processed_dir,
        "synthetic_dir": synthetic_dir,
        "models_dir": models_dir,
        "figures_dir": figures_dir,
    }


def run_pipeline(
    config_path: str | Path,
    data_path_override: str | Path | None = None,
) -> dict[str, Any]:
    """Run preprocessing, training, generation, constraints, evaluation, and reporting."""
    config = load_config(config_path)
    output_dirs = _ensure_output_dirs(config)

    data_path = (
        Path(data_path_override) if data_path_override else Path(config["data"]["default_path"])
    )
    raw_df = load_and_validate_data(data_path)

    preprocess_result = preprocess_and_split(
        raw_df,
        cutoff_date=str(config["split"]["cutoff_date"]),
    )

    processed_outputs = save_processed_outputs(
        preprocess_result,
        processed_dir=output_dirs["processed_dir"],
    )
    preprocess_artifacts_path = save_preprocess_artifacts(
        preprocess_result.artifacts,
        artifacts_path=config["artifacts"]["preprocess_artifacts_path"],
    )

    random_seed = int(config["runtime"].get("random_seed", 42))
    training_result = train_models(
        train_df=preprocess_result.train_model,
        models_config=config["models"],
        models_dir=output_dirs["models_dir"],
        random_seed=random_seed,
    )

    scales = [int(scale) for scale in config["generation"]["scales"]]
    generated = generate_for_scales(
        trained_models=training_result.models,
        scales=scales,
        train_size=len(preprocess_result.train_model),
        processed_dir=output_dirs["processed_dir"],
    )

    results: dict[str, Any] = {}

    for model_name, scale_payload in generated.items():
        results[model_name] = {}

        for scale, synthetic_raw_modeling in scale_payload.items():
            constrained_modeling, synthetic_final = apply_constraints_and_reconstruct(
                synthetic_df=synthetic_raw_modeling,
                artifacts=preprocess_result.artifacts,
            )

            constrained_path = (
                output_dirs["processed_dir"] / f"synth_constrained_{model_name}_{scale}x.csv"
            )
            synthetic_path = output_dirs["synthetic_dir"] / f"{model_name}_{scale}x.csv"

            constrained_modeling.to_csv(constrained_path, index=False)
            synthetic_final.to_csv(synthetic_path, index=False)

            similarity_metrics = evaluate_similarity(
                real_df=preprocess_result.train_model,
                synthetic_df=constrained_modeling,
                random_seed=random_seed,
            )
            utility_metrics = evaluate_utility(
                real_train_df=preprocess_result.train_model,
                real_test_df=preprocess_result.test_model,
                synthetic_train_df=constrained_modeling,
                random_seed=random_seed,
            )
            privacy_metrics = evaluate_privacy(
                real_train_with_dates=preprocess_result.train_with_dates,
                real_test_with_dates=preprocess_result.test_with_dates,
                synthetic_final_df=synthetic_final,
            )

            scale_label = f"{scale}x"
            results[model_name][scale_label] = {
                "similarity": similarity_metrics,
                "utility": utility_metrics,
                "privacy": privacy_metrics,
                "output_files": {
                    "synthetic_csv": str(synthetic_path),
                    "constrained_modeling_csv": str(constrained_path),
                },
            }

    metrics_payload = {
        "run_metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "config_path": str(config_path),
            "data_path": str(data_path),
            "train_rows": int(len(preprocess_result.train_model)),
            "test_rows": int(len(preprocess_result.test_model)),
            "models": list(results.keys()),
            "scales": [f"{scale}x" for scale in scales],
        },
        "preprocess_artifacts": preprocess_result.artifacts,
        "results": results,
    }

    generate_figures(metrics_payload=metrics_payload, figures_dir=output_dirs["figures_dir"])

    metrics_path = save_metrics(
        metrics=metrics_payload,
        metrics_path=config["reports"]["metrics_path"],
    )

    summary = {
        "metrics_path": metrics_path,
        "processed_outputs": processed_outputs,
        "preprocess_artifacts_path": preprocess_artifacts_path,
        "model_paths": training_result.model_paths,
        "synthetic_outputs": {
            model_name: {
                scale_label: run_metrics["output_files"]["synthetic_csv"]
                for scale_label, run_metrics in scale_metrics.items()
            }
            for model_name, scale_metrics in results.items()
        },
    }

    return summary


def run_pipeline_from_config(
    config_path: str | Path, data_path_override: str | Path | None = None
) -> dict[str, Any]:
    """Convenience wrapper for external callers."""
    return run_pipeline(config_path=config_path, data_path_override=data_path_override)
