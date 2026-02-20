"""Model training utilities for CTGAN and TVAE."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

from src.schema import (
    CATEGORICAL_MODELING_COLUMNS,
    FLOAT_MODELING_COLUMNS,
    INTEGER_MODELING_COLUMNS,
)


@dataclass
class TrainingResult:
    """Container for trained synthesizers and saved model paths."""

    models: dict[str, Any]
    model_paths: dict[str, str]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _build_metadata(train_df: pd.DataFrame) -> SingleTableMetadata:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)
    if metadata.primary_key:
        metadata.remove_primary_key()

    for column in CATEGORICAL_MODELING_COLUMNS:
        metadata.update_column(column_name=column, sdtype="categorical")

    for column in INTEGER_MODELING_COLUMNS + FLOAT_MODELING_COLUMNS:
        metadata.update_column(column_name=column, sdtype="numerical")

    return metadata


def _build_synthesizer(
    model_name: str,
    metadata: SingleTableMetadata,
    model_config: dict[str, Any],
) -> Any:
    epochs = int(model_config.get("epochs", 50))
    batch_size = int(model_config.get("batch_size", 500))
    verbose = bool(model_config.get("verbose", False))

    if model_name == "ctgan":
        return CTGANSynthesizer(
            metadata=metadata,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            cuda=False,
        )

    if model_name == "tvae":
        return TVAESynthesizer(
            metadata=metadata,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            cuda=False,
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def train_models(
    train_df: pd.DataFrame,
    models_config: dict[str, Any],
    models_dir: str | Path,
    random_seed: int,
) -> TrainingResult:
    """Train enabled synthesizers on train modeling data and save model artifacts."""
    _set_seed(random_seed)

    output_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enabled_models = [str(name).lower() for name in models_config.get("enabled", [])]
    if not enabled_models:
        raise ValueError("No models enabled in configuration.")

    metadata = _build_metadata(train_df)

    trained_models: dict[str, Any] = {}
    model_paths: dict[str, str] = {}

    for model_name in enabled_models:
        config_for_model = models_config.get(model_name, {})
        synthesizer = _build_synthesizer(model_name, metadata, config_for_model)
        with parallel_backend("threading", n_jobs=1):
            synthesizer.fit(train_df)

        model_path = output_dir / f"{model_name}.joblib"
        joblib.dump({"model_name": model_name, "model": synthesizer}, model_path)

        trained_models[model_name] = synthesizer
        model_paths[model_name] = str(model_path)

    return TrainingResult(models=trained_models, model_paths=model_paths)


def load_model(model_path: str | Path) -> Any:
    """Load a serialized synthesizer from disk."""
    payload = joblib.load(model_path)
    return payload["model"]
