"""Synthetic data sampling utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.schema import MODELING_COLUMNS


def sample_synthetic_rows(model: Any, num_rows: int) -> pd.DataFrame:
    """Sample synthetic rows from a trained SDV synthesizer."""
    if num_rows <= 0:
        raise ValueError("num_rows must be positive.")

    sampled = model.sample(num_rows=num_rows)

    missing = [column for column in MODELING_COLUMNS if column not in sampled.columns]
    if missing:
        raise ValueError(f"Sampled data is missing modeling columns: {missing}")

    return sampled[MODELING_COLUMNS].copy()


def generate_for_scales(
    trained_models: Mapping[str, Any],
    scales: Sequence[int],
    train_size: int,
    processed_dir: str | Path,
) -> dict[str, dict[int, pd.DataFrame]]:
    """Generate synthetic modeling datasets for each model and scale."""
    output_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, dict[int, pd.DataFrame]] = {}

    for model_name, model in trained_models.items():
        generated[model_name] = {}

        for scale in scales:
            rows_to_generate = int(scale) * int(train_size)
            synthetic_modeling_df = sample_synthetic_rows(model=model, num_rows=rows_to_generate)

            raw_output_path = output_dir / f"synth_raw_{model_name}_{scale}x.csv"
            synthetic_modeling_df.to_csv(raw_output_path, index=False)

            generated[model_name][int(scale)] = synthetic_modeling_df

    return generated
