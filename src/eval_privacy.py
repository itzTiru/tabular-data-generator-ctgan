"""Privacy risk indicator metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.schema import CATEGORICAL_MODELING_COLUMNS, MODELING_COLUMNS, QID_COLUMNS

NUMERIC_PRIVACY_COLUMNS = [
    "Sales",
    "Profit",
    "Quantity",
    "Discount",
    "ship_delay_days",
    "order_day_index",
]


def _row_tuples(df: pd.DataFrame, columns: list[str]) -> list[tuple[Any, ...]]:
    return list(df[columns].itertuples(index=False, name=None))


def _attach_order_month(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["order_month"] = (
        pd.to_datetime(output["Order Date"], errors="coerce").dt.to_period("M").astype(str)
    )
    return output


def _collision_rate(reference_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
    reference_month = _attach_order_month(reference_df)
    synthetic_month = _attach_order_month(synthetic_df)

    reference_set = set(_row_tuples(reference_month, QID_COLUMNS))
    synth_rows = _row_tuples(synthetic_month, QID_COLUMNS)
    if not synth_rows:
        return 0.0

    collisions = sum(1 for row in synth_rows if row in reference_set)
    return float(collisions / len(synth_rows))


def _nearest_neighbor_summary(
    real_train_model: pd.DataFrame, synth_model: pd.DataFrame
) -> dict[str, Any]:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_PRIVACY_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_MODELING_COLUMNS),
        ]
    )

    real_features = preprocessor.fit_transform(real_train_model)
    synth_features = preprocessor.transform(synth_model)

    synth_to_real_nn = NearestNeighbors(n_neighbors=1)
    synth_to_real_nn.fit(real_features)
    synth_distances, _ = synth_to_real_nn.kneighbors(synth_features)

    real_to_real_nn = NearestNeighbors(n_neighbors=2)
    real_to_real_nn.fit(real_features)
    real_distances, _ = real_to_real_nn.kneighbors(real_features)

    synth_dist_values = synth_distances[:, 0]
    real_baseline_values = real_distances[:, 1]

    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    return {
        "quantiles": quantiles,
        "synthetic_to_real": [
            float(np.quantile(synth_dist_values, quantile)) for quantile in quantiles
        ],
        "real_to_real_baseline": [
            float(np.quantile(real_baseline_values, quantile)) for quantile in quantiles
        ],
    }


def evaluate_privacy(
    real_train_with_dates: pd.DataFrame,
    real_test_with_dates: pd.DataFrame,
    synthetic_final_df: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate exact duplicates, QID collisions, and nearest-neighbor indicators."""
    real_all_model = pd.concat(
        [real_train_with_dates[MODELING_COLUMNS], real_test_with_dates[MODELING_COLUMNS]],
        ignore_index=True,
    )
    synthetic_model = synthetic_final_df[MODELING_COLUMNS].copy()

    real_set = set(_row_tuples(real_all_model, MODELING_COLUMNS))
    synthetic_rows = _row_tuples(synthetic_model, MODELING_COLUMNS)

    duplicate_count = sum(1 for row in synthetic_rows if row in real_set)
    duplicate_rate = float(duplicate_count / len(synthetic_rows)) if synthetic_rows else 0.0

    privacy = {
        "exact_duplicate_rate": duplicate_rate,
        "qid_collision_rate_train_real": _collision_rate(real_train_with_dates, synthetic_final_df),
        "qid_collision_rate_test_real": _collision_rate(real_test_with_dates, synthetic_final_df),
        "nearest_neighbor_distance": _nearest_neighbor_summary(
            real_train_model=real_train_with_dates[MODELING_COLUMNS],
            synth_model=synthetic_model,
        ),
    }

    return privacy
