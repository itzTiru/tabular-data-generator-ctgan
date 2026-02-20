"""Preprocessing and train/test split logic."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.schema import (
    CATEGORICAL_MODELING_COLUMNS,
    DETERMINISTIC_COLUMNS,
    DROP_COLUMNS,
    FLOAT_MODELING_COLUMNS,
    INTEGER_MODELING_COLUMNS,
    MODELING_COLUMNS,
    NON_MODELED_LOCATION_COLUMNS,
    NUMERIC_MODELING_COLUMNS,
    coerce_numeric_columns,
    normalize_categorical_columns,
    validate_modeling_columns,
    validate_required_columns,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """Container for preprocessing outputs."""

    train_model: pd.DataFrame
    test_model: pd.DataFrame
    train_with_dates: pd.DataFrame
    test_with_dates: pd.DataFrame
    artifacts: dict[str, Any]


def load_and_validate_data(data_path: str | Path) -> pd.DataFrame:
    """Load CSV and validate required raw schema."""
    csv_path = Path(data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    validate_required_columns(dataframe)
    return dataframe


def _build_deterministic_mapping(
    dataframe: pd.DataFrame,
    key_column: str,
    value_column: str,
) -> dict[str, str]:
    """Build deterministic mapping using the most frequent value per key."""
    mapping: dict[str, str] = {}
    ambiguous_keys: list[str] = []

    for key, group in dataframe.groupby(key_column):
        counts = group[value_column].value_counts(dropna=False)
        resolved_value = str(counts.index[0])
        mapping[str(key)] = resolved_value
        if len(counts.index) > 1:
            ambiguous_keys.append(str(key))

    if ambiguous_keys:
        LOGGER.warning(
            "Found %d non-deterministic mappings for %s -> %s. Using mode per key.",
            len(ambiguous_keys),
            key_column,
            value_column,
        )

    return mapping


def preprocess_and_split(raw_df: pd.DataFrame, cutoff_date: str = "2017-01-01") -> PreprocessResult:
    """Transform raw dataframe into modeling tables and split by time."""
    validate_required_columns(raw_df)

    dataframe = raw_df.copy()
    dataframe["Order Date"] = pd.to_datetime(dataframe["Order Date"], errors="raise").dt.normalize()
    dataframe["Ship Date"] = pd.to_datetime(dataframe["Ship Date"], errors="raise").dt.normalize()

    min_order_date = dataframe["Order Date"].min()
    if pd.isna(min_order_date):
        raise ValueError("Order Date has no valid values.")

    dataframe["ship_delay_days"] = (dataframe["Ship Date"] - dataframe["Order Date"]).dt.days
    dataframe["order_day_index"] = (dataframe["Order Date"] - min_order_date).dt.days

    for column in NUMERIC_MODELING_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")

    for column in CATEGORICAL_MODELING_COLUMNS:
        dataframe[column] = dataframe[column].fillna("Unknown").astype(str)

    modeling_source = dataframe.drop(
        columns=DROP_COLUMNS + DETERMINISTIC_COLUMNS + NON_MODELED_LOCATION_COLUMNS
    )
    modeling_source = modeling_source.drop(columns=["Order Date", "Ship Date"])
    modeling_source = modeling_source[MODELING_COLUMNS]

    modeling_source = coerce_numeric_columns(modeling_source, NUMERIC_MODELING_COLUMNS)
    modeling_source = normalize_categorical_columns(modeling_source, CATEGORICAL_MODELING_COLUMNS)

    for column in INTEGER_MODELING_COLUMNS:
        modeling_source[column] = modeling_source[column].round().astype(int)

    for column in FLOAT_MODELING_COLUMNS:
        modeling_source[column] = modeling_source[column].astype(float)

    validate_modeling_columns(modeling_source)

    split_cutoff = pd.Timestamp(cutoff_date)
    train_mask = dataframe["Order Date"] < split_cutoff

    train_model = modeling_source.loc[train_mask].reset_index(drop=True)
    test_model = modeling_source.loc[~train_mask].reset_index(drop=True)

    with_dates_columns = MODELING_COLUMNS + ["Order Date"]
    with_dates_frame = dataframe[with_dates_columns].copy()
    train_with_dates = with_dates_frame.loc[train_mask].reset_index(drop=True)
    test_with_dates = with_dates_frame.loc[~train_mask].reset_index(drop=True)

    train_reference = dataframe.loc[train_mask].copy()
    discount_values = sorted(
        float(value) for value in train_model["Discount"].dropna().unique().tolist()
    )

    artifacts: dict[str, Any] = {
        "min_order_date": min_order_date.strftime("%Y-%m-%d"),
        "max_order_day_index": int(train_model["order_day_index"].max()),
        "discount_allowed_values": discount_values,
        "subcategory_to_category": _build_deterministic_mapping(
            train_reference,
            key_column="Sub-Category",
            value_column="Category",
        ),
        "state_to_region": _build_deterministic_mapping(
            train_reference,
            key_column="State",
            value_column="Region",
        ),
        "split_cutoff_date": split_cutoff.strftime("%Y-%m-%d"),
        "train_rows": int(len(train_model)),
        "test_rows": int(len(test_model)),
    }

    return PreprocessResult(
        train_model=train_model,
        test_model=test_model,
        train_with_dates=train_with_dates,
        test_with_dates=test_with_dates,
        artifacts=artifacts,
    )


def save_processed_outputs(result: PreprocessResult, processed_dir: str | Path) -> dict[str, str]:
    """Persist processed train/test dataframes for reproducibility."""
    output_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "train_model": output_dir / "train_model.csv",
        "test_model": output_dir / "test_model.csv",
        "train_with_dates": output_dir / "train_with_dates.csv",
        "test_with_dates": output_dir / "test_with_dates.csv",
    }

    result.train_model.to_csv(output_paths["train_model"], index=False)
    result.test_model.to_csv(output_paths["test_model"], index=False)

    train_dates = result.train_with_dates.copy()
    test_dates = result.test_with_dates.copy()
    train_dates["Order Date"] = pd.to_datetime(train_dates["Order Date"]).dt.strftime("%Y-%m-%d")
    test_dates["Order Date"] = pd.to_datetime(test_dates["Order Date"]).dt.strftime("%Y-%m-%d")
    train_dates.to_csv(output_paths["train_with_dates"], index=False)
    test_dates.to_csv(output_paths["test_with_dates"], index=False)

    return {name: str(path) for name, path in output_paths.items()}


def save_preprocess_artifacts(artifacts: dict[str, Any], artifacts_path: str | Path) -> str:
    """Save preprocessing artifacts to JSON."""
    output_path = Path(artifacts_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    return str(output_path)
