"""Schema definitions and validation utilities."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

RAW_REQUIRED_COLUMNS = [
    "Row ID",
    "Order ID",
    "Order Date",
    "Ship Date",
    "Ship Mode",
    "Customer ID",
    "Customer Name",
    "Segment",
    "Country",
    "City",
    "State",
    "Postal Code",
    "Region",
    "Product ID",
    "Category",
    "Sub-Category",
    "Product Name",
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
]

DROP_COLUMNS = ["Row ID", "Order ID", "Customer Name", "Product Name", "Customer ID", "Product ID"]
DETERMINISTIC_COLUMNS = ["Country", "Category", "Region"]
NON_MODELED_LOCATION_COLUMNS = ["City", "Postal Code"]

MODELING_COLUMNS = [
    "order_day_index",
    "ship_delay_days",
    "Ship Mode",
    "Segment",
    "State",
    "Sub-Category",
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
]

NUMERIC_MODELING_COLUMNS = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
    "ship_delay_days",
    "order_day_index",
]
INTEGER_MODELING_COLUMNS = ["order_day_index", "ship_delay_days", "Quantity"]
FLOAT_MODELING_COLUMNS = ["Sales", "Discount", "Profit"]
CATEGORICAL_MODELING_COLUMNS = ["Ship Mode", "Segment", "State", "Sub-Category"]

SIMILARITY_NUMERIC_COLUMNS = [
    "Sales",
    "Profit",
    "Quantity",
    "Discount",
    "ship_delay_days",
    "order_day_index",
]
SIMILARITY_CATEGORICAL_COLUMNS = ["Ship Mode", "Segment", "State", "Sub-Category"]

SHIP_MODE_DELAY_BOUNDS: dict[str, tuple[int, int]] = {
    "Same Day": (0, 1),
    "First Class": (1, 4),
    "Second Class": (1, 5),
    "Standard Class": (3, 7),
}

QID_COLUMNS = [
    "State",
    "Sub-Category",
    "Segment",
    "Ship Mode",
    "Discount",
    "Quantity",
    "order_month",
]


def validate_required_columns(df: pd.DataFrame) -> None:
    """Validate that the raw dataframe contains all required columns."""
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_modeling_columns(df: pd.DataFrame) -> None:
    """Validate exact modeling columns and order."""
    if list(df.columns) != MODELING_COLUMNS:
        raise ValueError(
            f"Modeling columns mismatch. Expected {MODELING_COLUMNS}, got {list(df.columns)}"
        )


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy of dataframe with selected columns coerced to numeric."""
    output = df.copy()
    for column in columns:
        output[column] = pd.to_numeric(output[column], errors="raise")
    return output


def normalize_categorical_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy of dataframe with selected columns normalized as strings."""
    output = df.copy()
    for column in columns:
        output[column] = output[column].fillna("Unknown").astype(str)
    return output
