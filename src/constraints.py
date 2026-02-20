"""Constraint enforcement and deterministic reconstruction logic."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.schema import MODELING_COLUMNS, SHIP_MODE_DELAY_BOUNDS

FINAL_OUTPUT_COLUMNS = [
    "Order Date",
    "Ship Date",
    "Ship Mode",
    "Segment",
    "Country",
    "State",
    "Region",
    "Category",
    "Sub-Category",
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
    "order_day_index",
    "ship_delay_days",
]


def _snap_to_allowed(values: pd.Series, allowed_values: Sequence[float]) -> pd.Series:
    if not allowed_values:
        raise ValueError("allowed_values must not be empty.")

    allowed = np.asarray(sorted(float(value) for value in allowed_values), dtype=float)
    raw = pd.to_numeric(values, errors="coerce").fillna(allowed[0]).to_numpy(dtype=float)

    nearest_indices = np.abs(raw[:, None] - allowed[None, :]).argmin(axis=1)
    snapped = allowed[nearest_indices]
    return pd.Series(snapped, index=values.index)


def apply_hard_constraints(
    synthetic_df: pd.DataFrame,
    max_order_day_index: int,
    discount_allowed_values: Sequence[float],
) -> pd.DataFrame:
    """Enforce required hard constraints on synthetic modeling columns."""
    constrained = synthetic_df.copy()

    constrained["order_day_index"] = pd.to_numeric(
        constrained["order_day_index"],
        errors="coerce",
    ).fillna(0)
    constrained["order_day_index"] = constrained["order_day_index"].round().astype(int)
    constrained["order_day_index"] = constrained["order_day_index"].clip(
        lower=0, upper=max_order_day_index
    )

    constrained["ship_delay_days"] = pd.to_numeric(
        constrained["ship_delay_days"],
        errors="coerce",
    ).fillna(0)
    constrained["ship_delay_days"] = constrained["ship_delay_days"].round().astype(int)
    constrained["ship_delay_days"] = constrained["ship_delay_days"].clip(lower=0, upper=7)

    constrained["Quantity"] = pd.to_numeric(constrained["Quantity"], errors="coerce").fillna(1)
    constrained["Quantity"] = constrained["Quantity"].round().astype(int)
    constrained["Quantity"] = constrained["Quantity"].clip(lower=1, upper=14)

    constrained["Sales"] = pd.to_numeric(constrained["Sales"], errors="coerce").fillna(0.0)
    constrained["Sales"] = constrained["Sales"].clip(lower=0.0)

    constrained["Discount"] = _snap_to_allowed(constrained["Discount"], discount_allowed_values)
    constrained["Discount"] = constrained["Discount"].astype(float)

    constrained["Profit"] = (
        pd.to_numeric(constrained["Profit"], errors="coerce").fillna(0.0).astype(float)
    )

    constrained["Ship Mode"] = constrained["Ship Mode"].fillna("Standard Class").astype(str)
    constrained["Segment"] = constrained["Segment"].fillna("Unknown").astype(str)
    constrained["State"] = constrained["State"].fillna("Unknown").astype(str)
    constrained["Sub-Category"] = constrained["Sub-Category"].fillna("Unknown").astype(str)

    for ship_mode, (low, high) in SHIP_MODE_DELAY_BOUNDS.items():
        mode_mask = constrained["Ship Mode"] == ship_mode
        constrained.loc[mode_mask, "ship_delay_days"] = constrained.loc[
            mode_mask,
            "ship_delay_days",
        ].clip(lower=low, upper=high)

    constrained = constrained[MODELING_COLUMNS]
    return constrained


def reconstruct_columns(
    constrained_df: pd.DataFrame,
    min_order_date: str,
    subcategory_to_category: dict[str, str],
    state_to_region: dict[str, str],
) -> pd.DataFrame:
    """Reconstruct deterministic and date columns from constrained synthetic data."""
    reconstructed = constrained_df.copy()

    base_date = pd.Timestamp(min_order_date)
    reconstructed["Order Date"] = reconstructed["order_day_index"].apply(
        lambda value: base_date + timedelta(days=int(value))
    )
    reconstructed["Ship Date"] = reconstructed["Order Date"] + pd.to_timedelta(
        reconstructed["ship_delay_days"],
        unit="D",
    )

    reconstructed["Country"] = "United States"
    reconstructed["Category"] = (
        reconstructed["Sub-Category"].map(subcategory_to_category).fillna("Unknown")
    )
    reconstructed["Region"] = reconstructed["State"].map(state_to_region).fillna("Unknown")

    reconstructed["Order Date"] = pd.to_datetime(reconstructed["Order Date"]).dt.strftime(
        "%Y-%m-%d"
    )
    reconstructed["Ship Date"] = pd.to_datetime(reconstructed["Ship Date"]).dt.strftime("%Y-%m-%d")

    return reconstructed[FINAL_OUTPUT_COLUMNS].copy()


def apply_constraints_and_reconstruct(
    synthetic_df: pd.DataFrame,
    artifacts: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply hard constraints and deterministic reconstruction."""
    constrained = apply_hard_constraints(
        synthetic_df=synthetic_df,
        max_order_day_index=int(artifacts["max_order_day_index"]),
        discount_allowed_values=[float(value) for value in artifacts["discount_allowed_values"]],
    )

    reconstructed = reconstruct_columns(
        constrained_df=constrained,
        min_order_date=str(artifacts["min_order_date"]),
        subcategory_to_category={
            str(key): str(value)
            for key, value in dict(artifacts["subcategory_to_category"]).items()
        },
        state_to_region={
            str(key): str(value) for key, value in dict(artifacts["state_to_region"]).items()
        },
    )

    return constrained, reconstructed
