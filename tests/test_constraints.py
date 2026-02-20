"""Tests for constraint enforcement and reconstruction."""

from __future__ import annotations

import pandas as pd

from src.constraints import apply_constraints_and_reconstruct, apply_hard_constraints


def test_apply_hard_constraints_and_reconstruction() -> None:
    synthetic = pd.DataFrame(
        {
            "order_day_index": [-3, 999],
            "ship_delay_days": [9, 0],
            "Ship Mode": ["Same Day", "Standard Class"],
            "Segment": ["Consumer", "Corporate"],
            "State": ["California", "Texas"],
            "Sub-Category": ["Phones", "Chairs"],
            "Sales": [-10.0, 100.0],
            "Quantity": [0, 40],
            "Discount": [0.13, 0.77],
            "Profit": [5.0, -12.0],
        }
    )

    constrained = apply_hard_constraints(
        synthetic_df=synthetic,
        max_order_day_index=100,
        discount_allowed_values=[0.0, 0.1, 0.2, 0.8],
    )

    assert constrained["order_day_index"].min() >= 0
    assert constrained["order_day_index"].max() <= 100
    assert constrained["Sales"].min() >= 0
    assert constrained["Quantity"].min() >= 1
    assert constrained["Quantity"].max() <= 14

    assert constrained.loc[0, "ship_delay_days"] == 1
    assert constrained.loc[1, "ship_delay_days"] == 3

    assert constrained.loc[0, "Discount"] == 0.1
    assert constrained.loc[1, "Discount"] == 0.8

    artifacts = {
        "min_order_date": "2016-01-01",
        "max_order_day_index": 100,
        "discount_allowed_values": [0.0, 0.1, 0.2, 0.8],
        "subcategory_to_category": {"Phones": "Technology", "Chairs": "Furniture"},
        "state_to_region": {"California": "West", "Texas": "Central"},
    }

    _, reconstructed = apply_constraints_and_reconstruct(
        synthetic_df=synthetic, artifacts=artifacts
    )

    assert (reconstructed["Country"] == "United States").all()
    assert reconstructed.loc[0, "Category"] == "Technology"
    assert reconstructed.loc[1, "Region"] == "Central"
    assert reconstructed.loc[0, "Order Date"] == "2016-01-01"
