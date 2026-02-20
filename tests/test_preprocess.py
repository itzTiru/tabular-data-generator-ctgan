"""Tests for preprocessing behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocess import preprocess_and_split
from src.schema import MODELING_COLUMNS


def _sample_raw_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Row ID": 1,
                "Order ID": "CA-2016-000001",
                "Order Date": "12/29/2016",
                "Ship Date": "12/31/2016",
                "Ship Mode": "Second Class",
                "Customer ID": "CG-10000",
                "Customer Name": "Alice",
                "Segment": "Consumer",
                "Country": "United States",
                "City": "Los Angeles",
                "State": "California",
                "Postal Code": 90001,
                "Region": "West",
                "Product ID": "TEC-PH-0001",
                "Category": "Technology",
                "Sub-Category": "Phones",
                "Product Name": "Phone A",
                "Sales": 100.0,
                "Quantity": 2,
                "Discount": 0.1,
                "Profit": 20.0,
            },
            {
                "Row ID": 2,
                "Order ID": "CA-2016-000002",
                "Order Date": "12/30/2016",
                "Ship Date": "1/2/2017",
                "Ship Mode": "Standard Class",
                "Customer ID": "CG-10001",
                "Customer Name": "Bob",
                "Segment": "Corporate",
                "Country": "United States",
                "City": "San Francisco",
                "State": "California",
                "Postal Code": 94105,
                "Region": "West",
                "Product ID": "FUR-CH-0002",
                "Category": "Furniture",
                "Sub-Category": "Chairs",
                "Product Name": "Chair B",
                "Sales": 250.0,
                "Quantity": 3,
                "Discount": 0.2,
                "Profit": -5.0,
            },
            {
                "Row ID": 3,
                "Order ID": "CA-2017-000003",
                "Order Date": "1/5/2017",
                "Ship Date": "1/6/2017",
                "Ship Mode": "Same Day",
                "Customer ID": "CG-10002",
                "Customer Name": "Cara",
                "Segment": "Home Office",
                "Country": "United States",
                "City": "Dallas",
                "State": "Texas",
                "Postal Code": 75001,
                "Region": "Central",
                "Product ID": "OFF-BI-0003",
                "Category": "Office Supplies",
                "Sub-Category": "Binders",
                "Product Name": "Binder C",
                "Sales": 80.0,
                "Quantity": 4,
                "Discount": 0.0,
                "Profit": 10.0,
            },
            {
                "Row ID": 4,
                "Order ID": "CA-2017-000004",
                "Order Date": "1/8/2017",
                "Ship Date": "1/11/2017",
                "Ship Mode": "First Class",
                "Customer ID": "CG-10003",
                "Customer Name": "Dylan",
                "Segment": "Consumer",
                "Country": "United States",
                "City": "Austin",
                "State": "Texas",
                "Postal Code": 73301,
                "Region": "Central",
                "Product ID": "TEC-AC-0004",
                "Category": "Technology",
                "Sub-Category": "Accessories",
                "Product Name": "Accessory D",
                "Sales": 45.0,
                "Quantity": 1,
                "Discount": 0.0,
                "Profit": -2.0,
            },
        ]
    )


def test_preprocess_and_split_outputs_expected_schema() -> None:
    raw_df = _sample_raw_dataframe()

    result = preprocess_and_split(raw_df, cutoff_date="2017-01-01")

    assert list(result.train_model.columns) == MODELING_COLUMNS
    assert list(result.test_model.columns) == MODELING_COLUMNS

    assert len(result.train_model) == 2
    assert len(result.test_model) == 2

    assert result.artifacts["min_order_date"] == "2016-12-29"
    assert result.artifacts["max_order_day_index"] >= 1
    assert result.artifacts["subcategory_to_category"]["Phones"] == "Technology"
    assert result.artifacts["state_to_region"]["California"] == "West"


def test_preprocess_rejects_missing_required_columns() -> None:
    raw_df = _sample_raw_dataframe().drop(columns=["Profit"])

    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess_and_split(raw_df, cutoff_date="2017-01-01")
