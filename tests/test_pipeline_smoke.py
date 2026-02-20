"""Fast smoke test for full pipeline execution."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytest.importorskip("sdv")

from src.pipeline import run_pipeline


def _build_smoke_dataframe() -> pd.DataFrame:
    subcategory_to_category = {
        "Phones": "Technology",
        "Chairs": "Furniture",
        "Binders": "Office Supplies",
    }
    state_to_region = {
        "California": "West",
        "Texas": "Central",
        "New York": "East",
    }
    ship_modes = ["Same Day", "First Class", "Second Class", "Standard Class"]
    segments = ["Consumer", "Corporate", "Home Office"]
    discounts = [0.0, 0.1, 0.2]
    subcategories = list(subcategory_to_category.keys())
    states = list(state_to_region.keys())

    rows: list[dict[str, object]] = []
    for index in range(36):
        train_period = index < 24
        if train_period:
            order_date = pd.Timestamp("2016-01-01") + pd.Timedelta(days=index * 5)
        else:
            order_date = pd.Timestamp("2017-01-02") + pd.Timedelta(days=(index - 24) * 5)

        ship_delay = [1, 2, 3, 4][index % 4]
        ship_date = order_date + pd.Timedelta(days=ship_delay)

        subcategory = subcategories[index % len(subcategories)]
        state = states[index % len(states)]

        rows.append(
            {
                "Row ID": index + 1,
                "Order ID": f"CA-2016-{index + 1:06d}",
                "Order Date": order_date.strftime("%m/%d/%Y"),
                "Ship Date": ship_date.strftime("%m/%d/%Y"),
                "Ship Mode": ship_modes[index % len(ship_modes)],
                "Customer ID": f"CG-{10000 + index}",
                "Customer Name": f"Customer {index}",
                "Segment": segments[index % len(segments)],
                "Country": "United States",
                "City": "Sample City",
                "State": state,
                "Postal Code": 90000 + index,
                "Region": state_to_region[state],
                "Product ID": f"PRD-{index:05d}",
                "Category": subcategory_to_category[subcategory],
                "Sub-Category": subcategory,
                "Product Name": f"Product {index}",
                "Sales": float(50 + index),
                "Quantity": int((index % 7) + 1),
                "Discount": float(discounts[index % len(discounts)]),
                "Profit": float((index % 10) - 4),
            }
        )

    return pd.DataFrame(rows)


def test_pipeline_smoke(tmp_path: Path) -> None:
    data_path = tmp_path / "smoke_dataset.csv"
    _build_smoke_dataframe().to_csv(data_path, index=False)

    config = {
        "data": {
            "default_path": str(data_path),
            "processed_dir": str(tmp_path / "data_processed"),
            "synthetic_dir": str(tmp_path / "data_synthetic"),
        },
        "artifacts": {
            "preprocess_artifacts_path": str(tmp_path / "artifacts" / "preprocess_artifacts.json"),
            "models_dir": str(tmp_path / "models"),
        },
        "reports": {
            "metrics_path": str(tmp_path / "reports" / "metrics.json"),
            "figures_dir": str(tmp_path / "reports" / "figures"),
        },
        "split": {"cutoff_date": "2017-01-01"},
        "generation": {"scales": [1]},
        "models": {
            "enabled": ["ctgan"],
            "ctgan": {"epochs": 1, "batch_size": 10, "verbose": False},
            "tvae": {"epochs": 1, "batch_size": 10, "verbose": False},
        },
        "runtime": {"random_seed": 42},
    }

    config_path = tmp_path / "smoke_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = run_pipeline(config_path=config_path, data_path_override=data_path)

    metrics_path = Path(summary["metrics_path"])
    assert metrics_path.exists()

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "results" in metrics_payload
    assert "ctgan" in metrics_payload["results"]
    assert "1x" in metrics_payload["results"]["ctgan"]

    synth_csv = Path(metrics_payload["results"]["ctgan"]["1x"]["output_files"]["synthetic_csv"])
    assert synth_csv.exists()
