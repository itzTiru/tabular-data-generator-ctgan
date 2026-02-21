"""Streamlit dashboard for synthetic data evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

METRICS_PATH = Path("reports/metrics.json")
SYNTHETIC_DIR = Path("data/synthetic")


def _load_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}

    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _utility_to_frame(utility_payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, scenarios in utility_payload.get("models", {}).items():
        for scenario_name, metrics in scenarios.items():
            rows.append(
                {
                    "classifier": model_name,
                    "scenario": scenario_name,
                    "roc_auc": metrics.get("roc_auc"),
                    "f1": metrics.get("f1"),
                }
            )

    return pd.DataFrame(rows)


def _privacy_quantile_frame(privacy_payload: dict[str, Any]) -> pd.DataFrame:
    nearest = privacy_payload.get("nearest_neighbor_distance", {})
    quantiles = nearest.get("quantiles", [])
    synth = nearest.get("synthetic_to_real", [])
    baseline = nearest.get("real_to_real_baseline", [])

    rows = []
    for index, quantile in enumerate(quantiles):
        rows.append(
            {
                "quantile": quantile,
                "synthetic_to_real": synth[index] if index < len(synth) else None,
                "real_to_real_baseline": baseline[index] if index < len(baseline) else None,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Tabular Synthetic Data Dashboard", layout="wide")
    st.title("Tabular Synthetic Data Generator Dashboard")

    metrics_payload = _load_metrics(METRICS_PATH)
    if not metrics_payload:
        st.warning("No metrics found. Run the pipeline first to generate reports/metrics.json.")
        st.stop()

    results = metrics_payload.get("results", {})
    if not results:
        st.warning("No model results found in metrics payload.")
        st.stop()

    model_options = sorted(results.keys())
    selected_model = st.sidebar.selectbox("Model", model_options)

    scale_options = sorted(
        results[selected_model].keys(),
        key=lambda item: int(item.rstrip("x")),
    )
    selected_scale = st.sidebar.selectbox("Scale", scale_options)

    available_csv_files = sorted([path.name for path in SYNTHETIC_DIR.glob("*.csv")])
    default_csv_name = Path(
        results[selected_model][selected_scale].get("output_files", {}).get("synthetic_csv", "")
    ).name

    selected_output_file = ""
    if available_csv_files:
        default_index = 0
        if default_csv_name in available_csv_files:
            default_index = available_csv_files.index(default_csv_name)

        selected_output_file = st.sidebar.selectbox(
            "Output File",
            options=available_csv_files,
            index=default_index,
        )
    else:
        st.sidebar.warning("No synthetic CSV files found in data/synthetic.")

    run_metrics = results[selected_model][selected_scale]

    st.subheader("Similarity Metrics")
    similarity = run_metrics.get("similarity", {})
    numeric_table = pd.DataFrame.from_dict(similarity.get("numeric", {}), orient="index")
    categorical_table = pd.DataFrame.from_dict(similarity.get("categorical", {}), orient="index")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Numeric Distribution Metrics")
        st.dataframe(numeric_table, use_container_width=True)
    with col2:
        st.write("Categorical Distribution Metrics")
        st.dataframe(categorical_table, use_container_width=True)

    st.metric("Correlation Drift (Frobenius)", similarity.get("correlation_drift_frobenius"))
    st.metric("Real-vs-Synthetic ROC-AUC", similarity.get("real_vs_synth_roc_auc"))

    st.subheader("Utility Metrics")
    utility_df = _utility_to_frame(run_metrics.get("utility", {}))
    st.dataframe(utility_df, use_container_width=True)

    st.subheader("Privacy Metrics")
    privacy = run_metrics.get("privacy", {})

    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("Exact Duplicate Rate", privacy.get("exact_duplicate_rate"))
    pcol2.metric("QID Collision (Train)", privacy.get("qid_collision_rate_train_real"))
    pcol3.metric("QID Collision (Test)", privacy.get("qid_collision_rate_test_real"))

    st.write("Nearest Neighbor Distance Quantiles")
    st.dataframe(_privacy_quantile_frame(privacy), use_container_width=True)

    st.subheader("Generated Plots")
    output_files = run_metrics.get("output_files", {})
    for key in ["similarity_plot", "utility_plot", "privacy_plot"]:
        plot_path = output_files.get(key)
        if plot_path and Path(plot_path).exists():
            st.image(plot_path, caption=key.replace("_", " ").title(), use_container_width=True)

    if not available_csv_files:
        st.warning("No synthetic CSV files found in data/synthetic.")
        return

    selected_path = SYNTHETIC_DIR / selected_output_file
    if selected_path.exists():
        preview_df = pd.read_csv(selected_path)
        st.subheader("Selected Synthetic Output Preview")
        st.dataframe(preview_df.head(100), use_container_width=True)

        st.download_button(
            label="Download Synthetic CSV",
            data=selected_path.read_bytes(),
            file_name=selected_path.name,
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
