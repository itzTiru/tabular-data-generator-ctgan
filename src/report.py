"""Reporting helpers for metrics and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def save_metrics(metrics: dict[str, Any], metrics_path: str | Path) -> str:
    """Save metrics payload as JSON."""
    output_path = Path(metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return str(output_path)


def _plot_similarity(similarity: dict[str, Any], output_path: Path) -> None:
    numeric = similarity.get("numeric", {})
    categorical = similarity.get("categorical", {})

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    if numeric:
        columns = list(numeric.keys())
        ks_values = [numeric[column]["ks_statistic"] for column in columns]
        axes[0].bar(columns, ks_values)
        axes[0].set_title("KS Statistic by Numeric Column")
        axes[0].tick_params(axis="x", rotation=45)

    if categorical:
        columns = list(categorical.keys())
        js_values = [categorical[column]["js_divergence"] for column in columns]
        axes[1].bar(columns, js_values)
        axes[1].set_title("Jensen-Shannon Divergence by Categorical Column")
        axes[1].tick_params(axis="x", rotation=45)

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _plot_utility(utility: dict[str, Any], output_path: Path) -> None:
    rows: list[tuple[str, float]] = []
    model_metrics = utility.get("models", {})

    for model_name, scenarios in model_metrics.items():
        for scenario, values in scenarios.items():
            rows.append((f"{model_name}:{scenario}", float(values["roc_auc"])))

    labels = [row[0] for row in rows]
    roc_auc_values = [row[1] for row in rows]

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(labels, roc_auc_values)
    axis.set_title("Utility ROC-AUC")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _plot_privacy(privacy: dict[str, Any], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = [
        ("exact_duplicate_rate", float(privacy.get("exact_duplicate_rate", 0.0))),
        ("qid_train", float(privacy.get("qid_collision_rate_train_real", 0.0))),
        ("qid_test", float(privacy.get("qid_collision_rate_test_real", 0.0))),
    ]

    axes[0].bar([item[0] for item in bars], [item[1] for item in bars])
    axes[0].set_title("Privacy Risk Rates")

    nn = privacy.get("nearest_neighbor_distance", {})
    quantiles = nn.get("quantiles", [])
    synth = nn.get("synthetic_to_real", [])
    baseline = nn.get("real_to_real_baseline", [])

    if quantiles and synth and baseline:
        axes[1].plot(quantiles, synth, marker="o", label="Synthetic to Real")
        axes[1].plot(quantiles, baseline, marker="o", label="Real to Real Baseline")
        axes[1].set_title("Nearest Neighbor Distance Quantiles")
        axes[1].set_xlabel("Quantile")
        axes[1].set_ylabel("Distance")
        axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def generate_figures(metrics_payload: dict[str, Any], figures_dir: str | Path) -> None:
    """Generate plot files for each model-scale result."""
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = metrics_payload.get("results", {})

    for model_name, scales in results.items():
        for scale_label, run_metrics in scales.items():
            suffix = f"{model_name}_{scale_label}"

            similarity_path = output_dir / f"similarity_{suffix}.png"
            utility_path = output_dir / f"utility_{suffix}.png"
            privacy_path = output_dir / f"privacy_{suffix}.png"

            _plot_similarity(run_metrics["similarity"], similarity_path)
            _plot_utility(run_metrics["utility"], utility_path)
            _plot_privacy(run_metrics["privacy"], privacy_path)

            run_metrics.setdefault("output_files", {})
            run_metrics["output_files"]["similarity_plot"] = str(similarity_path)
            run_metrics["output_files"]["utility_plot"] = str(utility_path)
            run_metrics["output_files"]["privacy_plot"] = str(privacy_path)
