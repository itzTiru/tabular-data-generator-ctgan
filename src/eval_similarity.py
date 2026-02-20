"""Similarity evaluation metrics for real vs synthetic data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.schema import (
    CATEGORICAL_MODELING_COLUMNS,
    MODELING_COLUMNS,
    SIMILARITY_CATEGORICAL_COLUMNS,
    SIMILARITY_NUMERIC_COLUMNS,
)


def _js_divergence(real_values: pd.Series, synthetic_values: pd.Series) -> float:
    categories = sorted(
        set(real_values.astype(str).unique()) | set(synthetic_values.astype(str).unique())
    )
    real_dist = (
        real_values.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0.0)
    )
    synth_dist = (
        synthetic_values.astype(str)
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0.0)
    )

    p = real_dist.to_numpy(dtype=float)
    q = synth_dist.to_numpy(dtype=float)
    m = 0.5 * (p + q)

    divergence = 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)
    return float(divergence)


def _real_vs_synthetic_roc_auc(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    random_seed: int,
) -> float:
    labeled_real = real_df[MODELING_COLUMNS].copy()
    labeled_real["_label"] = 1

    labeled_synth = synthetic_df[MODELING_COLUMNS].copy()
    labeled_synth["_label"] = 0

    combined = pd.concat([labeled_real, labeled_synth], ignore_index=True)
    x = combined.drop(columns=["_label"])
    y = combined["_label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_seed,
        stratify=y,
    )

    numeric_columns = [
        column for column in MODELING_COLUMNS if column not in CATEGORICAL_MODELING_COLUMNS
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_MODELING_COLUMNS,
            ),
        ]
    )

    classifier = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    classifier.fit(x_train, y_train)
    scores = classifier.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, scores))


def evaluate_similarity(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Compute required distributional similarity metrics."""
    similarity: dict[str, Any] = {
        "numeric": {},
        "categorical": {},
    }

    for column in SIMILARITY_NUMERIC_COLUMNS:
        real_values = pd.to_numeric(real_df[column], errors="coerce").dropna().to_numpy()
        synth_values = pd.to_numeric(synthetic_df[column], errors="coerce").dropna().to_numpy()

        ks_result = ks_2samp(real_values, synth_values)
        similarity["numeric"][column] = {
            "ks_statistic": float(ks_result.statistic),
            "ks_pvalue": float(ks_result.pvalue),
            "wasserstein": float(wasserstein_distance(real_values, synth_values)),
        }

    for column in SIMILARITY_CATEGORICAL_COLUMNS:
        similarity["categorical"][column] = {
            "js_divergence": _js_divergence(real_df[column], synthetic_df[column]),
        }

    real_corr = real_df[SIMILARITY_NUMERIC_COLUMNS].corr().fillna(0.0)
    synth_corr = synthetic_df[SIMILARITY_NUMERIC_COLUMNS].corr().fillna(0.0)
    correlation_drift = np.linalg.norm((real_corr - synth_corr).to_numpy(), ord="fro")

    similarity["correlation_drift_frobenius"] = float(correlation_drift)
    similarity["real_vs_synth_roc_auc"] = _real_vs_synthetic_roc_auc(
        real_df=real_df,
        synthetic_df=synthetic_df,
        random_seed=random_seed,
    )

    return similarity
