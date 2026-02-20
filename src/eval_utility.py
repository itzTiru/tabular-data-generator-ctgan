"""ML utility evaluation using TSTR-style experiments."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.schema import CATEGORICAL_MODELING_COLUMNS, MODELING_COLUMNS

UTILITY_CLASSIFIERS = {
    "logistic_regression",
    "random_forest",
}


def _build_estimator(model_name: str, random_seed: int) -> Any:
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000)

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported utility model: {model_name}")


def _evaluate_single(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_seed: int,
) -> dict[str, float]:
    estimator = _build_estimator(model_name, random_seed=random_seed)

    numeric_columns = [
        column for column in x_train.columns if column not in CATEGORICAL_MODELING_COLUMNS
    ]
    categorical_columns = [
        column for column in x_train.columns if column in CATEGORICAL_MODELING_COLUMNS
    ]

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("numeric", StandardScaler(), numeric_columns),
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore"),
                            categorical_columns,
                        ),
                    ]
                ),
            ),
            ("model", estimator),
        ]
    )

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "f1": float(f1_score(y_test, predictions)),
    }


def evaluate_utility(
    real_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    synthetic_train_df: pd.DataFrame,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Evaluate real baseline, TSTR, and augmentation utility."""
    feature_columns = [column for column in MODELING_COLUMNS if column != "Profit"]

    y_real_train = (real_train_df["Profit"] < 0).astype(int)
    y_real_test = (real_test_df["Profit"] < 0).astype(int)
    y_synth_train = (synthetic_train_df["Profit"] < 0).astype(int)

    x_real_train = real_train_df[feature_columns]
    x_real_test = real_test_df[feature_columns]
    x_synth_train = synthetic_train_df[feature_columns]

    x_augmented = pd.concat([x_real_train, x_synth_train], ignore_index=True)
    y_augmented = pd.concat([y_real_train, y_synth_train], ignore_index=True)

    scenarios = {
        "real_to_real": (x_real_train, y_real_train, x_real_test, y_real_test),
        "synthetic_to_real": (x_synth_train, y_synth_train, x_real_test, y_real_test),
        "real_plus_synthetic_to_real": (x_augmented, y_augmented, x_real_test, y_real_test),
    }

    utility: dict[str, Any] = {
        "target": "is_loss",
        "features": feature_columns,
        "models": {},
    }

    for model_name in sorted(UTILITY_CLASSIFIERS):
        utility["models"][model_name] = {}
        for scenario_name, (x_train, y_train, x_test, y_test) in scenarios.items():
            utility["models"][model_name][scenario_name] = _evaluate_single(
                model_name=model_name,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                random_seed=random_seed,
            )

    return utility
