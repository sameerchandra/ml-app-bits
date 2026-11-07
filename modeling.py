import pandas as pd
from numpy.typing import ArrayLike
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def calculate_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    positive_scores: ArrayLike,
    positive_label: bool | int | str,
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, positive_scores),
        "precision": precision_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def make_logistic_regression_pipeline(features: pd.DataFrame) -> Pipeline:
    numeric_columns = features.select_dtypes(include=["number"]).columns
    categorical_columns = features.columns.difference(numeric_columns)

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
