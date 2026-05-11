"""
Train a baseline model to predict activity spikes.
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config


FEATURE_COLUMNS = [
    "volume",
    "liquidity",
    "volume_24h",
    "volume_1wk",
    "volume_1mo",
    "spread",
    "one_day_price_change",
    "one_week_price_change",
    "one_month_price_change",
    "last_trade_price",
    "best_bid",
    "best_ask",
    "competitive",
    "market_age_days",
    "time_to_resolution_days",
]

TARGET_COLUMN = "activity_spike"

config = load_config()

LABELED_DATA_PATH = Path(config["files"]["labeled_snapshot"])
MODEL_PATH = Path(config["files"]["model_artifact"])
TEST_SIZE = config["model"]["test_size"]
RANDOM_STATE = config["model"]["random_state"]
EXPERIMENT_NAME = config["model"]["experiment_name"]


def load_labeled_data(path: Path = LABELED_DATA_PATH) -> pd.DataFrame:
    """Load labeled training data."""
    return pd.read_csv(path)


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into model features and target."""
    X = df[FEATURE_COLUMNS].fillna(0)
    y = df[TARGET_COLUMN]

    return X, y


def train_logistic_regression_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """Train a logistic regression baseline model."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train a random forest model."""
    model = Pipeline(
        steps=[
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


def train_lightgbm_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train a LightGBM model."""
    model = Pipeline(
        steps=[
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train an XGBoost model."""
    model = Pipeline(
        steps=[
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Evaluate model performance on a holdout set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nMetrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    return metrics


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> Path:
    """Save trained model artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

    return path


def log_mlflow_run(
    model: Pipeline,
    metrics: dict[str, float],
    model_name: str,
    model_params: dict[str, int | str | float],
    train_rows: int,
    test_rows: int,
    spike_rate: float,
) -> None:
    """Log training metadata, metrics, and model artifact to MLflow."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.log_param("train_rows", train_rows)
        mlflow.log_param("test_rows", test_rows)
        mlflow.log_metric("spike_rate", spike_rate)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(model, name="model")


if __name__ == "__main__":
    labeled_data = load_labeled_data()
    X, y = split_features_and_target(labeled_data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    models_to_train = {
        "logistic-regression-baseline": {
            "model": train_logistic_regression_model(X_train, y_train),
            "params": {
                "model_type": "logistic_regression",
                "class_weight": "balanced",
                "max_iter": 1000,
            },
        },
        "random-forest-baseline": {
            "model": train_random_forest_model(X_train, y_train),
            "params": {
                "model_type": "random_forest",
                "n_estimators": 200,
                "class_weight": "balanced",
            },
        },
        "lightgbm-baseline": {
            "model": train_lightgbm_model(X_train, y_train),
            "params": {
                "model_type": "lightgbm",
                "n_estimators": 200,
                "learning_rate": 0.05,
                "class_weight": "balanced",
            },
        },
        "xgboost-baseline": {
            "model": train_xgboost_model(X_train, y_train),
            "params": {
                "model_type": "xgboost",
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "logloss",
            },
        },
    }

    best_model_name = None
    best_model = None
    best_metrics = None
    best_score = -1.0

    for model_name, model_info in models_to_train.items():
        print(f"\nEvaluating {model_name}")

        model = model_info["model"]
        metrics = evaluate_model(model, X_test, y_test)

        log_mlflow_run(
            model=model,
            metrics=metrics,
            model_name=model_name,
            model_params=model_info["params"],
            train_rows=len(X_train),
            test_rows=len(X_test),
            spike_rate=float(y.mean()),
        )

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model_name = model_name
            best_model = model
            best_metrics = metrics

    saved_path = save_model(best_model)

    print(f"\nBest model: {best_model_name}")
    print(f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"Saved best model to {saved_path}")
