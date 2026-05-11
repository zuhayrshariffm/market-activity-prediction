"""
Run the offline market activity prediction workflow.

This script executes the main local pipeline steps in order:
1. fetch raw market data
2. build processed market snapshot
3. create labels
4. train and compare models
5. score markets
6. export feature importance
7. summarize prediction logs if available
"""

from pathlib import Path

from src.evaluation.feature_importance import (
    build_feature_importance,
    load_model as load_feature_importance_model,
    save_feature_importance,
)
from src.features.build_market_snapshot import (
    build_market_snapshot,
    get_latest_raw_file,
    load_raw_markets,
    save_market_snapshot,
)
from src.ingestion.fetch_markets import fetch_and_save_markets
from src.inference.score_markets import (
    load_model as load_scoring_model,
    load_market_snapshot,
    save_predictions,
    score_markets,
)
from src.labeling.create_labels import (
    add_activity_spike_label,
    load_market_snapshot as load_unlabeled_snapshot,
    save_labeled_data,
)
from src.models.train_model import (
    evaluate_model,
    load_labeled_data,
    log_mlflow_run,
    save_model,
    split_features_and_target,
    train_logistic_regression_model,
    train_random_forest_model,
)
from src.monitoring.summarize_predictions import (
    load_prediction_log,
    summarize_predictions,
)

from sklearn.model_selection import train_test_split

from src.models.train_model import RANDOM_STATE, TEST_SIZE


def run_ingestion() -> Path:
    """Fetch raw market data."""
    print("\n[1/7] Fetching raw market data")
    return fetch_and_save_markets(limit=100)


def run_feature_building() -> Path:
    """Build processed market snapshot."""
    print("\n[2/7] Building market snapshot")
    raw_path = get_latest_raw_file()
    raw_markets = load_raw_markets(raw_path)
    snapshot = build_market_snapshot(raw_markets)

    return save_market_snapshot(snapshot)


def run_labeling() -> Path:
    """Create labels."""
    print("\n[3/7] Creating labels")
    snapshot = load_unlabeled_snapshot()
    labeled_snapshot = add_activity_spike_label(snapshot)

    return save_labeled_data(labeled_snapshot)


def run_training() -> Path:
    """Train model candidates, log MLflow runs, and save the best model."""
    print("\n[4/7] Training and comparing models")
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
    }

    best_model = None
    best_model_name = None
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
            best_model = model
            best_model_name = model_name
            best_metrics = metrics

    saved_path = save_model(best_model)

    print(f"\nBest model: {best_model_name}")
    print(f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}")

    return saved_path


def run_scoring() -> Path:
    """Score all processed markets."""
    print("\n[5/7] Scoring markets")
    market_snapshot = load_market_snapshot()
    model = load_scoring_model()
    predictions = score_markets(market_snapshot, model)

    return save_predictions(predictions)


def run_feature_importance() -> Path:
    """Export feature importance for the saved model."""
    print("\n[6/7] Exporting feature importance")
    model = load_feature_importance_model()
    feature_importance = build_feature_importance(model)

    return save_feature_importance(feature_importance)


def run_monitoring_summary() -> None:
    """Summarize prediction logs if they exist."""
    print("\n[7/7] Summarizing prediction logs")

    try:
        prediction_log = load_prediction_log()
    except FileNotFoundError:
        print("No prediction log found yet. Skipping monitoring summary.")
        return

    summary = summarize_predictions(prediction_log)

    for metric_name, metric_value in summary.items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value}")


if __name__ == "__main__":
    raw_path = run_ingestion()
    snapshot_path = run_feature_building()
    labeled_path = run_labeling()
    model_path = run_training()
    predictions_path = run_scoring()
    feature_importance_path = run_feature_importance()
    run_monitoring_summary()

    print("\nOffline pipeline complete")
    print(f"Raw data: {raw_path}")
    print(f"Market snapshot: {snapshot_path}")
    print(f"Labeled data: {labeled_path}")
    print(f"Model: {model_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Feature importance: {feature_importance_path}")
