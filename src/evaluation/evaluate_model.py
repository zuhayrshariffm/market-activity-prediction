"""
Evaluate the saved activity spike prediction model.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from src.models.train_model import FEATURE_COLUMNS, TARGET_COLUMN


def load_labeled_data(path: Path = Path("data/processed/market_snapshot_labeled.csv")) -> pd.DataFrame:
    """Load labeled market data."""
    return pd.read_csv(path)


def load_model(path: Path = Path("models/activity_spike_model.joblib")):
    """Load the trained model artifact."""
    return joblib.load(path)


def evaluate_saved_model(df: pd.DataFrame, model) -> None:
    """Evaluate the saved model on a holdout split."""
    X = df[FEATURE_COLUMNS].fillna(0)
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nROC-AUC:")
    print(roc_auc_score(y_test, y_prob))


if __name__ == "__main__":
    labeled_data = load_labeled_data()
    trained_model = load_model()
    evaluate_saved_model(labeled_data, trained_model)
