"""
Export feature importance values from the saved activity spike model.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.config import load_config
from src.models.train_model import FEATURE_COLUMNS


config = load_config()

MODEL_PATH = Path(config["files"]["model_artifact"])
OUTPUT_PATH = Path("data/processed/feature_importance.csv")


def load_model(path: Path = MODEL_PATH):
    """Load saved model artifact."""
    return joblib.load(path)


def get_classifier(model):
    """Return the classifier step from the model pipeline."""
    return model.named_steps["classifier"]


def build_feature_importance(model) -> pd.DataFrame:
    """Build a feature importance dataframe from the saved model."""
    classifier = get_classifier(model)

    if hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(classifier, "coef_"):
        importance_values = abs(classifier.coef_[0])
        importance_type = "absolute_coefficient"
    else:
        raise ValueError("Model does not expose feature importances or coefficients.")

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": importance_values,
            "importance_type": importance_type,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    )

    return importance_df


def save_feature_importance(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> Path:
    """Save feature importance output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    return path


if __name__ == "__main__":
    model = load_model()
    feature_importance = build_feature_importance(model)
    saved_path = save_feature_importance(feature_importance)

    print(f"Saved feature importance to {saved_path}")
    print("\nTop features:")
    print(feature_importance.head(10).to_string(index=False))
