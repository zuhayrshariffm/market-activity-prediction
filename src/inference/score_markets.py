"""
Score processed markets with the trained activity spike model.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.models.train_model import FEATURE_COLUMNS


MODEL_PATH = Path("models/activity_spike_model.joblib")
INPUT_PATH = Path("data/processed/market_snapshot.csv")
OUTPUT_PATH = Path("data/processed/market_predictions.csv")


def load_market_snapshot(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load processed market snapshot data."""
    return pd.read_csv(path)


def load_model(path: Path = MODEL_PATH):
    """Load trained model artifact."""
    return joblib.load(path)


def score_markets(df: pd.DataFrame, model) -> pd.DataFrame:
    """Add spike probabilities and predictions to market snapshot data."""
    scored_df = df.copy()

    X = scored_df[FEATURE_COLUMNS].fillna(0)

    scored_df["spike_probability"] = model.predict_proba(X)[:, 1]
    scored_df["prediction"] = (scored_df["spike_probability"] >= 0.5).astype(int)

    scored_df = scored_df.sort_values(
        by="spike_probability",
        ascending=False,
    )

    return scored_df


def save_predictions(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> Path:
    """Save scored market predictions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    return path


if __name__ == "__main__":
    market_snapshot = load_market_snapshot()
    model = load_model()

    predictions = score_markets(market_snapshot, model)
    saved_path = save_predictions(predictions)

    print(f"Saved market predictions to {saved_path}")
    print("\nTop 10 markets by predicted spike probability:")
    print(
        predictions[
            [
                "market_id",
                "question",
                "spike_probability",
                "prediction",
                "volume_24h",
                "volume_1wk",
                "spread",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
