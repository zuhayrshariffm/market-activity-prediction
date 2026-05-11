"""
Summarize logged model predictions for basic monitoring.
"""

from pathlib import Path

import pandas as pd


PREDICTION_LOG_PATH = Path("data/monitoring/prediction_log.csv")


def load_prediction_log(path: Path = PREDICTION_LOG_PATH) -> pd.DataFrame:
    """Load logged model predictions."""
    if not path.exists():
        raise FileNotFoundError(
            f"No prediction log found at {path}. Run the API and make predictions first."
        )

    return pd.read_csv(path)


def summarize_predictions(df: pd.DataFrame) -> dict[str, float | int]:
    """Create summary metrics from logged predictions."""
    return {
        "prediction_count": len(df),
        "average_spike_probability": df["spike_probability"].mean(),
        "min_spike_probability": df["spike_probability"].min(),
        "max_spike_probability": df["spike_probability"].max(),
        "predicted_spike_rate": df["prediction"].mean(),
        "average_volume_24h": df["volume_24h"].mean(),
        "average_liquidity": df["liquidity"].mean(),
        "average_spread": df["spread"].mean(),
    }


if __name__ == "__main__":
    prediction_log = load_prediction_log()
    summary = summarize_predictions(prediction_log)

    print("Prediction monitoring summary:")
    for metric_name, metric_value in summary.items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value}")
