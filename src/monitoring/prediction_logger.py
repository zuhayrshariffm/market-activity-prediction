"""
Utilities for logging model predictions for monitoring.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PREDICTION_LOG_PATH = Path("data/monitoring/prediction_log.csv")


def log_prediction(
    features: dict[str, Any],
    spike_probability: float,
    prediction: int,
    log_path: Path = PREDICTION_LOG_PATH,
) -> Path:
    """Append a model prediction record to the prediction log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        "spike_probability": spike_probability,
        "prediction": prediction,
        **features,
    }

    record_df = pd.DataFrame([record])

    if log_path.exists():
        record_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        record_df.to_csv(log_path, index=False)

    return log_path
