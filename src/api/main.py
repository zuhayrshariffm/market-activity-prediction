"""
FastAPI app for serving activity spike predictions.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.models.train_model import FEATURE_COLUMNS


MODEL_PATH = Path("models/activity_spike_model.joblib")

app = FastAPI(title="Market Activity Spike Prediction API")


class MarketFeatures(BaseModel):
    volume: float
    liquidity: float
    volume_24h: float
    volume_1wk: float
    volume_1mo: float
    spread: float
    one_day_price_change: float
    one_week_price_change: float
    one_month_price_change: float
    last_trade_price: float
    best_bid: float
    best_ask: float
    competitive: float
    market_age_days: float
    time_to_resolution_days: float


def load_model():
    """Load the trained model artifact."""
    return joblib.load(MODEL_PATH)


model = load_model()


@app.get("/")
def root() -> dict[str, str]:
    """Health check endpoint."""
    return {"message": "Market Activity Spike Prediction API"}


@app.post("/predict")
def predict(features: MarketFeatures) -> dict[str, float | int]:
    """Predict activity spike probability for a single market."""
    feature_df = pd.DataFrame([features.model_dump()])
    feature_df = feature_df[FEATURE_COLUMNS]

    spike_probability = model.predict_proba(feature_df)[0, 1]
    prediction = int(spike_probability >= 0.5)

    return {
        "spike_probability": float(spike_probability),
        "prediction": prediction,
    }
