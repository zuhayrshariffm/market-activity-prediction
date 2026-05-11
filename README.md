# Market Activity Spike Prediction

This project builds an end-to-end machine learning pipeline to predict short-term activity spikes in prediction markets using Polymarket API data.

The goal is to identify markets that are likely to experience a near-term surge in activity, such as increased volume, higher liquidity, greater volatility, more trades etc.

These markets would then be prime candidates for extra advertising, boosting (think landing page of the website etc.).

## Project Objective

Given recent market-level data, the model predicts whether a market is likely to experience an activity spike over a short future window.

Initial target:

> Predict whether a market will experience a volume spike in the next 24 hours.

A volume spike can be defined as future 24-hour volume exceeding a market’s recent rolling baseline, such as 2x its trailing 7-day average volume.

## Data Source

The project uses prediction market data from Polymarket.

Potential data fields include:

- Market ID
- Market question/title
- Category or topic

## Feature Engineering

The pipeline will create features such as:

- Recent volume over 1-hour, 6-hour, and 24-hour windows
- Rolling average volume

## Label Definition

The first version of the model will use a binary label:

```text
activity_spike = 1 if next_24h_volume > 2 * trailing_7d_average_volume
activity_spike = 0 otherwise
```

## Architecture

```text
Polymarket API
   ↓
Raw JSON ingestion
   ↓
Market snapshot feature table
   ↓
Proxy activity spike labels
   ↓
Model training + MLflow experiment tracking
   ↓
Saved model artifact
   ↓
Batch scoring + FastAPI prediction serving
   ↓
Prediction logging + monitoring summary
   ↓
Streamlit dashboard
```

## Project Structure

```text
configs/        Shared project configuration
scripts/        Offline workflow runner
src/ingestion/  Polymarket API client and raw data fetch
src/features/   Market snapshot feature engineering
src/labeling/   Activity spike label generation
src/models/     Model training and MLflow logging
src/evaluation/ Model evaluation and feature importance
src/inference/  Batch market scoring
src/api/        FastAPI prediction service
src/monitoring/ Prediction logging and monitoring summaries
src/dashboard/  Streamlit dashboard
```

## How to Run

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full offline workflow:

```bash
python -m scripts.run_offline_pipeline
```

This fetches recent Polymarket market data, builds features, creates labels, trains and compares models, scores markets, exports feature importance, and summarizes prediction logs if available.

Run the prediction API:

```bash
uvicorn src.api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API docs are available at:

```text
http://127.0.0.1:8000/docs
```

Run the Streamlit dashboard:

```bash
streamlit run src/dashboard/app.py
```

The dashboard will usually be available at:

```text
http://localhost:8501
```

Run MLflow UI:

```bash
mlflow ui
```

The MLflow experiment dashboard will usually be available at:

```text
http://127.0.0.1:5000
```

## Current Modeling Note

The current target uses a proxy activity spike label based on recent 24-hour volume relative to trailing 1-week average daily volume. This allows the end-to-end ML workflow to be validated with currently available snapshot data.

A future version will use time-series market snapshots to create forward-looking labels, such as whether a market experiences a volume, liquidity, volatility, or trade-count spike in the next 24 hours.

## What This Project Demonstrates

- ML feature engineering on prediction market data
- Binary classification target design
- Baseline and ensemble model comparison
- MLflow experiment tracking
- Model artifact saving and reuse
- FastAPI model serving
- Batch inference
- Prediction logging and monitoring summaries
- Streamlit dashboarding
- Dockerized API environment

## Limitations and Future Work

- Current labels use a proxy spike definition based on recent volume, not a fully forward-looking historical outcome.
- Future data collection should store repeated market snapshots over time.
- Future labels should measure whether a market spikes in the next 24 hours after each snapshot.
- Add time-based train/validation/test splits once historical snapshots are available.
- Add feature drift monitoring and realized precision/recall tracking after outcomes are observed.
- Compare additional models such as XGBoost or LightGBM.
- Move storage from local CSV files to PostgreSQL or DuckDB for more scalable analysis.
