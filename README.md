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
