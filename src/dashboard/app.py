"""
Streamlit dashboard for market activity spike predictions.
"""

from pathlib import Path

import pandas as pd
import streamlit as st


PREDICTIONS_PATH = Path("data/processed/market_predictions.csv")
MONITORING_LOG_PATH = Path("data/monitoring/prediction_log.csv")


st.set_page_config(
    page_title="Market Activity Spike Predictions",
    layout="wide",
)


@st.cache_data
def load_predictions(path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    """Load scored market predictions."""
    if not path.exists():
        raise FileNotFoundError(
            f"No predictions found at {path}. Run python -m src.inference.score_markets first."
        )

    return pd.read_csv(path)


@st.cache_data
def load_monitoring_log(path: Path = MONITORING_LOG_PATH) -> pd.DataFrame | None:
    """Load prediction monitoring log if available."""
    if not path.exists():
        return None

    return pd.read_csv(path)


predictions = load_predictions()
monitoring_log = load_monitoring_log()

st.title("Market Activity Spike Predictions")

total_markets = len(predictions)
predicted_spikes = int(predictions["prediction"].sum())
average_probability = predictions["spike_probability"].mean()

metric_col_1, metric_col_2, metric_col_3 = st.columns(3)

metric_col_1.metric("Markets scored", total_markets)
metric_col_2.metric("Predicted spikes", predicted_spikes)
metric_col_3.metric("Average spike probability", f"{average_probability:.2%}")

st.subheader("Top Markets by Spike Probability")

top_markets = predictions[
    [
        "question",
        "spike_probability",
        "prediction",
        "volume_24h",
        "volume_1wk",
        "liquidity",
        "spread",
    ]
].head(20)

st.dataframe(
    top_markets,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Prediction Distribution")

st.bar_chart(predictions["spike_probability"])

st.subheader("Scored Market Data")

st.dataframe(
    predictions,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Prediction Monitoring")

if monitoring_log is None:
    st.info("No prediction log found yet. Use the API to generate prediction logs.")
else:
    monitoring_col_1, monitoring_col_2, monitoring_col_3 = st.columns(3)

    monitoring_col_1.metric("Logged predictions", len(monitoring_log))
    monitoring_col_2.metric(
        "Logged spike rate",
        f"{monitoring_log['prediction'].mean():.2%}",
    )
    monitoring_col_3.metric(
        "Avg logged probability",
        f"{monitoring_log['spike_probability'].mean():.2%}",
    )

    st.dataframe(
        monitoring_log.tail(25),
        use_container_width=True,
        hide_index=True,
    )
