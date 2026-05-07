"""
Create activity spike labels from processed market snapshot data.

This first version uses a proxy label based on recent 24-hour volume
relative to the market's trailing 1-week average daily volume.
"""

from pathlib import Path

import pandas as pd

def load_market_snapshot(path: Path = Path("data/processed/market_snapshot.csv")) -> pd.DataFrame:
    """Load the processed market snapshot dataset."""
    return pd.read_csv(path)


def add_activity_spike_label(df: pd.DataFrame, spike_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Add a binary activity spike label.

    A market is labeled as a spike if its 24-hour volume is greater than
    spike_multiplier times its average daily volume over the past week.
    """
    labeled_df = df.copy()

    labeled_df["avg_daily_volume_1wk"] = labeled_df["volume_1wk"] / 7
    labeled_df["activity_spike"] = (
        labeled_df["volume_24h"] > spike_multiplier * labeled_df["avg_daily_volume_1wk"]
    ).astype(int)

    return labeled_df


def save_labeled_data(df: pd.DataFrame) -> Path:
    """Save labeled market data."""
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "market_snapshot_labeled.csv"
    df.to_csv(output_path, index=False)

    return output_path


if __name__ == "__main__":
    snapshot = load_market_snapshot()
    labeled_snapshot = add_activity_spike_label(snapshot)
    saved_path = save_labeled_data(labeled_snapshot)

    spike_count = labeled_snapshot["activity_spike"].sum()
    total_count = len(labeled_snapshot)

    print(f"Saved labeled data to {saved_path}")
    print(f"Rows: {total_count}")
    print(f"Activity spikes: {spike_count}")
    print(f"Spike rate: {spike_count / total_count:.2%}")
