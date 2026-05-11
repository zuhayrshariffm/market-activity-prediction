"""
Build a cleaned market snapshot dataset from raw Polymarket market data.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def get_latest_raw_file(raw_dir: Path = Path("data/raw")) -> Path:
    """Return the most recent raw Polymarket JSON file."""
    files = sorted(raw_dir.glob("polymarket_markets_*.json"))

    if not files:
        raise FileNotFoundError("No raw Polymarket market files found in data/raw.")

    return files[-1]


def load_raw_markets(path: Path) -> list[dict[str, Any]]:
    """Load raw market records from a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_market_snapshot(markets: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw market records into a cleaned market-level table."""
    rows = []

    for market in markets:
        rows.append(
            {
                "market_id": market.get("id"),
                "question": market.get("question"),
                "slug": market.get("slug"),
                "event_title": (
                    market.get("events", [{}])[0].get("title")
                    if market.get("events")
                    else None
                ),
                "active": market.get("active"),
                "closed": market.get("closed"),
                "accepting_orders": market.get("acceptingOrders"),
                "enable_order_book": market.get("enableOrderBook"),
                "created_at": market.get("createdAt"),
                "updated_at": market.get("updatedAt"),
                "start_date": market.get("startDate"),
                "end_date": market.get("endDate"),
                "volume": market.get("volumeNum"),
                "liquidity": market.get("liquidityNum"),
                "volume_24h": market.get("volume24hr"),
                "volume_1wk": market.get("volume1wk"),
                "volume_1mo": market.get("volume1mo"),
                "spread": market.get("spread"),
                "one_day_price_change": market.get("oneDayPriceChange"),
                "one_week_price_change": market.get("oneWeekPriceChange"),
                "one_month_price_change": market.get("oneMonthPriceChange"),
                "last_trade_price": market.get("lastTradePrice"),
                "best_bid": market.get("bestBid"),
                "best_ask": market.get("bestAsk"),
                "competitive": market.get("competitive"),
            }
        )

    df = pd.DataFrame(rows)

    numeric_columns = [
        "volume",
        "liquidity",
        "volume_24h",
        "volume_1wk",
        "volume_1mo",
        "spread",
        "one_day_price_change",
        "one_week_price_change",
        "one_month_price_change",
        "last_trade_price",
        "best_bid",
        "best_ask",
        "competitive",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    date_columns = ["created_at", "updated_at", "start_date", "end_date"]

    for column in date_columns:
        df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)

    df["market_age_days"] = (
        df["updated_at"] - df["created_at"]
    ).dt.total_seconds() / 86400
    df["time_to_resolution_days"] = (
        df["end_date"] - df["updated_at"]
    ).dt.total_seconds() / 86400

    return df


def save_market_snapshot(df: pd.DataFrame) -> Path:
    """Save cleaned market snapshot data."""
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "market_snapshot.csv"
    df.to_csv(output_path, index=False)

    return output_path


if __name__ == "__main__":
    raw_path = get_latest_raw_file()
    raw_markets = load_raw_markets(raw_path)
    snapshot = build_market_snapshot(raw_markets)
    saved_path = save_market_snapshot(snapshot)

    print(f"Loaded raw data from {raw_path}")
    print(f"Saved market snapshot to {saved_path}")
    print(f"Rows: {len(snapshot)}")
    print(f"Columns: {list(snapshot.columns)}")
