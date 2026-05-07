"""
Fetch raw Polymarket market data and save it locally.

This script is the first ingestion step in the pipeline.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.polymarket_client import PolymarketClient


def fetch_and_save_markets(limit: int = 100) -> Path:
    """
    Fetch markets from Polymarket and save the raw response as JSON.

    Parameters
    ----------
    limit:
        Maximum number of markets to fetch.

    Returns
    -------
    Path
        Path to the saved raw data file.
    """
    client = PolymarketClient()
    markets = client.get_markets(limit=limit)

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"polymarket_markets_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(markets, file, indent=2)

    return output_path


if __name__ == "__main__":
    saved_path = fetch_and_save_markets(limit=100)
    print(f"Saved raw market data to {saved_path}")
