"""
Client for fetching market data from the Polymarket API.

This module is responsible for collecting raw market-level data that will
later be transformed into features for activity spike prediction.
"""

from typing import Any

import requests


class PolymarketClient:
    """Simple client for Polymarket market data."""

    def __init__(self, base_url: str = "https://gamma-api.polymarket.com"):
        self.base_url = base_url.rstrip("/")

    def get_markets(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Fetch recent Polymarket markets.

        Parameters
        ----------
        limit:
            Maximum number of markets to fetch.

        Returns
        -------
        list[dict[str, Any]]
            Raw market records returned by the Polymarket API.
        """
        url = f"{self.base_url}/markets"
        params = {"limit": limit}

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        return response.json()
