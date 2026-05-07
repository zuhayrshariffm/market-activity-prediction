from zoneinfo import ZoneInfo
from polymarket_client import PolymarketClient
from datetime import datetime, timezone
from fetch_markets import fetch_and_save_markets

poly = PolymarketClient()

dict = poly.get_markets(5)

utc = datetime.now(timezone.utc)
local = utc.astimezone(ZoneInfo("America/New_York"))

new = fetch_and_save_markets(limit=5)
print(dict[0])