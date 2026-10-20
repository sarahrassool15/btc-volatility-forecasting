# fetch_data.py
# Downloads the SAME 90 days of 1-minute BTC/USDT candles my TML paper used,
# from the Binance US public API: 2 January 2026 18:12 to 2 April 2026 18:11 (UTC).
# NOTE: my original notebook fetched "the last 90 days from now", so the sample changed
# every time it ran. Fixing the dates makes this version reproducible.

import time
from datetime import datetime, timezone

import pandas as pd
import requests

URL = "https://api.binance.us/api/v3/klines"
COLUMNS = ["open_time", "open", "high", "low", "close", "volume",
           "close_time", "quote_volume", "n_trades",
           "taker_buy_base", "taker_buy_quote", "ignore"]

start = datetime(2026, 1, 2, 18, 12, tzinfo=timezone.utc)
end = datetime(2026, 4, 2, 18, 12, tzinfo=timezone.utc)     # stops just before this minute
start_ms = int(start.timestamp() * 1000)
end_ms = int(end.timestamp() * 1000) - 1

rows = []
while start_ms <= end_ms:
    params = {"symbol": "BTCUSDT", "interval": "1m",
              "startTime": start_ms, "endTime": end_ms, "limit": 1000}
    response = requests.get(URL, params=params, timeout=10)
    response.raise_for_status()
    batch = response.json()
    if not batch:
        break
    rows.extend(batch)
    start_ms = batch[-1][0] + 60_000      # next request starts one minute after the last candle
    time.sleep(0.2)

df = pd.DataFrame(rows, columns=COLUMNS)
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
for col in ["open", "high", "low", "close", "volume", "quote_volume",
            "taker_buy_base", "taker_buy_quote"]:
    df[col] = df[col].astype(float)
df = df.drop(columns=["close_time", "ignore"]).drop_duplicates("open_time").set_index("open_time")

df.to_csv("data/btcusdt_1m.csv")
print(f"{len(df):,} candles from {df.index.min()} to {df.index.max()}")
print(f"minutes with no trades at all: {(df['n_trades'] == 0).mean():.1%}")
