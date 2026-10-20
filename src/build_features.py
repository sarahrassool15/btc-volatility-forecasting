# build_features.py
# Step 1 of my volatility forecasting project (v2 of my TML paper).
# Turns 1-minute Binance US candles into the features and targets for the model.
#
# Rule I follow everywhere: features only use data UP TO minute t,
# targets only use returns AFTER minute t. No overlap, so no leakage.
#
# Target:  realized volatility over the next h minutes, h = 5, 10, 15
#     RV(t, h) = sqrt( (r_{t+1}^2 + ... + r_{t+h}^2) / h )     (per-minute scale)
# HAR features: volatility over the last 5 minutes, last hour, last day
# Microstructure features (as in my TML paper): order flow imbalance, trade count, volume

import numpy as np
import pandas as pd

df = pd.read_csv("data/btcusdt_1m.csv", index_col="open_time", parse_dates=True)

# NOTE: Binance US can have missing minutes (like the outage I found in my first project).
# I put the data on a complete minute grid, so missing minutes become NaN
# instead of silently joining two minutes that aren't next to each other.
full = pd.date_range(df.index.min(), df.index.max(), freq="1min")
df = df.reindex(full)

r = np.log(df["close"]).diff()        # 1-minute log returns (NaN next to any gap)
r2 = r ** 2

out = pd.DataFrame(index=df.index)

# ---------- targets: FUTURE realized volatility ----------
# rolling(h).mean() at minute t+h covers minutes t+1..t+h, so I shift it back by h
# NOTE: I use the mean (not the sum) so every horizon is on the same per-minute scale
for h in [5, 10, 15]:
    out[f"rv_next_{h}"] = np.sqrt(r2.rolling(h).mean().shift(-h))

# ---------- HAR features: PAST realized volatility over three windows ----------
out["rv_5"] = np.sqrt(r2.rolling(5).mean())        # last 5 minutes
out["rv_60"] = np.sqrt(r2.rolling(60).mean())      # last hour
out["rv_1440"] = np.sqrt(r2.rolling(1440).mean())  # last day

# ---------- microstructure features (same ideas as my TML paper) ----------
# order flow imbalance: (buyer-initiated - seller-initiated) / total volume, in [-1, 1]
# taker_buy_base is the volume where the BUYER was the aggressive side
# NOTE: Binance US is thin, so many minutes have no trades. Then there's no imbalance,
# so I set OFI to 0 instead of dividing by zero.
buy = df["taker_buy_base"]
sell = df["volume"] - buy
out["ofi"] = np.where(df["volume"] > 0, (buy - sell) / df["volume"], 0.0)
out["trades"] = df["n_trades"]
# dollar volume is very skewed, so I log it; log1p = log(1 + x) handles minutes with zero volume
out["log_volume"] = np.log1p(df["quote_volume"])

# hour of day (UTC), for the daily cycle I found in my first project
out["hour"] = out.index.hour

# drop minutes where any value is missing (the first day, the last 15 minutes, and around gaps)
before = len(out)
out = out.dropna()

out.to_csv("data/features.csv")
print(f"kept {len(out):,} of {before:,} minutes")
print(f"from {out.index.min()} to {out.index.max()}")
print(f"minutes with no trades: {(out['trades'] == 0).mean():.1%}")
print(f"minutes where the price didn't move over the next 5 minutes: {(out['rv_next_5'] == 0).mean():.1%}")
print()
print(out.describe().T[["mean", "std", "min", "max"]].round(6))
