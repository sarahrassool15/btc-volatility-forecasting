# diagnose.py
# My replication gave negative R^2 instead of the 0.78 in my notebook.
# Before trusting either number, I check three things:
#   1. Is this the same data? My paper's Table 4 has an in-sample OLS fit on the first
#      90,702 rows: R^2 = 0.859, realized_vol coefficient = 0.9093. I recompute it.
#   2. Is one feature wrecking the forecasts? I rerun each round with realized_vol alone.
#   3. Are there extreme values in the test periods that weren't in training?

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("data/btcusdt_1m.csv", index_col="open_time", parse_dates=True)

# same features and original target as my notebook
feat = pd.DataFrame(index=df.index)
buy = df["taker_buy_base"]
feat["OFI"] = (buy - (df["volume"] - buy)) / (df["volume"] + 1e-10)
feat["trade_intensity"] = df["n_trades"]
log_return = np.log(df["close"] / df["close"].shift(1))
feat["realized_vol"] = log_return.rolling(window=20, min_periods=10).std()
feat["volume_proxy"] = df["quote_volume"]
feat = feat.dropna(subset=["realized_vol"])
feat["target_vol_5"] = feat["realized_vol"].shift(-5)
feat = feat.dropna()
cols = ["OFI", "trade_intensity", "realized_vol", "volume_proxy"]

total = len(feat)
train_size = int(total * 0.70)
test_size = int(total * 0.10)
print(f"rows: {total:,}   (notebook had 129,575)   training rows: {train_size:,}   (notebook had 90,702)\n")

# ---------- check 1: same data? ----------
X = sm.add_constant(feat[cols].iloc[:train_size])
fit = sm.OLS(feat["target_vol_5"].iloc[:train_size], X).fit()
print("Check 1: in-sample OLS on the training rows (paper's Table 4: R^2 0.859, realized_vol 0.9093)")
print(f"  R^2 {fit.rsquared:.4f}")
print(fit.params.to_string(), "\n")

# ---------- check 2: all features vs realized_vol alone, round by round ----------
print("Check 2: out-of-sample R^2 per round, 5-minute original target")
print("  round   all four features   realized_vol only")
end = train_size
rnd = 1
while end + test_size <= total:
    tr, te = slice(0, end), slice(end, end + test_size)
    y_tr, y_te = feat["target_vol_5"].iloc[tr], feat["target_vol_5"].iloc[te]
    all4 = LinearRegression().fit(feat[cols].iloc[tr], y_tr).predict(feat[cols].iloc[te])
    one = LinearRegression().fit(feat[["realized_vol"]].iloc[tr], y_tr).predict(feat[["realized_vol"]].iloc[te])
    print(f"  {rnd:>5}   {r2_score(y_te, all4):>17.4f}   {r2_score(y_te, one):>17.4f}")
    end += test_size
    rnd += 1

# ---------- check 3: extreme values ----------
print("\nCheck 3: largest value of each feature, training rows vs everything after")
for c in cols:
    print(f"  {c:<16} train max {feat[c].iloc[:train_size].max():>14.6g}"
          f"   test max {feat[c].iloc[train_size:].max():>14.6g}")
