# replicate_original.py
# Step 2: how much of my original R^2 of 0.78 came from the target overlapping the feature?
#
# Part A rebuilds my TML notebook exactly: same features, same target, same rolling split,
#   same linear regression. If it's right, it should give back R^2 = 0.78, 0.54, 0.33.
# Part B keeps EVERYTHING the same except the target. Instead of the 20-minute rolling
#   volatility shifted forward (which shares minutes with the feature), it uses volatility
#   over the next h minutes only. The difference between A and B is the overlap.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("data/btcusdt_1m.csv", index_col="open_time", parse_dates=True)

# ---------- my original features, exactly as in the notebook ----------
feat = pd.DataFrame(index=df.index)
buy = df["taker_buy_base"]
sell = df["volume"] - buy
feat["OFI"] = (buy - sell) / (df["volume"] + 1e-10)
feat["trade_intensity"] = df["n_trades"]
log_return = np.log(df["close"] / df["close"].shift(1))
feat["realized_vol"] = log_return.rolling(window=20, min_periods=10).std()   # minutes t-19 .. t
feat["volume_proxy"] = df["quote_volume"]
feature_cols = ["OFI", "trade_intensity", "realized_vol", "volume_proxy"]

r2 = log_return ** 2
for h in [5, 10, 15]:
    # A: my ORIGINAL target. The same 20-minute window, h minutes later: minutes t-19+h .. t+h
    # NOTE: this shares 20 - h of its 20 minutes with the realized_vol feature
    feat[f"original_{h}"] = feat["realized_vol"].shift(-h)
    # B: the CLEAN target. Volatility over minutes t+1 .. t+h only, no shared minutes
    feat[f"clean_{h}"] = np.sqrt(r2.rolling(h).mean().shift(-h))

feat = feat.dropna()


# ---------- the same rolling-window test as my notebook ----------
# train on the first 70%, test on the next 10%, move forward, three rounds
def rolling_r2(target_col):
    total = len(feat)
    train_size = int(total * 0.70)
    test_size = int(total * 0.10)
    scores = []
    end = train_size
    while end + test_size <= total:
        X_train = feat[feature_cols].iloc[:end]
        y_train = feat[target_col].iloc[:end]
        X_test = feat[feature_cols].iloc[end:end + test_size]
        y_test = feat[target_col].iloc[end:end + test_size]
        model = LinearRegression().fit(X_train, y_train)
        scores.append(r2_score(y_test, model.predict(X_test)))
        end += test_size
    return np.mean(scores)


print("Out-of-sample R^2, linear regression, my original four features\n")
print("horizon   minutes shared   original target   clean target")
rows = []
for h in [5, 10, 15]:
    a = rolling_r2(f"original_{h}")
    b = rolling_r2(f"clean_{h}")
    shared = (20 - h) / 20
    rows.append({"horizon": h, "shared": shared, "original": a, "clean": b})
    print(f"{h:>4} min        {shared:>5.0%}          {a:>8.4f}         {b:>8.4f}")

pd.DataFrame(rows).to_csv("data/replication.csv", index=False)
