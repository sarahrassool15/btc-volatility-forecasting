# har_models.py
# Steps 3 and 4: a proper volatility benchmark, then the real research question.
#
# Every model predicts the CLEAN target (volatility over the next h minutes only) and is
# tested exactly like my original study: train on the first 70%, test on the next 10%,
# move forward, three rounds, average the out-of-sample R^2.
#
# Models, each adding something to the one before:
#   1. HAR: volatility over the last 5 minutes, last hour, last day  (Corsi, 2009)
#   2. HAR + hour of day, for the daily cycle I found in my first project
#   3. HAR + hour + order flow: OFI, |OFI|, trade count, volume
# (My original four features scored 0.14-0.16 on the clean target in step 2, for comparison.)
# The question: does order flow add anything once volatility's own history is in the model?

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

feat = pd.read_csv("data/features.csv", index_col=0, parse_dates=True)

# |OFI|: how one-sided trading was, whichever side.
# NOTE: volatility has no direction, so a stampede of buyers or of sellers should both
# raise it. Signed OFI can't capture that in a linear model; |OFI| can.
feat["abs_ofi"] = feat["ofi"].abs()

# hour of day as 23 yes/no columns (one hour is left out as the baseline)
hours = pd.get_dummies(feat["hour"], prefix="h", drop_first=True).astype(float)
feat = pd.concat([feat, hours], axis=1)
hour_cols = list(hours.columns)

har = ["rv_5", "rv_60", "rv_1440"]
micro = ["ofi", "abs_ofi", "trades", "log_volume"]
models = {
    "HAR": har,
    "HAR + hour": har + hour_cols,
    "HAR + hour + order flow": har + hour_cols + micro,
}


def rolling_r2(cols, target):
    total = len(feat)
    train_size, test_size = int(total * 0.70), int(total * 0.10)
    scores, end = [], train_size
    while end + test_size <= total:
        X_tr, y_tr = feat[cols].iloc[:end], feat[target].iloc[:end]
        X_te, y_te = feat[cols].iloc[end:end + test_size], feat[target].iloc[end:end + test_size]
        # NOTE: always standardise first (I learned this the hard way in step 2)
        model = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
        scores.append(r2_score(y_te, model.predict(X_te)))
        end += test_size
    return np.mean(scores)


rows = []
for name, cols in models.items():
    row = {"model": name}
    for h in [5, 10, 15]:
        row[f"{h} min"] = rolling_r2(cols, f"rv_next_{h}")
    rows.append(row)

table = pd.DataFrame(rows).set_index("model")
table.to_csv("data/har_results.csv")
print("Out-of-sample R^2 (clean target, average of three rolling rounds)\n")
print(table.round(4).to_string())

gain = table.loc["HAR + hour + order flow"] - table.loc["HAR + hour"]
print("\nWhat order flow adds on top of HAR + hour:")
print(gain.round(4).to_string())
