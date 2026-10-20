# significance_regimes.py
# Steps 5 and 6: is the order-flow effect statistically real, and does it matter more
# when the market is turbulent?
#
# Step 5, Newey-West: I fit HAR + hour + order flow on the whole sample and test each
#   order-flow coefficient. Ordinary OLS standard errors assume every minute is independent,
#   but neighbouring targets share h-1 of their h minutes and volatility is persistent,
#   so ordinary standard errors are far too small. Newey-West corrects for that.
#
# Step 6, regimes: I split the TEST minutes into calm and turbulent ones, using last hour's
#   volatility (known at time t) above the 80th percentile of the training sample as
#   "turbulent". Then I compare HAR + hour with and without order flow in each regime.

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

feat = pd.read_csv("data/features.csv", index_col=0, parse_dates=True)
feat["abs_ofi"] = feat["ofi"].abs()
hours = pd.get_dummies(feat["hour"], prefix="h", drop_first=True).astype(float)
feat = pd.concat([feat, hours], axis=1)
hour_cols = list(hours.columns)

har = ["rv_5", "rv_60", "rv_1440"]
micro = ["ofi", "abs_ofi", "trades", "log_volume"]
base = har + hour_cols
full = base + micro

# ---------- step 5: ordinary vs Newey-West standard errors ----------
print("Step 5: order-flow coefficients in HAR + hour + order flow (whole sample)")
print("        t-statistics with ordinary OLS vs Newey-West standard errors\n")
for h in [5, 15]:
    y = feat[f"rv_next_{h}"]
    # standardise so every coefficient is "effect of a one-standard-deviation change"
    X = (feat[full] - feat[full].mean()) / feat[full].std()
    X = sm.add_constant(X)
    ols = sm.OLS(y, X).fit()
    # NOTE: maxlags must cover the overlap between neighbouring targets (h - 1 minutes)
    # and volatility's own persistence, so I use a generous 60 minutes
    nw = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 60})
    print(f"  {h}-minute horizon")
    print("    feature        coefficient    t (OLS)   t (Newey-West)")
    for c in micro + ["rv_5", "rv_60", "rv_1440"]:
        print(f"    {c:<12} {ols.params[c]:>13.2e} {ols.tvalues[c]:>10.1f} {nw.tvalues[c]:>13.1f}")
    print()

# ---------- step 6: calm vs turbulent minutes ----------
print("Step 6: out-of-sample R^2 in calm and turbulent minutes (5-minute horizon)\n")
target = "rv_next_5"
total = len(feat)
train_size, test_size = int(total * 0.70), int(total * 0.10)
preds = {"base": [], "full": []}
actual, turbulent = [], []
end = train_size
while end + test_size <= total:
    tr, te = feat.iloc[:end], feat.iloc[end:end + test_size]
    cutoff = tr["rv_60"].quantile(0.80)              # decided from training data only
    for name, cols in [("base", base), ("full", full)]:
        m = make_pipeline(StandardScaler(), LinearRegression()).fit(tr[cols], tr[target])
        preds[name].append(m.predict(te[cols]))
    actual.append(te[target].values)
    turbulent.append((te["rv_60"] > cutoff).values)
    end += test_size

actual = np.concatenate(actual)
turbulent = np.concatenate(turbulent)
base_pred = np.concatenate(preds["base"])
full_pred = np.concatenate(preds["full"])

rows = []
for label, mask in [("calm", ~turbulent), ("turbulent", turbulent), ("all", np.ones_like(turbulent))]:
    r_base = r2_score(actual[mask], base_pred[mask])
    r_full = r2_score(actual[mask], full_pred[mask])
    rows.append({"regime": label, "share of minutes": mask.mean(),
                 "HAR + hour": r_base, "+ order flow": r_full, "gain": r_full - r_base})
table = pd.DataFrame(rows).set_index("regime")
table.to_csv("data/regimes.csv")
print(table.round(4).to_string())
