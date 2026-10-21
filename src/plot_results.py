# plot_results.py
# One figure for the README: how much of my original R^2 was overlap, and what a proper
# benchmark and order flow add on the clean target.
# Run replicate_original.py and har_models.py first (they save the numbers in data/).

import pandas as pd
import matplotlib.pyplot as plt

rep = pd.read_csv("data/replication.csv")
har = pd.read_csv("data/har_results.csv", index_col="model")
h = [5, 10, 15]

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(h, rep["original"], "o--", color="grey", label="original study (target overlaps feature)")
ax.plot(h, rep["clean"], "o-", label="original features, clean target")
ax.plot(h, har.loc["HAR", ["5 min", "10 min", "15 min"]], "s-", label="HAR benchmark")
ax.plot(h, har.loc["HAR + hour + order flow", ["5 min", "10 min", "15 min"]], "^:",
        label="HAR + hour + order flow")

ax.set_xticks(h)
ax.set_xlabel("Forecast horizon (minutes)")
ax.set_ylabel("Out-of-sample R²")
ax.set_ylim(0, 0.85)
ax.set_title("Most of the original result was overlap; order flow adds nothing", loc="left")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("figures/r2_comparison.png", dpi=150)
