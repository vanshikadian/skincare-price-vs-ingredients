"""
03_describe.py

Purpose: describe price, rating and review counts. First figures for the paper.
Reads:   data/processed/products_clean.csv
Writes:  figures/fig_descriptive.png, results/tier_ci.csv
Supports: confidence intervals and hypothesis tests (lecture Oct 19)

We model LOG price: raw price is heavily right-skewed ($3 to $449), while log
price is roughly symmetric and makes errors proportional (a $5 miss matters
more on a $20 product than on a $200 one).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SEED = 810
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed" / "products_clean.csv"
FIG = ROOT / "figures" / "fig_descriptive.png"
TIERS = ROOT / "results" / "tier_ci.csv"


def tier_table(df):
    """Mean rating and a hand-computed 95% t interval for each price tier."""
    rows = []
    for tier, g in df.groupby("tier", observed=True):
        n, mean, sd = len(g), g["rating"].mean(), g["rating"].std(ddof=1)
        se = sd / np.sqrt(n)
        t = stats.t.ppf(0.975, n - 1)
        rows.append({"tier": tier, "n": n, "mean_rating": mean,
                     "ci_low": mean - t * se, "ci_high": mean + t * se})
    return pd.DataFrame(rows)


def main():
    """Make the four-panel descriptive figure and the tier table."""
    df = pd.read_csv(IN)
    df["tier"] = pd.qcut(df["price_usd"], 4, labels=["Q1 cheapest", "Q2", "Q3", "Q4 priciest"])

    r = df["log_price"].corr(df["rating"])
    print(f"corr(log price, rating) = {r:.3f}")

    tab = tier_table(df)
    print(tab.round(3).to_string(index=False))
    TIERS.parent.mkdir(exist_ok=True)
    tab.to_csv(TIERS, index=False)

    lo = df.loc[df["tier"] == "Q1 cheapest", "rating"]
    hi = df.loc[df["tier"] == "Q4 priciest", "rating"]
    t, p = stats.ttest_ind(hi, lo, equal_var=False)
    print(f"Welch t-test, top vs bottom tier: diff = {hi.mean() - lo.mean():.3f}, "
          f"t = {t:.2f}, p = {p:.4f}")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].hist(df["price_usd"], bins=40, color="#4c78a8")
    ax[0, 0].set(title="Price is right-skewed", xlabel="Price (USD)", ylabel="Products")
    ax[0, 1].hist(df["log_price"], bins=40, color="#4c78a8")
    ax[0, 1].set(title="Log price is closer to symmetric (we model this)",
                 xlabel="log(price)", ylabel="Products")
    ax[1, 0].scatter(df["log_price"], df["rating"], s=6, alpha=0.4, color="#4c78a8")
    ax[1, 0].set(title=f"Price and rating barely move together (r = {r:.2f})",
                 xlabel="log(price)", ylabel="Rating (stars)")
    ax[1, 1].errorbar(range(len(tab)), tab["mean_rating"],
                      yerr=[tab["mean_rating"] - tab["ci_low"], tab["ci_high"] - tab["mean_rating"]],
                      fmt="o", capsize=5, color="#4c78a8")
    ax[1, 1].set_xticks(range(len(tab)), tab["tier"])
    ax[1, 1].set(title="Mean rating by price quartile, 95% CI",
                 xlabel="Price tier", ylabel="Mean rating (stars)")
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print("saved", FIG)


if __name__ == "__main__":
    main()
