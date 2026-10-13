"""
07_describe_styled.py

Purpose: the four-panel descriptive figure, restyled in the slide deck's colors
         and sized for a 16:9 slide.
Reads:   data/processed/products_clean.csv, results/tier_ci.csv
Writes:  figures/fig_descriptive_slides.png
Note:    same content as 03_describe.py, colors and sizing changed for the
         presentation. Kept as a separate file so 03 stays the paper version.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CREAM = "#FFFBEC"
GREEN = "#C0DB41"
INK = "#2B2B2B"

# Bigger text so the panels stay readable from the back of the room.
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
})

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "processed" / "products_clean.csv")
tab = pd.read_csv(ROOT / "results" / "tier_ci.csv")

r = df["log_price"].corr(df["rating"])

# 16:9-ish canvas so the figure fills the slide instead of leaving side margins.
fig, ax = plt.subplots(2, 2, figsize=(15, 8.4), facecolor=CREAM)

ax[0, 0].hist(df["price_usd"], bins=40, color=GREEN, edgecolor=INK, linewidth=0.5)
ax[0, 0].set(title="Price is right-skewed", xlabel="Price (USD)", ylabel="Products")

ax[0, 1].hist(df["log_price"], bins=40, color=GREEN, edgecolor=INK, linewidth=0.5)
ax[0, 1].set(title="Log price is closer to symmetric (we model this)",
             xlabel="log(price)", ylabel="Products")

ax[1, 0].scatter(df["log_price"], df["rating"], s=9, alpha=0.3, color=INK)
ax[1, 0].set(title=f"Price and rating barely move together (r = {r:.2f})",
             xlabel="log(price)", ylabel="Rating (stars)")

ax[1, 1].errorbar(range(len(tab)), tab["mean_rating"],
                  yerr=[tab["mean_rating"] - tab["ci_low"],
                        tab["ci_high"] - tab["mean_rating"]],
                  fmt="o", capsize=7, color=INK, ecolor=INK,
                  markerfacecolor=GREEN, markeredgecolor=INK,
                  markersize=13, markeredgewidth=1.5, linewidth=2.5)
ax[1, 1].set_xticks(range(len(tab)), tab["tier"])
ax[1, 1].set(title="Mean rating by price quartile, 95% CI",
             xlabel="Price tier", ylabel="Mean rating (stars)")

for a in ax.flat:
    a.set_facecolor(CREAM)
    a.tick_params(colors=INK)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        a.spines[s].set_color(INK)

fig.tight_layout(pad=1.6)
fig.savefig(ROOT / "figures" / "fig_descriptive_slides.png", dpi=200, facecolor=CREAM)
print("saved figures/fig_descriptive_slides.png")