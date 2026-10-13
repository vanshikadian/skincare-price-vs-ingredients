"""
06_tier_figure.py

Purpose: standalone mean-rating-by-price-tier figure, styled for the slides.
Reads:   results/tier_ci.csv
Writes:  figures/fig_tier.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CREAM = "#FFFBEC"
GREEN = "#C0DB41"
INK = "#2B2B2B"

ROOT = Path(__file__).resolve().parent.parent
tab = pd.read_csv(ROOT / "results" / "tier_ci.csv")

fig, ax = plt.subplots(figsize=(7, 5), facecolor=CREAM)
ax.set_facecolor(CREAM)
ax.errorbar(range(len(tab)), tab["mean_rating"],
            yerr=[tab["mean_rating"] - tab["ci_low"],
                  tab["ci_high"] - tab["mean_rating"]],
            fmt="o", capsize=6, color=INK, ecolor=INK,
            markerfacecolor=GREEN, markeredgecolor=INK,
            markersize=12, markeredgewidth=1.5, linewidth=2)
ax.set_xticks(range(len(tab)), tab["tier"])
ax.set(title="Mean rating by price quartile, 95% CI",
       xlabel="Price tier", ylabel="Mean rating (stars)")
ax.tick_params(colors=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("bottom", "left"):
    ax.spines[s].set_color(INK)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig_tier.png", dpi=200, facecolor=CREAM)
print("saved figures/fig_tier.png")