"""
04_baseline_ols.py

Purpose: hand-built OLS for log price, and the first rungs of the R-squared ladder.
Reads:   data/processed/products_clean.csv, data/processed/ingredient_matrix.csv
Writes:  results/ladder.csv, figures/fig_ladder.png
Supports: regression with categorical variables, hand-created model (complex aspect)

All rungs use the same products (those with a known size) and the same
train/test split, so the R-squared values are comparable. Brand comes later.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

SEED = 810
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"


def ols_fit(X, y):
    """Solve the normal equations (X'X) beta = X'y with NumPy. Returns beta."""
    return np.linalg.pinv(X.T @ X) @ (X.T @ y)


def r2(y, yhat):
    """Coefficient of determination."""
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    """Verify the hand OLS against a library, then fit the ladder rungs."""
    df = pd.read_csv(PROC / "products_clean.csv")
    ing = pd.read_csv(PROC / "ingredient_matrix.csv")
    counts = pd.read_csv(PROC / "ingredient_lists.csv")[["product_id", "n_ingredients"]]
    df = df.merge(counts, on="product_id").merge(ing, on="product_id")
    df = df.dropna(subset=["log_size"]).reset_index(drop=True)
    print("products with a size:", len(df))

    ing_cols = [c for c in ing.columns if c != "product_id"]
    cat = pd.get_dummies(df["secondary_category"], drop_first=True).astype(float)
    y = df["log_price"].to_numpy()
    one = np.ones((len(df), 1))

    # Hand OLS vs library on the size model (full data)
    X = np.hstack([one, cat.to_numpy(), df[["log_size"]].to_numpy()])
    beta = ols_fit(X, y)
    lib = LinearRegression(fit_intercept=False).fit(X, y)
    print("largest |hand - library| coefficient difference:",
          f"{np.abs(beta - lib.coef_).max():.2e}")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    cut = int(0.8 * len(df))
    tr, te = idx[:cut], idx[cut:]

    rungs = {
        "Category": [one, cat.to_numpy()],
        "+ Size": [one, cat.to_numpy(), df[["log_size"]].to_numpy()],
        "+ Ingredient count": [one, cat.to_numpy(), df[["log_size", "n_ingredients"]].to_numpy()],
        "+ Ingredient matrix": [one, cat.to_numpy(),
                                df[["log_size", "n_ingredients"]].to_numpy(),
                                df[ing_cols].to_numpy(float)],
    }
    rows = []
    for name, parts in rungs.items():
        X = np.hstack(parts)
        b = ols_fit(X[tr], y[tr])
        rows.append({"rung": name, "n_features": X.shape[1] - 1,
                     "train_r2": r2(y[tr], X[tr] @ b), "test_r2": r2(y[te], X[te] @ b)})
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    (ROOT / "results").mkdir(exist_ok=True)
    res.to_csv(ROOT / "results" / "ladder.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(res["rung"], res["test_r2"], color="#4c78a8")
    for i, v in enumerate(res["test_r2"]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    ax.set(title="How much of log price each piece explains (test R-squared)",
           xlabel="Predictors added", ylabel="Test R-squared")
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "fig_ladder.png", dpi=150)
    print("saved figures/fig_ladder.png")


if __name__ == "__main__":
    main()
