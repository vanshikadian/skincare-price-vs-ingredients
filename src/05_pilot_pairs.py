"""
05_pilot_pairs.py

Purpose: count how many near-identical-ingredient product pairs exist, to check
the placebo test is feasible before building it.
Reads:   data/processed/products_clean.csv, data/processed/ingredient_lists.csv
Writes:  results/pair_counts.csv
Supports: hypothesis tests, Monte Carlo (later)

Price is compared PER OUNCE so a pair is not "different" just because one
product is bigger. Pairs are reported overall and for different brands only
(same-brand pairs are mostly size variants of one formula).

A product can sit in many pairs, so pairs are not independent. For the real
test we will either keep one pair per product or resample by brand.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 810
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
MIN_RATIO = 1.5


def jaccard(a, b):
    """Shared ingredients divided by all distinct ingredients across two sets."""
    return len(a & b) / len(a | b)


def find_pairs(df, threshold):
    """Return pairs within a secondary category with Jaccard >= threshold and price ratio >= MIN_RATIO."""
    out = []
    for _, g in df.groupby("secondary_category"):
        for i, j in combinations(g.index, 2):
            pa, pb = g.at[i, "ppo"], g.at[j, "ppo"]
            if np.isnan(pa) or np.isnan(pb):
                continue
            ratio = max(pa, pb) / min(pa, pb)
            if ratio < MIN_RATIO:
                continue
            s = jaccard(g.at[i, "ing_set"], g.at[j, "ing_set"])
            if s >= threshold:
                out.append((g.at[i, "product_id"], g.at[j, "product_id"], s, ratio))
    return pd.DataFrame(out, columns=["id_a", "id_b", "jaccard", "price_ratio"])


def main():
    """Count pairs and distinct products at Jaccard 0.8 and 0.7."""
    df = pd.read_csv(PROC / "products_clean.csv")
    lists = pd.read_csv(PROC / "ingredient_lists.csv")
    df = df.merge(lists[["product_id", "ingredient_list"]], on="product_id")
    df["ppo"] = df["price_usd"] / df["size_oz"]
    df["ing_set"] = df["ingredient_list"].apply(lambda s: set(s.split("|")))

    rows = []
    for thr in (0.8, 0.7):
        pairs = find_pairs(df, thr)
        prods = pd.concat([pairs["id_a"], pairs["id_b"]]).nunique()
        brands_same = 0
        if len(pairs):
            b = df.set_index("product_id")["brand_name"]
            brands_same = int((pairs["id_a"].map(b) == pairs["id_b"].map(b)).sum())
        cross = len(pairs) - brands_same
        print(f"Jaccard >= {thr}: {len(pairs)} pairs, {prods} distinct products, "
              f"{brands_same} same-brand, {cross} cross-brand")
        rows.append({"jaccard": thr, "pairs": len(pairs), "distinct_products": prods,
                     "same_brand_pairs": brands_same, "cross_brand_pairs": cross})
        if cross < 100:
            print("  fewer than ~100 pairs: consider relaxing the price ratio or Jaccard cutoff")
    (ROOT / "results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "pair_counts.csv", index=False)


if __name__ == "__main__":
    main()
