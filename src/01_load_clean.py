"""
01_load_clean.py

Purpose: load the Sephora product file and keep usable skincare products.
Reads:   data/raw/product_info.csv
Writes:  data/processed/products_clean.csv
Supports: data prep; feature engineering (log price, size in ounces)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 810
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "product_info.csv"
OUT = ROOT / "data" / "processed" / "products_clean.csv"
ML_PER_OZ = 29.5735


def size_to_oz(text):
    """Convert a size string such as '1.7 oz/ 50 mL' to ounces.

    Uses the oz number when present, mL / 29.5735 when only mL is present,
    and returns NaN for sizes like '30 Capsules'.
    """
    if not isinstance(text, str):
        return np.nan
    oz = re.search(r"(\d+(?:\.\d+)?)\s*oz", text, flags=re.I)
    if oz:
        return float(oz.group(1))
    ml = re.search(r"(\d+(?:\.\d+)?)\s*mL", text, flags=re.I)
    if ml:
        return float(ml.group(1)) / ML_PER_OZ
    return np.nan


def main():
    """Run the cleaning steps, print counts at each stage, save the table."""
    df = pd.read_csv(RAW)
    print("raw rows:", len(df))

    df = df[df["primary_category"] == "Skincare"]
    print("skincare:", len(df), "(expect 2,420)")

    df = df[~df["secondary_category"].isin(["Value & Gift Sets", "Mini Size"])]
    print("no gift sets / minis:", len(df), "(expect 2,112)")

    df = df.dropna(subset=["price_usd", "ingredients", "rating"])
    print("no missing price/ingredients/rating:", len(df), "(expect 1,951)")

    # Minis are also hiding in the regular categories, flagged only by name
    is_mini = df["product_name"].str.contains(r"\bmini\b", case=False, regex=True)
    print("name says mini (dropped):", is_mini.sum())
    df = df[~is_mini].copy()
    print("after dropping minis by name:", len(df))
    df["size_oz"] = df["size"].apply(size_to_oz)
    print("missing size_oz:", df["size_oz"].isna().sum())
    df["log_price"] = np.log(df["price_usd"])
    df["log_size"] = np.log(df["size_oz"])

    top = df.sort_values("price_usd").iloc[-1]
    print("most expensive:", top["product_name"], top["price_usd"])
    print("brands:", df["brand_name"].nunique())
    print("products with 26+ reviews:", (df["reviews"] >= 26).sum())

    cols = ["product_id", "product_name", "brand_name", "primary_category",
            "secondary_category", "tertiary_category", "price_usd", "log_price",
            "size", "size_oz", "log_size", "rating", "reviews", "loves_count",
            "ingredients"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(OUT, index=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
