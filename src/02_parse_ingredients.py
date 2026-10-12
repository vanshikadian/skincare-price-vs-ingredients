"""
02_parse_ingredients.py

Purpose: turn the ingredient text into clean lists and a 0/1 feature matrix.
Reads:   data/processed/products_clean.csv
Writes:  data/processed/ingredient_lists.csv, data/processed/ingredient_matrix.csv
Supports: feature engineering (complex aspect)

List-picking rule: the raw column is a Python list written as a string. When it
has several elements (labels, marketing text, per-variant lists), we keep the
LONGEST element, since the real ingredient list is almost always the longest
piece of text. Labels ending in ':' are ignored.
"""
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 810
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed" / "products_clean.csv"
OUT_LISTS = ROOT / "data" / "processed" / "ingredient_lists.csv"
OUT_MATRIX = ROOT / "data" / "processed" / "ingredient_matrix.csv"
MIN_PRODUCTS = 50
WATER_WORDS = {"water", "aqua", "eau", "agua"}


def pick_list(raw):
    """Parse the raw string into a list and return the single text to use."""
    try:
        parts = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw
    if isinstance(parts, str):
        return parts
    parts = [p for p in parts if isinstance(p, str) and not p.strip().endswith(":")]
    return max(parts, key=len) if parts else ""


def split_outside_parens(text):
    """Split text on commas that are not inside parentheses."""
    items, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        if ch == "," and depth == 0:
            items.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    items.append("".join(cur))
    return items


def clean_item(item):
    """Lowercase, strip, drop a trailing period, merge water synonyms."""
    item = re.sub(r"\s+", " ", item.strip().lower()).rstrip(".").strip()
    words = set(re.findall(r"[a-z]+", item))
    if words and words <= WATER_WORDS:
        return "water"
    return item


def parse(raw):
    """Full pipeline for one product: raw string -> list of clean ingredients."""
    items = [clean_item(i) for i in split_outside_parens(pick_list(raw))]
    seen, out = set(), []
    for i in items:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def main():
    """Parse every product, build the 0/1 matrix, print checks."""
    df = pd.read_csv(IN)
    df["ingredient_list"] = df["ingredients"].apply(parse)
    df["n_ingredients"] = df["ingredient_list"].apply(len)

    print("products:", len(df))
    print("ingredient count: min", df["n_ingredients"].min(),
          "median", df["n_ingredients"].median(), "max", df["n_ingredients"].max())
    print("products with <5 ingredients (check these):",
          (df["n_ingredients"] < 5).sum())

    counts = pd.Series([i for lst in df["ingredient_list"] for i in lst]).value_counts()
    keep = counts[counts >= MIN_PRODUCTS].index.tolist()
    print(f"distinct ingredient names: {len(counts)}; "
          f"kept (in >= {MIN_PRODUCTS} products): {len(keep)}")

    matrix = pd.DataFrame(
        {ing: df["ingredient_list"].apply(lambda l, g=ing: int(g in l)) for ing in keep}
    )
    matrix.insert(0, "product_id", df["product_id"].values)

    out = df[["product_id", "n_ingredients"]].copy()
    out["ingredient_list"] = df["ingredient_list"].apply("|".join)
    out.to_csv(OUT_LISTS, index=False)
    matrix.to_csv(OUT_MATRIX, index=False)

    print("\n20 random products for an eyeball check:")
    for _, r in df.sample(20, random_state=SEED).iterrows():
        print(f"- {r['product_name'][:45]} ({r['n_ingredients']}): "
              f"{r['ingredient_list'][:6]}")


if __name__ == "__main__":
    main()
