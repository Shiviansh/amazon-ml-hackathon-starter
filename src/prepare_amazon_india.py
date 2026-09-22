"""Download, clean, and partition the Amazon India 30k e-commerce dataset.

This produces a real-world, highly noisy e-commerce catalog benchmark matching
the exact problem structure of the Amazon ML Challenge:
- High-cardinality catalog text (TITLE, DESCRIPTION, PACK_SIZE)
- High-cardinality categoricals (BRAND, CATEGORY)
- Multimodal link assets (IMAGE_URL)
- Heavily right-skewed pricing target (PRICE)
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HF_URL = (
    "https://huggingface.co/datasets/pgurazada1/amazon_india_products"
    "/resolve/main/marketing_sample_for_amazon_in-ecommerce__20191001_20191031__30k_data.csv"
)

COLUMN_MAPPING = {
    "Uniq Id": "PRODUCT_ID",
    "Product Title": "TITLE",
    "Product Description": "DESCRIPTION",
    "Brand": "BRAND",
    "Category": "CATEGORY",
    "Pack Size Or Quantity": "PACK_SIZE",
    "Image Urls": "IMAGE_URL",
    "Price": "PRICE",
}


def parse_price(val: Any) -> float:
    """Parse raw noisy price strings into positive floats."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", "").replace("₹", "").replace("Rs.", "").replace("Rs", "").strip()
    if s.startswith("."):
        s = s[1:]
    try:
        f = float(s)
        return f if f > 0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def download_and_prepare(
    raw_dir: Path = Path("data/raw"),
    archive_dir: Path = Path("data/archive/amazon_india_products"),
    random_state: int = 42,
    train_ratio: float = 0.75,
) -> dict[str, Any]:
    """Download, preprocess, and partition Amazon India catalog dataset."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    cache_file = archive_dir / "raw_amazon_in_30k.csv"
    if not cache_file.exists():
        print(f"Downloading raw dataset from Hugging Face...")
        urllib.request.urlretrieve(HF_URL, cache_file)
        print(f"Downloaded and cached to {cache_file}")
    else:
        print(f"Using cached raw file: {cache_file}")

    print("Loading raw CSV...")
    df_raw = pd.read_csv(cache_file)
    print(f"Raw shape: {df_raw.shape}")

    # Keep only target columns and map to standard Amazon ML Challenge schema
    keep_cols = [c for c in COLUMN_MAPPING if c in df_raw.columns]
    df = df_raw[keep_cols].rename(columns=COLUMN_MAPPING).copy()

    # Clean price
    df["PRICE_CLEAN"] = df["PRICE"].apply(parse_price)

    # Separate rows with valid ground-truth price from missing price
    has_price = df["PRICE_CLEAN"].notna()
    labeled_df = df[has_price].copy()
    unlabeled_df = df[~has_price].copy()

    print(f"Valid labeled rows: {len(labeled_df)}")
    print(f"Unlabeled / missing price rows in scrape: {len(unlabeled_df)}")

    # Shuffle labeled rows deterministically
    shuffled = labeled_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n_train = int(len(shuffled) * train_ratio)

    train_part = shuffled.iloc[:n_train].copy()
    test_labeled_part = shuffled.iloc[n_train:].copy()

    # Test set includes the held-out labeled sample + naturally unlabeled rows
    test_part = pd.concat([test_labeled_part, unlabeled_df], ignore_index=True)
    # Shuffle test set so unlabeled rows aren't clustered at the end
    test_part = test_part.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # In train, assign cleaned PRICE
    train_df = train_part.copy()
    train_df["PRICE"] = train_df["PRICE_CLEAN"]
    train_df = train_df.drop(columns=["PRICE_CLEAN"])

    # In test, keep features only (withhold PRICE)
    test_features = test_part.drop(columns=["PRICE", "PRICE_CLEAN"]).copy()

    # Sample submission: PRODUCT_ID and placeholder PRICE
    sample_submission = pd.DataFrame({
        "PRODUCT_ID": test_features["PRODUCT_ID"],
        "PRICE": 0.0,
    })

    # Save test ground truth in archive for holdout validation
    test_truth = test_part[["PRODUCT_ID", "PRICE_CLEAN"]].rename(columns={"PRICE_CLEAN": "PRICE"})
    test_truth.to_csv(archive_dir / "test_ground_truth.csv", index=False)

    # Save active raw files
    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"
    sub_path = raw_dir / "sample_submission.csv"

    train_df.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)
    sample_submission.to_csv(sub_path, index=False)

    stats = {
        "dataset_name": "Amazon India Products (E-Commerce Catalog)",
        "total_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_features),
        "test_labeled_rows": len(test_labeled_part),
        "train_price_min": float(train_df["PRICE"].min()),
        "train_price_median": float(train_df["PRICE"].median()),
        "train_price_mean": float(train_df["PRICE"].mean()),
        "train_price_max": float(train_df["PRICE"].max()),
        "unique_brands": int(df["BRAND"].nunique()),
        "unique_categories": int(df["CATEGORY"].nunique()),
        "missing_descriptions": int(df["DESCRIPTION"].isna().sum()),
        "missing_pack_sizes": int(df["PACK_SIZE"].isna().sum()),
    }

    manifest_path = archive_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n--- Preparation Summary ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nSaved train to: {train_path}")
    print(f"Saved test to: {test_path}")
    print(f"Saved sample submission to: {sub_path}")
    print(f"Saved test ground truth to: {archive_dir / 'test_ground_truth.csv'}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Amazon India e-commerce dataset.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--archive-dir", type=Path, default=Path("data/archive/amazon_india_products"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    args = parser.parse_args()

    download_and_prepare(
        raw_dir=args.raw_dir,
        archive_dir=args.archive_dir,
        random_state=args.random_state,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
