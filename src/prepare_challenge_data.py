"""Adapt the public 2025-style catalog CSV schema to the starter's canonical schema.

Input columns:
  train: sample_id, catalog_content, image_link, price
  test: sample_id, catalog_content, image_link
  sample submission: sample_id, price

The adapter preserves catalog_content as a whole in TITLE. It does not pretend
to recover separate title, description, brand, category, or pack-size fields;
the existing pipeline receives empty placeholders for those optional fields.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path: Path, required: set[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    frame = pd.read_csv(path, dtype={"sample_id": "string"})
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required column(s): {missing}")
    if frame.empty:
        raise ValueError(f"{label} contains no rows.")
    ids = frame["sample_id"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any():
        raise ValueError(f"{label} contains missing or blank sample_id values.")
    duplicated = ids.duplicated(keep=False)
    if duplicated.any():
        examples = ids[duplicated].head(5).tolist()
        raise ValueError(f"{label} contains duplicate sample_id values, e.g. {examples}.")
    frame["sample_id"] = ids
    return frame


def adapt_frames(
    train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate frames and map the competition columns to pipeline defaults."""
    for label, frame, required in (
        ("train", train, {"sample_id", "catalog_content", "image_link", "price"}),
        ("test", test, {"sample_id", "catalog_content", "image_link"}),
        ("sample submission", sample, {"sample_id", "price"}),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} is missing required column(s): {missing}")
        ids = frame["sample_id"].astype("string").str.strip()
        if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
            raise ValueError(f"{label} sample_id values must be nonblank and unique.")

    if not test["sample_id"].reset_index(drop=True).equals(
        sample["sample_id"].reset_index(drop=True)
    ):
        raise ValueError("test and sample submission IDs must match exactly and in the same order.")

    price = pd.to_numeric(train["price"], errors="coerce")
    values = price.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("train.price must contain finite, non-negative numeric values.")

    def catalog_frame(frame: pd.DataFrame, *, has_price: bool) -> pd.DataFrame:
        catalog = frame["catalog_content"].fillna("").astype("string")
        result = pd.DataFrame({
            "PRODUCT_ID": frame["sample_id"].astype("string"),
            "TITLE": catalog,
            "DESCRIPTION": "",
            "PACK_SIZE": "",
            "BRAND": "",
            "CATEGORY": "",
            "IMAGE_URL": frame["image_link"].fillna("").astype("string"),
        })
        if has_price:
            result["PRICE"] = pd.to_numeric(frame["price"], errors="raise").astype(float)
        return result

    canonical_train = catalog_frame(train, has_price=True)
    canonical_test = catalog_frame(test, has_price=False)
    canonical_sample = pd.DataFrame({
        "PRODUCT_ID": sample["sample_id"].astype("string"),
        "PRICE": pd.to_numeric(sample["price"], errors="coerce"),
    })
    return canonical_train, canonical_test, canonical_sample


def adapt_files(
    train_path: Path, test_path: Path, sample_path: Path, output_dir: Path, *, overwrite: bool = False
) -> tuple[Path, Path, Path]:
    train = _read(train_path, {"sample_id", "catalog_content", "image_link", "price"}, "train")
    test = _read(test_path, {"sample_id", "catalog_content", "image_link"}, "test")
    sample = _read(sample_path, {"sample_id", "price"}, "sample submission")
    frames = adapt_frames(train, test, sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = tuple(output_dir / name for name in (
        "train.csv", "test.csv", "sample_submission.csv"
    ))
    existing = [str(path) for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): " + ", ".join(existing)
            + ". Pass --overwrite to replace them."
        )

    staged: list[Path] = []
    try:
        for frame, destination in zip(frames, destinations):
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", suffix=".tmp",
                dir=output_dir, delete=False,
            ) as handle:
                temporary = Path(handle.name)
                frame.to_csv(handle, index=False)
            staged.append(temporary)
        for temporary, destination in zip(staged, destinations):
            os.replace(temporary, destination)
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
    return destinations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path, help="Raw competition train CSV")
    parser.add_argument("--test", required=True, type=Path, help="Raw competition test CSV")
    parser.add_argument("--sample", required=True, type=Path, help="Raw sample submission CSV")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace canonical output files")
    args = parser.parse_args()
    paths = adapt_files(args.train, args.test, args.sample, args.output_dir, overwrite=args.overwrite)
    print("Canonical files written:")
    for path in paths:
        print(f"  {path}")
    print("Next: use --train, --test, and --sample with the pipeline; retain raw files unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
