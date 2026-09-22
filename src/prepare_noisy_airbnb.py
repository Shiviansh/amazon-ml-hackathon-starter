"""Prepare the Inside Airbnb NYC snapshot as a noisy regression challenge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "id",
    "name",
    "host_id",
    "neighbourhood_group",
    "neighbourhood",
    "latitude",
    "longitude",
    "room_type",
    "price",
}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def prepare(source: Path, output_directory: Path, source_url: str) -> dict:
    data = pd.read_csv(source)
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"Source data is missing required columns: {missing}.")
    if data.empty or data["id"].isna().any() or not data["id"].is_unique:
        raise ValueError("Source IDs must be non-missing and unique.")
    price = pd.to_numeric(data["price"], errors="coerce")
    non_numeric = int((price.isna() & data["price"].notna()).sum())
    if non_numeric or np.isinf(price.dropna().to_numpy(dtype=float)).any():
        raise ValueError("Price contains non-numeric or infinite populated values.")

    train = data.loc[price.notna()].copy()
    train["price"] = price.loc[price.notna()].astype(float)
    test = data.loc[price.isna()].drop(columns="price").copy()
    sample = test.loc[:, ["id"]].copy()
    sample["price"] = 0.0
    if train.empty or test.empty:
        raise ValueError("Expected both labeled and naturally unlabeled source rows.")

    output_directory.mkdir(parents=True, exist_ok=True)
    _atomic_csv(train, output_directory / "train.csv")
    _atomic_csv(test, output_directory / "test.csv")
    _atomic_csv(sample, output_directory / "sample_submission.csv")

    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    target = train["price"]
    manifest = {
        "dataset": "Inside Airbnb New York City summary listings",
        "source_url": source_url,
        "source_file": source.name,
        "source_sha256": source_hash,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "train": "rows where price is populated",
            "test": "naturally unlabeled rows where price is missing",
            "cleaning": "none beyond numeric price validation",
        },
        "rows": {
            "source": int(len(data)),
            "train": int(len(train)),
            "test": int(len(test)),
        },
        "columns": {
            "train": list(train.columns),
            "test": list(test.columns),
            "id": "id",
            "target": "price",
        },
        "noise_evidence": {
            "source_missing_values": {
                column: int(value)
                for column, value in data.isna().sum().sort_values(ascending=False).items()
                if value
            },
            "target_min": float(target.min()),
            "target_median": float(target.median()),
            "target_p99": float(target.quantile(0.99)),
            "target_max": float(target.max()),
            "duplicate_non_id_rows": int(
                data.drop(columns=["id"]).duplicated(keep=False).sum()
            ),
        },
    }
    manifest_path = output_directory / "dataset_manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-url", required=True)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output_dir, args.source_url)
    print(json.dumps(manifest["rows"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
