"""Step 1 - Download the dataset and record its checksums.

Downloads `flwrlabs/shakespeare` with the `datasets` library, saves a local copy
as Parquet, and writes a SHA256 checksum plus a small metadata file so the data
can be verified later.

Run with:

    python -m src.download
    python -m src.download --force    # download again even if a copy exists
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.config import (
    CHECKSUMS_FILE,
    DATASET_META_FILE,
    HF_DATASET,
    HF_REVISION,
    RAW_PARQUET,
    SEED,
    ensure_dirs,
    set_seeds,
)

set_seeds(SEED)


def download_dataset() -> pd.DataFrame:
    """Download the dataset from the Hub and return it as a DataFrame.

    The revision is pinned, so a later change upstream cannot silently change
    our data.
    """
    print(f"[download] {HF_DATASET} (revision {HF_REVISION[:12]})")
    dataset = load_dataset(HF_DATASET, split="train", revision=HF_REVISION)
    df = dataset.to_pandas()
    print(f"[download] {len(df):,} rows, columns: {list(df.columns)}")
    return df


def save_parquet(df: pd.DataFrame, path: Path = RAW_PARQUET) -> None:
    """Save the DataFrame as Parquet in data/raw/."""
    df.to_parquet(path, index=False)
    print(f"[save] {path.name} ({path.stat().st_size:,} bytes)")


def sha256_file(path: Path) -> str:
    """Return the SHA256 checksum of a file, reading it in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(paths: list[Path], out_path: Path = CHECKSUMS_FILE) -> dict[str, str]:
    """Write a SHA256 checksum file for the downloaded artefacts.

    The dataset provider does not publish checksums, so these are ours: they let
    us detect a corrupted or accidentally modified local copy. Verify later with
    `sha256sum -c CHECKSUMS.txt`.
    """
    checksums = {path.name: sha256_file(path) for path in paths if path.exists()}

    header = (
        "# SHA256 checksums for the files in data/raw/\n"
        f"# generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"# source:    {HF_DATASET} (revision {HF_REVISION})\n"
        "#\n"
        "# The dataset provider publishes no checksums, so these are generated\n"
        "# locally to detect corruption of our own copy.\n"
        "# Verify with:  sha256sum -c CHECKSUMS.txt\n"
    )
    lines = [f"{value}  {name}" for name, value in checksums.items()]
    out_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"[checksums] wrote {out_path.name}")
    return checksums


def write_metadata(df: pd.DataFrame, checksums: dict[str, str]) -> dict:
    """Write dataset provenance and shape to data/raw/dataset_meta.json."""
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": HF_DATASET,
        "revision": HF_REVISION,
        "url": f"https://huggingface.co/datasets/{HF_DATASET}",
        "license": "BSD-2-Clause",
        "benchmark": "LEAF (arXiv:1812.01097)",
        "n_rows": len(df),
        "columns": {name: str(dtype) for name, dtype in df.dtypes.items()},
        "parquet_size_bytes": RAW_PARQUET.stat().st_size,
        "checksums": checksums,
        "seed": SEED,
    }
    DATASET_META_FILE.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[metadata] wrote {DATASET_META_FILE.name}")
    return meta


def main(argv: list[str] | None = None) -> int:
    """Run step 1."""
    parser = argparse.ArgumentParser(
        prog="python -m src.download",
        description="Download flwrlabs/shakespeare and save it to data/raw/.",
    )
    parser.add_argument(
        "--force", action="store_true", help="download again even if a copy exists"
    )
    args = parser.parse_args(argv)

    ensure_dirs()

    if RAW_PARQUET.exists() and not args.force:
        print(f"[skip] {RAW_PARQUET} already exists (use --force to redownload)")
        df = pd.read_parquet(RAW_PARQUET)
    else:
        df = download_dataset()
        save_parquet(df)

    checksums = write_checksums([RAW_PARQUET])
    write_metadata(df, checksums)

    print("\n[done] step 1 complete. Next: python dataset_info.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
