"""Print a summary of the downloaded dataset.

Reads data/raw/shakespeare.parquet and prints: number of rows, data types, the
number of target classes and their counts, how many distinct character_id values
there are with their min/mean/max number of samples, the distribution of x
lengths, and the number of missing or empty values.

All figures cover the full dataset; nothing is sampled.

Usage:

    python dataset_info.py
    python dataset_info.py --top 20
    python dataset_info.py --all-classes
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.config import DATASET_META_FILE, RAW_PARQUET, SEED, set_seeds
from src.dataset import load_raw, summarize

set_seeds(SEED)


def show_char(char: str) -> str:
    """Make a character class readable in the terminal."""
    names = {" ": "' ' (space)", "\t": "'\\t' (tab)", "\n": "'\\n' (newline)"}
    return names.get(char, f"'{char}'")


def title(text: str) -> None:
    """Print a section heading."""
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def show_provenance() -> None:
    """Where the data came from."""
    if not DATASET_META_FILE.exists():
        return
    meta = json.loads(DATASET_META_FILE.read_text(encoding="utf-8"))
    title("0. WHERE THE DATA CAME FROM")
    print(f"dataset   : {meta['dataset']} ({meta['url']})")
    print(f"revision  : {meta['revision']}")
    print(f"license   : {meta['license']}")
    print(f"benchmark : {meta['benchmark']}")
    for name, checksum in meta["checksums"].items():
        print(f"sha256    : {checksum}  {name}")


def show_overview(summary: dict) -> None:
    """Rows, size and data types."""
    title("1. OVERVIEW")
    print(f"rows           : {summary['n_rows']:,}")
    print(f"columns        : {summary['n_columns']}")
    print(f"memory         : {summary['memory_bytes'] / 2**20:,.1f} MiB")
    print(f"parquet on disk: {RAW_PARQUET.stat().st_size / 2**20:,.1f} MiB")
    print("\ndata types:")
    for name, dtype in summary["schema"].items():
        print(f"  {name:<16} {dtype}")


def show_target(summary: dict, top: int, show_all: bool) -> None:
    """Target classes, their counts and what they imply."""
    counts = summary["target"]["class_counts"]
    total = summary["n_rows"]
    shares = np.array(list(counts.values())) / total
    entropy = float(-(shares * np.log2(shares)).sum())

    top_char = next(iter(counts))
    top_count = counts[top_char]

    title("2. TARGET y")
    print(f"number of classes    : {summary['target']['n_classes']}")
    print(f"most frequent class  : {show_char(top_char)} "
          f"({top_count:,} rows, {top_count / total:.4%})")
    print(f"imbalance ratio      : {top_count / min(counts.values()):,.0f}x "
          "(most frequent / least frequent)")
    print(f"entropy              : {entropy:.4f} bits "
          f"(would be {np.log2(len(counts)):.4f} if all classes were equal)")
    print(f"always-predict-space : {top_count / total:.4%} accuracy "
          "<- the score to beat")

    items = list(counts.items())
    print(f"\n{'class':<16}{'count':>14}{'share':>12}")
    print("-" * 42)
    for char, count in (items if show_all else items[:top]):
        print(f"{show_char(char):<16}{count:>14,}{count / total:>11.4%}")
    if not show_all and len(items) > 2 * top:
        print(f"{'...':<16}")
        for char, count in items[-top:]:
            print(f"{show_char(char):<16}{count:>14,}{count / total:>11.4%}")


def show_characters(summary: dict, top: int) -> None:
    """How many samples each character (speaker) has."""
    info = summary["characters"]
    counts = np.array(list(info["counts"].values()))

    # How few characters make up half of the data?
    biggest_first = np.sort(counts)[::-1]
    running_total = np.cumsum(biggest_first)
    n_for_half = int(np.searchsorted(running_total, counts.sum() / 2) + 1)

    title("3. CHARACTERS (character_id)")
    print(f"distinct characters : {info['n_unique']:,}")
    print(f"samples per character: min={info['min_samples']:,}  "
          f"median={info['median_samples']:,.0f}  "
          f"mean={info['mean_samples']:,.1f}  max={info['max_samples']:,}")
    print(f"long tail           : {n_for_half} characters "
          f"({n_for_half / info['n_unique']:.1%}) hold half of all rows")

    print(f"\ntop {top} characters:")
    print("-" * 70)
    for name, count in list(info["counts"].items())[:top]:
        print(f"  {name:<52}{count:>10,}")


def show_lengths_and_missing(summary: dict) -> None:
    """Field lengths and missing values."""
    title("4. FIELD LENGTHS")
    print("x length (characters):")
    for length, count in sorted(summary["x_length_counts"].items()):
        print(f"  {length:>4} -> {count:>12,} rows")
    print("y length (characters):")
    for length, count in sorted(summary["y_length_counts"].items()):
        print(f"  {length:>4} -> {count:>12,} rows")

    title("5. MISSING AND EMPTY VALUES")
    print(f"{'column':<20}{'missing':>12}{'empty strings':>16}")
    print("-" * 48)
    for name in summary["schema"]:
        print(f"{name:<20}{summary['nulls'][name]:>12,}"
              f"{summary['empty_strings'][name]:>16,}")


def main(argv: list[str] | None = None) -> int:
    """Print the summary."""
    parser = argparse.ArgumentParser(
        prog="python dataset_info.py",
        description="Print a summary of the downloaded Shakespeare dataset.",
    )
    parser.add_argument("--top", type=int, default=10, help="rows per table")
    parser.add_argument("--all-classes", action="store_true", help="list all classes")
    args = parser.parse_args(argv)

    if not RAW_PARQUET.exists():
        print(f"error: {RAW_PARQUET} not found. Run `python -m src.download` first.",
              file=sys.stderr)
        return 1

    summary = summarize(load_raw())

    show_provenance()
    show_overview(summary)
    show_target(summary, args.top, args.all_classes)
    show_characters(summary, args.top)
    show_lengths_and_missing(summary)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
