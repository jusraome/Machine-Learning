"""Steps 2-3 - Load, validate, clean and split the dataset.

This module turns the raw download into train/val/test files. It is used by
`dataset_info.py` and by `notebooks/eda.ipynb`, so the EDA and the pipeline
always report the same numbers.

Run the whole pipeline with:

    python -m src.dataset --prepare
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    PROCESSED_DIR,
    RAW_PARQUET,
    SEED,
    SPLIT_FRACTIONS,
    SPLIT_NAMES,
    SPLITS_FILE,
    TARGET_SIZE,
    UNK_INDEX,
    UNK_TOKEN,
    VOCAB_FILE,
    WINDOW_SIZE,
    ensure_dirs,
    set_seeds,
)

set_seeds(SEED)

SplitName = Literal["train", "val", "test"]


# ===========================================================================
# Loading
# ===========================================================================
def load_raw(path: Path = RAW_PARQUET) -> pd.DataFrame:
    """Load the raw dataset saved by `src.download`.

    Returns a DataFrame with columns `character_id`, `x` and `y`.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.download` first.")
    return pd.read_parquet(path)


def character_counts(series: pd.Series, chunk_size: int = 200_000) -> Counter:
    """Count how often each character appears in a column of strings.

    Counting in chunks keeps memory low: joining all 4.2 M windows into one
    string at once would need ~340 MB.
    """
    counts: Counter = Counter()
    for start in range(0, len(series), chunk_size):
        counts.update("".join(series.iloc[start : start + chunk_size]))
    return counts


# ===========================================================================
# Summary (used by dataset_info.py)
# ===========================================================================
def summarize(df: pd.DataFrame) -> dict:
    """Compute the summary that `dataset_info.py` prints.

    All figures cover the full dataset; nothing is sampled.
    """
    class_counts = df["y"].value_counts()
    per_character = df["character_id"].value_counts()

    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "schema": {name: str(dtype) for name, dtype in df.dtypes.items()},
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "target": {
            "n_classes": len(class_counts),
            "class_counts": class_counts.to_dict(),
        },
        "characters": {
            "n_unique": len(per_character),
            "min_samples": int(per_character.min()),
            "mean_samples": float(per_character.mean()),
            "median_samples": float(per_character.median()),
            "max_samples": int(per_character.max()),
            "counts": per_character.to_dict(),
        },
        "x_length_counts": df["x"].str.len().value_counts().to_dict(),
        "y_length_counts": df["y"].str.len().value_counts().to_dict(),
        "nulls": df.isna().sum().to_dict(),
        "empty_strings": {name: int((df[name] == "").sum()) for name in df.columns},
    }


# ===========================================================================
# Validation
# ===========================================================================
def validate(df: pd.DataFrame) -> dict:
    """Run the data-quality checks from the EDA and return a report.

    Covers missing values, malformed lengths, unexpected characters, duplicates
    and contexts whose targets disagree.
    """
    wrong_x_length = df["x"].str.len() != WINDOW_SIZE
    wrong_y_length = df["y"].str.len() != TARGET_SIZE

    # Windows that appear more than once with different targets: no model that
    # only reads x can get all of them right.
    targets_per_window = df.groupby("x")["y"].nunique()
    conflicting_windows = targets_per_window[targets_per_window > 1]

    # (x, y) pairs spoken by more than one character.
    speakers_per_pair = df.groupby(["x", "y"])["character_id"].nunique()

    is_duplicate = df.duplicated()

    return {
        "n_rows": len(df),
        "nulls": df.isna().sum().to_dict(),
        "empty_strings": {name: int((df[name] == "").sum()) for name in df.columns},
        "wrong_x_length": int(wrong_x_length.sum()),
        "wrong_y_length": int(wrong_y_length.sum()),
        "rows_with_non_ascii": int(df["x"].str.contains(r"[^\x00-\x7F]").sum()),
        "rows_with_control_chars": int(df["x"].str.contains(r"[\x00-\x1F\x7F]").sum()),
        "rows_with_repeated_spaces": int(df["x"].str.contains("  ", regex=False).sum()),
        "n_unique_x": int(df["x"].nunique()),
        "n_unique_xy": int(len(speakers_per_pair)),
        "exact_duplicate_rows": int(is_duplicate.sum()),
        "pairs_under_several_characters": int((speakers_per_pair > 1).sum()),
        "conflicting_windows": int(len(conflicting_windows)),
        "conflicting_rows": int(df["x"].isin(conflicting_windows.index).sum()),
        "malformed_rows": int((wrong_x_length | wrong_y_length).sum()),
    }


def usable_rows(report: dict) -> int:
    """Number of rows that `clean` would keep."""
    return report["n_rows"] - report["malformed_rows"] - report["exact_duplicate_rows"]


# ===========================================================================
# Cleaning
# ===========================================================================
def clean(
    df: pd.DataFrame,
    drop_malformed: bool = True,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Apply the cleaning rules the EDA justified, and only those.

    Applied:

    - `drop_malformed`: rows where len(x) != 80 or len(y) != 1, which cannot be
      turned into a fixed-size model input.
    - `drop_duplicates`: repeated (character_id, x, y) rows, keeping the first.
      They add no information, and since the split groups by character_id every
      copy stays inside one split, so removing them cannot leak anything.

    Deliberately NOT applied:

    - Lowercasing or removing punctuation: upper case, lower case, punctuation
      and the space are all target classes, so this would destroy the labels.
    - Collapsing repeated spaces: the EDA found zero rows with repeated spaces;
      LEAF already did this.
    - Unicode NFC normalisation: the EDA found zero non-ASCII characters, so it
      would change nothing here.

    Returns the cleaned DataFrame and a report of how many rows each rule removed.
    """
    n_before = len(df)
    rules = []

    if drop_malformed:
        keep = (df["x"].str.len() == WINDOW_SIZE) & (df["y"].str.len() == TARGET_SIZE)
        removed = int((~keep).sum())
        df = df[keep]
        rules.append(
            {
                "rule": "drop_malformed",
                "description": "len(x) != 80 or len(y) != 1",
                "reason": "cannot be encoded as a fixed-size window and target",
                "rows_removed": removed,
            }
        )

    if drop_duplicates:
        n = len(df)
        df = df.drop_duplicates()
        rules.append(
            {
                "rule": "drop_duplicates",
                "description": "repeated (character_id, x, y) rows, first one kept",
                "reason": "no extra information; stays inside one split anyway",
                "rows_removed": n - len(df),
            }
        )

    for rule, reason in [
        ("unicode_nfc_normalisation", "no non-ASCII characters in the data"),
        ("collapse_repeated_spaces", "no repeated spaces in the data"),
        ("lowercase_or_strip_punctuation", "case and punctuation are target classes"),
    ]:
        rules.append(
            {
                "rule": rule,
                "description": "not applied",
                "reason": reason,
                "rows_removed": 0,
            }
        )

    report = {
        "n_rows_before": n_before,
        "n_rows_after": len(df),
        "n_rows_removed": n_before - len(df),
        "rules": rules,
    }
    return df.reset_index(drop=True), report


# ===========================================================================
# Split: grouped by character_id
# ===========================================================================
def make_splits(df: pd.DataFrame, seed: int = SEED) -> dict[str, list[str]]:
    """Split the characters 70/15/15 with two GroupShuffleSplit stages.

    The split is by `character_id`, never by row. Because consecutive rows share
    79 of their 80 characters, a random split by row would put nearly identical
    windows in both train and validation.

    GroupShuffleSplit keeps whole groups together and picks them by group count,
    so it is applied to the list of unique character ids. Characters differ a lot
    in size, so the resulting *row* shares will not be exactly 70/15/15; the real
    shares are recorded in data/splits.json.
    """
    characters = np.array(sorted(df["character_id"].unique()))

    holdout_size = SPLIT_FRACTIONS["val"] + SPLIT_FRACTIONS["test"]
    first = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=seed)
    train_idx, holdout_idx = next(first.split(characters, groups=characters))

    holdout = characters[holdout_idx]
    # Cut the hold-out in half so val and test get 15% of characters each.
    second = GroupShuffleSplit(
        n_splits=1, test_size=SPLIT_FRACTIONS["test"] / holdout_size, random_state=seed
    )
    val_idx, test_idx = next(second.split(holdout, groups=holdout))

    return {
        "train": sorted(characters[train_idx]),
        "val": sorted(holdout[val_idx]),
        "test": sorted(holdout[test_idx]),
    }


def check_splits(splits: dict[str, list[str]]) -> dict:
    """Check that no character appears in more than one split."""
    sets = {name: set(ids) for name, ids in splits.items()}
    overlaps = {
        f"{a}&{b}": len(sets[a] & sets[b])
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    }
    if any(overlaps.values()):
        raise AssertionError(f"characters shared between splits: {overlaps}")
    return {"pairwise_overlaps": overlaps, "disjoint": True}


def save_splits_file(
    splits: dict[str, list[str]], row_counts: dict[str, int], cleaning: dict
) -> dict:
    """Save the split definition to data/splits.json.

    Only the character ids are stored, not the 4.2 M row indices: the ids fully
    determine the split and keep the file small.
    """
    n_characters = sum(len(ids) for ids in splits.values())
    n_rows = sum(row_counts.values())

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "strategy": "GroupShuffleSplit by character_id, two stages",
        "requested_fractions": SPLIT_FRACTIONS,
        "cleaning": cleaning,
        "n_rows_total": n_rows,
        "n_characters_total": n_characters,
        "counts": {
            name: {
                "n_characters": len(splits[name]),
                "n_rows": row_counts[name],
                "character_share": len(splits[name]) / n_characters,
                "row_share": row_counts[name] / n_rows,
            }
            for name in SPLIT_NAMES
        },
        "character_ids": {name: splits[name] for name in SPLIT_NAMES},
    }
    SPLITS_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def read_splits_file() -> dict:
    """Load data/splits.json."""
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(
            f"{SPLITS_FILE} not found. Run `python -m src.dataset --prepare` first."
        )
    return json.loads(SPLITS_FILE.read_text(encoding="utf-8"))


def load_split(
    name: SplitName, source: Literal["processed", "raw"] = "processed"
) -> pd.DataFrame:
    """Load one split.

    `source="processed"` reads data/processed/<name>.parquet, which also has the
    `y_id` column. `source="raw"` rebuilds the same rows from the raw data by
    re-applying the cleaning rules recorded in data/splits.json and then keeping
    that split's characters. Cleaning is not optional on this path: the raw file
    still contains the duplicate rows, so filtering it directly would give more
    rows than the split actually has.
    """
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")

    if source == "processed":
        path = PROCESSED_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.dataset --prepare` first."
            )
        return pd.read_parquet(path)

    splits = read_splits_file()
    cleaning = splits.get("cleaning", {})
    df, _ = clean(
        load_raw(),
        drop_malformed=cleaning.get("drop_malformed", True),
        drop_duplicates=cleaning.get("drop_duplicates", True),
    )
    return df[df["character_id"].isin(splits["character_ids"][name])].reset_index(
        drop=True
    )


# ===========================================================================
# Vocabulary and target encoding
# ===========================================================================
def build_vocab(train_df: pd.DataFrame) -> dict:
    """Build the character vocabulary from the training split only.

    Fitting on train only is what keeps the evaluation honest: a vocabulary built
    on the whole dataset would carry information about val and test. Index 0 is
    reserved for <UNK>, used for characters that never appear in train.
    """
    counts_x = character_counts(train_df["x"])
    counts_y = character_counts(train_df["y"])

    characters = sorted(set(counts_x) | set(counts_y))
    itos = [UNK_TOKEN] + characters
    stoi = {token: index for index, token in enumerate(itos)}

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "fitted_on": "train split only",
        "unk_token": UNK_TOKEN,
        "unk_index": UNK_INDEX,
        "size": len(itos),
        "itos": itos,
        "stoi": stoi,
        "train_char_counts_x": dict(sorted(counts_x.items())),
        "train_char_counts_y": dict(sorted(counts_y.items())),
    }


def save_vocab(vocab: dict) -> None:
    """Save the vocabulary to data/vocab.json."""
    VOCAB_FILE.write_text(json.dumps(vocab, indent=2) + "\n", encoding="utf-8")


def read_vocab() -> dict:
    """Load data/vocab.json."""
    if not VOCAB_FILE.exists():
        raise FileNotFoundError(
            f"{VOCAB_FILE} not found. Run `python -m src.dataset --prepare` first."
        )
    return json.loads(VOCAB_FILE.read_text(encoding="utf-8"))


def encode_targets(y: pd.Series, vocab: dict) -> pd.Series:
    """Turn each target character into its vocabulary index (<UNK> if unseen)."""
    return y.map(vocab["stoi"]).fillna(UNK_INDEX).astype("int16")


def count_unknown(df: pd.DataFrame, vocab: dict) -> dict:
    """Count characters that are missing from the train vocabulary."""
    known = set(vocab["stoi"]) - {UNK_TOKEN}
    unknown_in_x = {
        char: count
        for char, count in character_counts(df["x"]).items()
        if char not in known
    }
    return {
        "unknown_chars_in_x": sorted(unknown_in_x),
        "unknown_char_occurrences_in_x": int(sum(unknown_in_x.values())),
        "rows_with_unknown_target": int((~df["y"].isin(known)).sum()),
    }


# ===========================================================================
# Full pipeline
# ===========================================================================
def prepare(drop_duplicates: bool = True) -> dict:
    """Run steps 2-3 and write every output file."""
    set_seeds(SEED)
    ensure_dirs()

    print("[load] reading data/raw/shakespeare.parquet")
    df = load_raw()
    print(f"[load] {len(df):,} rows")

    print("[validate] running data-quality checks")
    validation = validate(df)
    print(
        f"[validate] usable {usable_rows(validation):,}/{validation['n_rows']:,}; "
        f"duplicates {validation['exact_duplicate_rows']:,}; "
        f"conflicting windows {validation['conflicting_windows']:,}"
    )

    print("[clean] applying the rules the EDA justified")
    df, cleaning = clean(df, drop_duplicates=drop_duplicates)
    print(f"[clean] {cleaning['n_rows_before']:,} -> {cleaning['n_rows_after']:,} rows")

    print("[split] GroupShuffleSplit by character_id (70/15/15 of characters)")
    splits = make_splits(df)
    split_check = check_splits(splits)

    parts = {}
    for name in SPLIT_NAMES:
        parts[name] = df[df["character_id"].isin(splits[name])].reset_index(drop=True)
        print(f"[split]   {name:5s} {len(splits[name]):4d} characters, "
              f"{len(parts[name]):,} rows")

    row_counts = {name: len(parts[name]) for name in SPLIT_NAMES}
    if sum(row_counts.values()) != len(df):
        raise AssertionError("split row counts do not add up to the cleaned data")

    splits_payload = save_splits_file(
        splits,
        row_counts,
        {"drop_malformed": True, "drop_duplicates": drop_duplicates},
    )
    print(f"[split] wrote {SPLITS_FILE.name}")

    print("[vocab] building the vocabulary from train only")
    vocab = build_vocab(parts["train"])
    save_vocab(vocab)
    print(f"[vocab] {vocab['size']} tokens (including {UNK_TOKEN}); "
          f"wrote {VOCAB_FILE.name}")

    unknown = {name: count_unknown(parts[name], vocab) for name in SPLIT_NAMES}
    for name in SPLIT_NAMES:
        print(f"[vocab]   {name:5s} unknown: {unknown[name]}")

    class_shares = {}
    for name in SPLIT_NAMES:
        part = parts[name].copy()
        part["y_id"] = encode_targets(part["y"], vocab)
        path = PROCESSED_DIR / f"{name}.parquet"
        part.to_parquet(path, index=False)
        print(f"[write] {path.name} ({path.stat().st_size:,} bytes)")
        class_shares[name] = (part["y"].value_counts(normalize=True)).to_dict()

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "validation": validation,
        "cleaning": cleaning,
        "split_check": split_check,
        "split_counts": splits_payload["counts"],
        "class_shares_by_split": class_shares,
        "largest_class_share_gap": largest_class_gap(class_shares),
        "unknown_characters": unknown,
        "vocab_size": vocab["size"],
    }
    report_path = PROCESSED_DIR / "prepare_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[report] wrote {report_path.name}")

    return report


def largest_class_gap(class_shares: dict[str, dict[str, float]]) -> dict:
    """Biggest difference in class frequency between any two splits.

    A small value means grouping by character did not distort the target
    distribution, which is the main risk of splitting by group.
    """
    all_classes = set().union(*(shares.keys() for shares in class_shares.values()))
    worst_class, worst_gap = "", 0.0
    for char in all_classes:
        values = [class_shares[name].get(char, 0.0) for name in SPLIT_NAMES]
        gap = max(values) - min(values)
        if gap > worst_gap:
            worst_class, worst_gap = char, gap
    return {"class": worst_class, "max_share_difference": worst_gap}


def main(argv: list[str] | None = None) -> int:
    """Run steps 2-3."""
    parser = argparse.ArgumentParser(
        prog="python -m src.dataset",
        description="Validate, clean and split the Shakespeare dataset.",
    )
    parser.add_argument("--prepare", action="store_true", help="run the whole pipeline")
    parser.add_argument(
        "--keep-duplicates", action="store_true", help="skip duplicate removal"
    )
    args = parser.parse_args(argv)

    if not args.prepare:
        parser.print_help()
        return 1

    try:
        prepare(drop_duplicates=not args.keep_duplicates)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print("\n[done] steps 2-3 complete. Files in data/processed/ and data/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
