"""Shared settings: seed, paths and dataset coordinates.

Every script imports `SEED` from here and calls `set_seeds()` before doing
anything random, so the whole project is reproducible from one constant.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

# --- Reproducibility -------------------------------------------------------
SEED: int = 42


def set_seeds(seed: int = SEED) -> None:
    """Seed the random number generators used in this project.

    scikit-learn has no global seed, so its splitters are always created with
    `random_state=SEED` where they are used.
    """
    random.seed(seed)
    np.random.seed(seed)


# --- Dataset on the Hugging Face Hub ---------------------------------------
HF_DATASET: str = "flwrlabs/shakespeare"

# Pinned commit, so re-running the pipeline always downloads the same data.
HF_REVISION: str = "2cfe3a5ba0d9b34634ade086803a479b2c2b8e11"

# --- Folders and files -----------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

RAW_PARQUET: Path = RAW_DIR / "shakespeare.parquet"
CHECKSUMS_FILE: Path = RAW_DIR / "CHECKSUMS.txt"
DATASET_META_FILE: Path = RAW_DIR / "dataset_meta.json"

SPLITS_FILE: Path = DATA_DIR / "splits.json"
VOCAB_FILE: Path = DATA_DIR / "vocab.json"

FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"

# --- What the data should look like (checked, not assumed) -----------------
WINDOW_SIZE: int = 80  # number of characters in every x
TARGET_SIZE: int = 1   # number of characters in every y

# --- Split settings --------------------------------------------------------
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
SPLIT_FRACTIONS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

# Characters that appear in val/test but not in train are replaced by this.
UNK_TOKEN: str = "<UNK>"
UNK_INDEX: int = 0


def ensure_dirs() -> None:
    """Create the folders the pipeline writes into."""
    for folder in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
