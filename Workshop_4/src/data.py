"""Bounded-memory processed Parquet reader; no downloads or implicit writes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path=ROOT / "config.json"):
    path = Path(path).resolve()
    config = read_json(path)
    for key in ("window_size", "batch_size"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("vocab_path", "splits_path", "processed_dir", "raw_path"):
        config[key] = (path.parent / config[key]).resolve()
    return config


def validate_vocab(vocab):
    tokens, mapping = vocab["itos"], vocab["stoi"]
    if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) for t in tokens):
        raise ValueError("itos must be a nonempty list of strings")
    if len(set(tokens)) != len(tokens):
        raise ValueError("duplicate vocabulary tokens")
    if mapping != {token: i for i, token in enumerate(tokens)} or any(type(i) is not int for i in mapping.values()):
        raise ValueError("stoi and itos must be inverse contiguous mappings")
    if vocab.get("size", len(tokens)) != len(tokens):
        raise ValueError("vocabulary size mismatch")
    if vocab.get("unk_index") != 0 or tokens[0] != "<UNK>":
        raise ValueError("expected <UNK> at index 0")
    if vocab.get("unk_token", "<UNK>") != "<UNK>" or any(len(t) != 1 for t in tokens[1:]):
        raise ValueError("expected character vocabulary")
    return vocab


def validate_splits(payload):
    groups = payload["character_ids"]
    if set(groups) != set(SPLITS):
        raise ValueError("expected train, val and test partitions")
    seen = set()
    for name in SPLITS:
        ids = groups[name]
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError(f"invalid character ids in {name}")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("duplicate or overlapping character ids")
        seen.update(ids)
        counts = payload.get("counts", {}).get(name, {})
        if counts.get("n_characters", len(ids)) != len(ids):
            raise ValueError(f"character count mismatch in {name}")
    return payload


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


class DataPipeline:
    def __init__(self, config=None):
        self.config = load_config() if config is None else config
        self.vocab = validate_vocab(read_json(self.config["vocab_path"]))
        self.splits = validate_splits(read_json(self.config["splits_path"]))
        self.groups = {name: set(ids) for name, ids in self.splits["character_ids"].items()}
        self._codepoint_lookup = None

    def inventory(self):
        paths = {"raw": Path(self.config["raw_path"])}
        paths.update({name: Path(self.config["processed_dir"]) / f"{name}.parquet" for name in SPLITS})
        return {key: {"path": str(path), "exists": path.is_file()} for key, path in paths.items()}

    def contract(self):
        """Persist this metadata in future AE/VAE checkpoints."""
        return {"version": 1, "window_size": self.config["window_size"],
                "vocab_sha256": fingerprint(self.vocab["itos"]),
                "splits_sha256": fingerprint(self.splits["character_ids"]),
                "dataset": self.config["dataset"], "revision": self.config["revision"]}

    def validate_checkpoint(self, checkpoint):
        """Validate an already loaded mapping, never deserialize executable files.

        W2-compatible envelope; tensor/model compatibility is left to load_state_dict.
        Legacy checkpoints without a data contract cannot prove split compatibility.
        """
        for key in ("state_dict", "model_hparams", "vocab"):
            if not isinstance(checkpoint.get(key), dict):
                raise ValueError(f"checkpoint requires mapping {key}")
        vocab = validate_vocab(checkpoint["vocab"])
        if vocab["itos"] != self.vocab["itos"]:
            raise ValueError("checkpoint token order mismatch")
        if checkpoint["model_hparams"].get("vocab_size") != len(self.vocab["itos"]):
            raise ValueError("checkpoint vocab_size mismatch")
        if checkpoint.get("data_contract") != self.contract():
            raise ValueError("missing or incompatible data_contract (legacy checkpoints unsupported)")
        return True

    def _validate_rows(self, rows, split):
        """Shared row contract for list and tensor readers; no fitting or writes."""
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        texts, targets = [], []
        stoi = self.vocab["stoi"]
        for row in rows:
            x, y = row.get("x"), row.get("y")
            if row.get("character_id") not in self.groups[split]:
                raise ValueError(f"character outside {split} partition")
            if not isinstance(x, str) or len(x) != self.config["window_size"] or not isinstance(y, str) or len(y) != 1:
                raise ValueError("malformed processed row")
            y_id = stoi.get(y, 0)
            if "y_id" in row and (type(row["y_id"]) is not int or row["y_id"] != y_id):
                raise ValueError("y_id inconsistent with vocabulary")
            texts.append(x)
            targets.append(y_id)
        return texts, targets

    def encode_rows(self, rows, split):
        texts, targets = self._validate_rows(rows, split)
        stoi = self.vocab["stoi"]
        inputs = [[stoi.get(char, 0) for char in text] for text in texts]
        # Reconstruction uses x as its target; y is retained for later evaluation.
        return {"x": inputs, "y": targets}

    def encode_tensor_rows(self, rows, split):
        """Equivalent CPU long tensors without one Python lookup per character.

        UTF-32 represents one code point per character, including non-BMP text.
        A bounded dense lookup is built from the existing vocabulary; out-of-range
        code points map to UNK (0). No PAD category or mask is introduced.
        NumPy/PyTorch are imported only for the optional tensor reader.
        """
        import numpy as np
        import torch

        texts, targets = self._validate_rows(rows, split)
        if self._codepoint_lookup is None:
            pairs = [(ord(token), index) for index, token in enumerate(self.vocab["itos"]) if index]
            largest = max((codepoint for codepoint, _ in pairs), default=0)
            # The Unicode space is bounded; Shakespeare uses only a small table.
            self._codepoint_lookup = np.zeros(max(256, largest + 1), dtype=np.int64)
            for codepoint, index in pairs:
                self._codepoint_lookup[codepoint] = index
        codepoints = np.frombuffer("".join(texts).encode("utf-32-le", errors="surrogatepass"), dtype="<u4")
        encoded = np.zeros(codepoints.shape, dtype=np.int64)
        in_range = codepoints < len(self._codepoint_lookup)
        encoded[in_range] = self._codepoint_lookup[codepoints[in_range]]
        x = torch.from_numpy(encoded.reshape(len(texts), self.config["window_size"]))
        return {"x": x, "y": torch.tensor(targets, dtype=torch.long)}

    def _iter_rows_batches(self, split, batch_size=None):
        """One checked Parquet path shared by both encoding implementations."""
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        size = self.config["batch_size"] if batch_size is None else batch_size
        if type(size) is not int or size <= 0:
            raise ValueError("batch_size must be a positive integer")
        path = Path(self.config["processed_dir"]) / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing processed dataset: {path}; no automatic download or rebuild")
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        try:
            required = ["character_id", "x", "y"]
            if not set(required).issubset(parquet.schema_arrow.names):
                raise ValueError("missing required Parquet columns")
            columns = required + (["y_id"] if "y_id" in parquet.schema_arrow.names else [])
            expected = self.splits.get("counts", {}).get(split, {}).get("n_rows")
            if expected is not None and parquet.metadata.num_rows != expected:
                raise ValueError(f"row count mismatch for {split}")
            for batch in parquet.iter_batches(batch_size=size, columns=columns, use_threads=False):
                yield batch.to_pylist()
        finally:
            parquet.close()

    def iter_batches(self, split, batch_size=None):
        reader = self._iter_rows_batches(split, batch_size)
        try:
            for rows in reader:
                yield self.encode_rows(rows, split)
        finally:
            reader.close()

    def iter_tensor_batches(self, split, batch_size=None):
        """Optional vectorized encoding; identical rows and order to iter_batches."""
        reader = self._iter_rows_batches(split, batch_size)
        try:
            for rows in reader:
                yield self.encode_tensor_rows(rows, split)
        finally:
            reader.close()


if __name__ == "__main__":
    pipeline = DataPipeline()
    print(json.dumps({"data": pipeline.inventory(), "contract": pipeline.contract()}, indent=2))
