"""Small equivalence checks for the optional tensor reader; existing splits stay immutable."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from src.data import DataPipeline


HAS_DEPENDENCIES = all(importlib.util.find_spec(name) is not None
                       for name in ("torch", "numpy", "pyarrow"))
if HAS_DEPENDENCIES:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch


@unittest.skipUnless(HAS_DEPENDENCIES, "tensor reader checks require numpy, torch and pyarrow")
class OptimizedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="workshop4-data-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        tokens = ["<UNK>", "a", "A", "é", "😀", "\ud800"]
        vocab = {"itos": tokens, "stoi": {token: index for index, token in enumerate(tokens)},
                 "size": len(tokens), "unk_index": 0}
        splits = {"character_ids": {"train": ["A"], "val": ["B"], "test": ["C"]}}
        for name, payload in (("vocab", vocab), ("splits", splits)):
            (self.root / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
        self.pipeline = DataPipeline({"vocab_path": self.root / "vocab.json", "splits_path": self.root / "splits.json",
            "processed_dir": self.root, "raw_path": self.root / "raw.parquet", "window_size": 4,
            "batch_size": 2, "dataset": "synthetic", "revision": "test"})

    def assert_equivalent(self, rows, split="train"):
        expected = self.pipeline.encode_rows(rows, split)
        actual = self.pipeline.encode_tensor_rows(rows, split)
        self.assertEqual(actual["x"].tolist(), expected["x"])
        self.assertEqual(actual["y"].tolist(), expected["y"])
        self.assertEqual(tuple(actual["x"].shape), (len(rows), 4))
        self.assertEqual(tuple(actual["y"].shape), (len(rows),))
        self.assertEqual(actual["x"].dtype, torch.long)
        self.assertEqual(actual["y"].dtype, torch.long)
        self.assertEqual(actual["x"].device.type, "cpu")
        self.assertTrue(actual["x"].is_contiguous())

    def test_unicode_unknown_case_and_empty_batches_are_exact(self):
        rows = [{"character_id": "A", "x": text, "y": target,
                 "y_id": self.pipeline.vocab["stoi"].get(target, 0)}
                for text, target in (("aAé😀", "é"), ("\ud800Z\x00\U0010ffff", "Z"), ("aaaa", "a"))]
        self.assert_equivalent(rows)
        self.assert_equivalent([])
        # A second call exercises the cached lookup and retains exact UNK=0.
        self.assert_equivalent(rows[:1])

    def test_both_encoders_reject_the_same_invalid_rows_and_splits(self):
        row = {"character_id": "A", "x": "aAé😀", "y": "a", "y_id": 1}
        for change in ({"character_id": "B"}, {"x": None}, {"x": "aaa"},
                       {"y": "aa"}, {"y_id": 2}, {"y_id": True}):
            for method in (self.pipeline.encode_rows, self.pipeline.encode_tensor_rows):
                with self.subTest(change=change, method=method.__name__), self.assertRaises(ValueError):
                    method([dict(row, **change)], "train")
        for method in (self.pipeline.encode_rows, self.pipeline.encode_tensor_rows):
            with self.assertRaises(ValueError):
                method([], "invalid")

    def test_parquet_readers_have_identical_batches_and_argument_checks(self):
        rows = [{"character_id": "A", "x": text, "y": "a", "y_id": 1}
                for text in ("aAé😀", "aZaZ", "😀😀😀😀", "aaaa", "AAAA")]
        pq.write_table(pa.Table.from_pylist(rows), self.root / "train.parquet", row_group_size=3)
        old = list(self.pipeline.iter_batches("train", batch_size=2))
        new = list(self.pipeline.iter_tensor_batches("train", batch_size=2))
        self.assertEqual([len(batch["x"]) for batch in new], [2, 2, 1])
        for expected, actual in zip(old, new):
            self.assertEqual(actual["x"].tolist(), expected["x"])
            self.assertEqual(actual["y"].tolist(), expected["y"])
        for method in (self.pipeline.iter_batches, self.pipeline.iter_tensor_batches):
            for split, size in (("bad", 2), ("train", 0), ("train", True)):
                with self.subTest(method=method.__name__, split=split, size=size), self.assertRaises(ValueError):
                    list(method(split, size))
            with self.assertRaises(FileNotFoundError):
                list(method("val"))

    def test_tensor_reader_keeps_partition_schema_and_row_count_validation(self):
        path = self.root / "train.parquet"
        pq.write_table(pa.table({"x": ["aaaa"]}), path)
        with self.assertRaises(ValueError):
            list(self.pipeline.iter_tensor_batches("train"))
        rows = [{"character_id": "B", "x": "aaaa", "y": "a", "y_id": 1}]
        pq.write_table(pa.Table.from_pylist(rows), path)
        with self.assertRaises(ValueError):
            list(self.pipeline.iter_tensor_batches("train"))
        self.pipeline.splits["counts"] = {"train": {"n_rows": 2}}
        with self.assertRaises(ValueError):
            list(self.pipeline.iter_tensor_batches("train"))

    def test_equal_token_tensors_produce_equal_outputs_and_gradients(self):
        from src.autoencoder import Autoencoder, reconstruction_loss

        rows = [{"character_id": "A", "x": "aAé😀", "y": "a"}]
        legacy = torch.tensor(self.pipeline.encode_rows(rows, "train")["x"], dtype=torch.long)
        optimized = self.pipeline.encode_tensor_rows(rows, "train")["x"]
        torch.manual_seed(42)
        first = Autoencoder(vocab_size=6, sequence_length=4, embedding_dim=3, hidden_size=4, latent_dim=2)
        second = Autoencoder(**first.hparams)
        second.load_state_dict(first.state_dict())
        left, right = first(legacy).logits, second(optimized).logits
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        reconstruction_loss(left, legacy).backward()
        reconstruction_loss(right, optimized).backward()
        for p1, p2 in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(p1.grad, p2.grad, rtol=0, atol=0)

    def test_two_real_batches_per_train_val_match_without_metadata_changes(self):
        pipeline = DataPipeline()
        if not all((Path(pipeline.config["processed_dir"]) / f"{name}.parquet").is_file()
                   for name in ("train", "val")):
            self.skipTest("local train/val Parquet unavailable")
        protected = [pipeline.config["vocab_path"], pipeline.config["splits_path"]]
        hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
        for split in ("train", "val"):
            old, new = pipeline.iter_batches(split, 4), pipeline.iter_tensor_batches(split, 4)
            try:
                for _ in range(2):
                    expected, actual = next(old), next(new)
                    self.assertEqual(actual["x"].tolist(), expected["x"])
                    self.assertEqual(actual["y"].tolist(), expected["y"])
            finally:
                old.close()
                new.close()
        self.assertEqual(hashes, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected})


if __name__ == "__main__":
    unittest.main()
