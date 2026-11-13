"""Small offline tests. Run from Workshop_4: python -m unittest discover -s tests -v."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from src.data import DataPipeline, validate_splits, validate_vocab


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.vocab = {"itos": ["<UNK>", "a", "b"], "stoi": {"<UNK>": 0, "a": 1, "b": 2}, "size": 3, "unk_index": 0}
        self.splits = {"character_ids": {"train": ["A"], "val": ["B"], "test": ["C"]}}
        for name, value in (("vocab", self.vocab), ("splits", self.splits)):
            (root / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
        self.pipeline = DataPipeline({"vocab_path": root / "vocab.json", "splits_path": root / "splits.json",
                                      "processed_dir": root, "raw_path": root / "raw.parquet",
                                      "window_size": 2, "batch_size": 2, "dataset": "synthetic", "revision": "test"})

    def test_encoding_unknown_and_case(self):
        self.assertEqual(self.pipeline.encode_rows([{"character_id": "A", "x": "aZ", "y": "b", "y_id": 2}], "train"),
                         {"x": [[1, 0]], "y": [2]})

    def test_invalid_vocab(self):
        for key, value in (("size", 4), ("unk_index", 1), ("stoi", {"a": 0})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_vocab(dict(self.vocab, **{key: value}))

    def test_partition_overlap_and_duplicates(self):
        for ids in (["A"], ["B", "B"], []):
            bad = copy.deepcopy(self.splits)
            bad["character_ids"]["val"] = ids
            with self.assertRaises(ValueError):
                validate_splits(bad)

    def test_bad_rows(self):
        row = {"character_id": "A", "x": "ab", "y": "b"}
        for change in ({"character_id": "B"}, {"x": None}, {"x": "a"}, {"y": "bb"}, {"y_id": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.pipeline.encode_rows([dict(row, **change)], "train")

    def test_missing_files_and_bad_arguments(self):
        self.assertTrue(all(not entry["exists"] for entry in self.pipeline.inventory().values()))
        with self.assertRaises(FileNotFoundError):
            list(self.pipeline.iter_batches("train"))
        for split, size in (("other", 2), ("train", 0), ("train", True)):
            with self.assertRaises(ValueError):
                list(self.pipeline.iter_batches(split, size))

    def test_checkpoint_contract(self):
        checkpoint = {"state_dict": {}, "model_hparams": {"vocab_size": 3}, "vocab": self.vocab,
                      "data_contract": self.pipeline.contract()}
        self.assertTrue(self.pipeline.validate_checkpoint(checkpoint))
        for key, value in (("data_contract", {}), ("state_dict", None), ("model_hparams", {"vocab_size": 4})):
            with self.assertRaises(ValueError):
                self.pipeline.validate_checkpoint(dict(checkpoint, **{key: value}))
        reordered = copy.deepcopy(checkpoint)
        reordered["vocab"] = dict(self.vocab, itos=["<UNK>", "b", "a"], stoi={"<UNK>": 0, "b": 1, "a": 2})
        with self.assertRaises(ValueError):
            self.pipeline.validate_checkpoint(reordered)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow not installed")
    def test_parquet_batches(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows = [{"character_id": "A", "x": x, "y": "b", "y_id": 2} for x in ("aa", "ab", "ba", "bb", "az")]
        path = self.pipeline.config["processed_dir"] / "train.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
        batches = list(self.pipeline.iter_batches("train"))
        self.assertEqual([len(b["x"]) for b in batches], [2, 2, 1])
        self.assertEqual(batches[-1]["x"], [[1, 0]])
        self.assertEqual(batches, list(self.pipeline.iter_batches("train")))
        self.pipeline.splits["counts"] = {"train": {"n_rows": 6}}
        with self.assertRaises(ValueError):
            list(self.pipeline.iter_batches("train"))
        del self.pipeline.splits["counts"]
        pq.write_table(pa.table({"x": ["ab"]}), path)
        with self.assertRaises(ValueError):
            list(self.pipeline.iter_batches("train"))


if __name__ == "__main__":
    unittest.main()
