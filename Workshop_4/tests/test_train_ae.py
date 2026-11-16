"""Tiny CPU training/recovery checks; no real experiment or test-set selection."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

HAS_DEPENDENCIES = all(importlib.util.find_spec(name) is not None
                       for name in ("torch", "pyarrow"))
if HAS_DEPENDENCIES:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from src.data import DataPipeline
    from src.train_ae import run_experiment


@unittest.skipUnless(HAS_DEPENDENCIES, "training integration requires torch and pyarrow")
class TrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="workshop4-training-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        vocab = {"itos": ["<UNK>", "a", "b"],
                 "stoi": {"<UNK>": 0, "a": 1, "b": 2},
                 "size": 3, "unk_index": 0}
        self.splits = {"train": ("A", 12), "val": ("B", 8), "test": ("C", 4)}
        split_payload = {
            "character_ids": {name: [speaker] for name, (speaker, _) in self.splits.items()},
            "counts": {name: {"n_characters": 1, "n_rows": count}
                       for name, (_, count) in self.splits.items()},
        }
        vocab_path, splits_path = self.root / "vocab.json", self.root / "splits.json"
        vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
        splits_path.write_text(json.dumps(split_payload), encoding="utf-8")
        all_rows = []
        for name, (speaker, count) in self.splits.items():
            rows = [{"character_id": speaker, "x": "aZbab" if i % 2 else "ababa",
                     "y": "a" if i % 2 == 0 else "b", "y_id": 1 if i % 2 == 0 else 2}
                    for i in range(count)]
            pq.write_table(pa.Table.from_pylist(rows), self.root / f"{name}.parquet")
            all_rows.extend(rows)
        pq.write_table(pa.Table.from_pylist(all_rows), self.root / "raw.parquet")
        self.pipeline = DataPipeline({
            "vocab_path": vocab_path, "splits_path": splits_path,
            "processed_dir": self.root, "raw_path": self.root / "raw.parquet",
            "window_size": 5, "batch_size": 2, "dataset": "synthetic", "revision": "test",
        })
        # Guard the scientific boundary: training code must never iterate test.
        self.seen_splits = []
        original = self.pipeline.iter_batches

        def guarded(split, batch_size=None):
            self.assertIn(split, ("train", "val"))
            self.seen_splits.append(split)
            yield from original(split, batch_size=batch_size)

        self.pipeline.iter_batches = guarded
        self.config = {
            "name": "tiny", "model": "vae", "epochs": 2, "batch_size": 2,
            "learning_rate": 0.001, "seed": 42, "beta": 0.5,
            "latent_dim": 3, "embedding_dim": 3, "hidden_size": 4,
            "num_layers": 1, "dropout": 0.0, "normal_class": None,
            "shuffle_buffer_batches": 2, "grad_clip": 5.0,
        }

    def run_tiny(self, output, config=None, **kwargs):
        return run_experiment(
            self.config if config is None else config, self.pipeline,
            self.root / output, device="cpu", smoke_test=True,
            max_train_batches=2, max_val_batches=2, **kwargs,
        )

    def checkpoint(self, output, kind="last"):
        path = self.root / output / self.config["name"] / "checkpoints" / f"{kind}.pt"
        return torch.load(path, map_location="cpu", weights_only=True)

    def assert_nested_equal(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(actual, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_tiny_ae_and_vae_save_metrics_and_checkpoints(self):
        for model in ("ae", "vae"):
            with self.subTest(model=model):
                config = {**self.config, "model": model}
                self.run_tiny(model, config=config)
                last, best = self.checkpoint(model), self.checkpoint(model, "best")
                self.assertEqual(last["epoch"], 2)
                self.assertEqual(len(last["history"]), 2)
                self.assertEqual(best["epoch"], last["best_epoch"])
                self.pipeline.validate_checkpoint(last)
                self.assertEqual(last["model_hparams"]["vocab_size"], 3)
                self.assertEqual(last["model_hparams"]["sequence_length"], 5)
                for row in last["history"]:
                    for metric in ("train_reconstruction", "train_kl", "train_total",
                                   "val_reconstruction", "val_kl", "val_total"):
                        self.assertTrue(math.isfinite(row[metric]), metric)
                    if model == "ae":
                        self.assertEqual(row["train_kl"], 0.0)
                        self.assertEqual(row["val_kl"], 0.0)
                csv_path = self.root / model / "tiny" / "history.csv"
                with csv_path.open(newline="", encoding="utf-8") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 2)
        self.assertEqual(set(self.seen_splits), {"train", "val"})

    def test_vae_resume_matches_uninterrupted_training(self):
        self.run_tiny("full")
        self.run_tiny("resumed", stop_after_epoch=1)
        interrupted = self.checkpoint("resumed")
        self.assertEqual(interrupted["epoch"], 1)
        self.run_tiny("resumed", resume="auto")
        full, resumed = self.checkpoint("full"), self.checkpoint("resumed")
        self.assertEqual(full["epoch"], resumed["epoch"])
        self.assertEqual(full["best_epoch"], resumed["best_epoch"])
        self.assert_nested_equal(full["state_dict"], resumed["state_dict"])
        self.assert_nested_equal(full["optimizer_state_dict"], resumed["optimizer_state_dict"])
        for left, right in zip(full["history"], resumed["history"]):
            for key in left.keys() - {"seconds"}:
                self.assertEqual(left[key], right[key], key)

    def test_resume_rejects_hyperparameter_or_data_contract_changes(self):
        self.run_tiny("mismatch", stop_after_epoch=1)
        with self.assertRaises(ValueError):
            self.run_tiny("mismatch", config={**self.config, "learning_rate": 0.01}, resume="auto")
        self.pipeline.config["revision"] = "different-data"
        with self.assertRaises(ValueError):
            self.run_tiny("mismatch", resume="auto")

    def test_resume_repairs_history_and_best_from_committed_last_checkpoint(self):
        self.run_tiny("repair", stop_after_epoch=1)
        run_dir = self.root / "repair" / "tiny"
        (run_dir / "history.csv").write_text("interrupted-write\n", encoding="utf-8")
        (run_dir / "checkpoints" / "best.pt").unlink()
        self.run_tiny("repair", resume="auto")
        last, best = self.checkpoint("repair"), self.checkpoint("repair", "best")
        self.assertEqual(last["epoch"], 2)
        self.assertEqual(best["epoch"], last["best_epoch"])
        with (run_dir / "history.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["epoch"]) for row in rows], [1, 2])

    def test_normal_only_filter_uses_existing_y_without_test_access(self):
        config = {**self.config, "normal_class": 1, "epochs": 1, "batch_size": 32}
        self.run_tiny("normal", config=config)
        checkpoint = self.checkpoint("normal")
        row = checkpoint["history"][0]
        self.assertEqual(row["train_examples"], 6)
        self.assertEqual(row["val_examples"], 4)
        self.assertNotIn("test", self.seen_splits)

    def test_short_training_requires_explicit_smoke_mode(self):
        with self.assertRaises(ValueError):
            run_experiment(self.config, self.pipeline, self.root / "forbidden", device="cpu")


if __name__ == "__main__":
    unittest.main()
