"""Small controlled-trial regressions; no production checkpoint or GPU run.

All optimization uses a synthetic vocabulary and at most a few CPU batches.
The historical-model test verifies evaluation-only loading, not convergence.
"""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "pyarrow", "numpy"))
if AVAILABLE:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from src.autoencoder import Autoencoder
    from src.data import DataPipeline
    from src import stability_experiments as experiments


@unittest.skipUnless(AVAILABLE, "controlled stability trials require torch, pyarrow and numpy")
class StabilityExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="workshop4-controlled-trials-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        vocab = {"itos": ["<UNK>", "a", "b"], "stoi": {"<UNK>": 0, "a": 1, "b": 2},
                 "size": 3, "unk_index": 0}
        split = {"character_ids": {"train": ["A"], "val": ["B"], "test": ["C"]},
                 "counts": {"train": {"n_characters": 1, "n_rows": 8},
                            "val": {"n_characters": 1, "n_rows": 4},
                            "test": {"n_characters": 1, "n_rows": 2}}}
        for name, value in (("vocab", vocab), ("splits", split)):
            (self.data / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
        for name, speaker, count in (("train", "A", 8), ("val", "B", 4), ("test", "C", 2)):
            rows = [{"character_id": speaker, "x": "ababa" if i % 2 else "baZab",
                     "y": "a", "y_id": 1} for i in range(count)]
            pq.write_table(pa.Table.from_pylist(rows), self.data / f"{name}.parquet")
        self.pipeline = DataPipeline({"vocab_path": self.data / "vocab.json",
                                      "splits_path": self.data / "splits.json",
                                      "processed_dir": self.data, "raw_path": self.data / "raw.parquet",
                                      "window_size": 5, "batch_size": 2,
                                      "dataset": "synthetic", "revision": "test"})
        self.seen_splits = []
        for method_name in ("iter_batches", "iter_tensor_batches"):
            original = getattr(self.pipeline, method_name)
            def guarded(name, batch_size=None, original=original):
                self.assertIn(name, ("train", "val"), "Test must never be used for model selection")
                self.seen_splits.append(name)
                yield from original(name, batch_size=batch_size)
            setattr(self.pipeline, method_name, guarded)
        self.plan = {
            "version": 1,
            "defaults": {"seed": 42, "embedding_dim": 3, "hidden_size": 4,
                         "latent_dim": 2, "num_layers": 1, "dropout": 0.0,
                         "batch_size": 2, "learning_rate": 0.001,
                         "latent_activation": "identity", "tensor_loader": True,
                         "grad_clip": 5.0},
            "variants": [{"name": "reference", "changes": {}},
                         {"name": "batch4", "changes": {"batch_size": 4}},
                         {"name": "batch8", "changes": {"batch_size": 8}},
                         {"name": "lr_small", "changes": {"learning_rate": 0.0003}},
                         {"name": "latent_tanh", "changes": {"latent_activation": "tanh"}, "optional": True}],
            "budgets": {"updates": {"max_updates": 2, "max_examples": 8},
                        "examples": {"max_updates": 4, "max_examples": 8}},
            "limits": {"max_seconds_per_trial": 10.0, "max_seconds_total": 20.0,
                       "max_total_updates": 32, "max_total_examples": 64,
                       "validation_examples": 4, "validation_batch_size": 2,
                       "evaluate_every_updates": 1, "max_grad_norm": 1e8,
                       "max_latent_abs": 100.0, "max_reconstruction_ratio": 3.0,
                       "deterioration_patience": 2},
        }

    def snapshot_sources(self):
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.data.rglob("*") if path.is_file()}

    def run_small(self, name="trial", **kwargs):
        return experiments.run_suite(self.plan, self.pipeline, self.root / name,
                                     variants=["reference"], budget="updates",
                                     device="cpu", execute=True, **kwargs)

    def test_plan_requires_one_factor_controls_and_explicit_architecture_opt_in(self):
        plan_path = self.root / "plan.yaml"
        plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        normalized = experiments.load_plan(plan_path)
        trials = experiments.expand_trials(normalized, budget="both")
        self.assertEqual(len(trials), 8)
        self.assertEqual({trial["budget"] for trial in trials}, {"examples", "updates"})
        for trial in trials:
            actual = trial["config"]
            changed = {key for key, value in self.plan["defaults"].items() if actual[key] != value}
            self.assertEqual(len(changed), 0 if trial["variant"] == "reference" else 1)
        with self.assertRaises(ValueError):
            experiments.expand_trials(normalized, variant_names=["latent_tanh"], include_architecture=False)
        chosen = experiments.expand_trials(normalized, variant_names=["latent_tanh"],
                                           budget="updates", include_architecture=True)
        self.assertEqual(chosen[0]["config"]["latent_activation"], "tanh")
        bad = copy.deepcopy(self.plan)
        bad["variants"][1]["changes"]["learning_rate"] = 0.0003
        with self.assertRaises(ValueError):
            experiments.validate_plan(bad)
        bad = copy.deepcopy(self.plan)
        bad["budgets"]["updates"]["max_updates"] = 129
        with self.assertRaises(ValueError):
            experiments.validate_plan(bad)

    def test_dry_plan_neither_reads_data_nor_starts_optimizer_or_creates_output(self):
        output = self.root / "dry"
        with patch.object(self.pipeline, "iter_tensor_batches", side_effect=AssertionError("dry data read")), \
             patch.object(self.pipeline, "iter_batches", side_effect=AssertionError("dry data read")), \
             patch.object(torch.optim, "Adam", side_effect=AssertionError("dry optimizer")):
            result = experiments.run_suite(self.plan, self.pipeline, output, execute=False)
        self.assertFalse(output.exists())
        self.assertFalse(self.seen_splits)
        self.assertTrue(result)

    def test_matched_updates_and_examples_are_separate_and_sources_unchanged(self):
        protected = self.snapshot_sources()
        training_sequences = {}
        original_forward = Autoencoder.forward
        def record_training_order(model, x):
            if model.training and torch.is_grad_enabled():
                training_sequences.setdefault(model, []).extend(x.detach().cpu().tolist())
            return original_forward(model, x)
        with patch.object(Autoencoder, "forward", new=record_training_order):
            result = experiments.run_suite(self.plan, self.pipeline, self.root / "matched",
                                           variants=["reference", "batch4"], budget="both",
                                           device="cpu", execute=True)
        trials = result["trials"]
        self.assertEqual(len(trials), 4)
        update_trials = [trial for trial in trials if trial["budget"] == "updates"]
        example_trials = [trial for trial in trials if trial["budget"] == "examples"]
        self.assertEqual({trial["updates"] for trial in update_trials}, {2})
        self.assertEqual({trial["examples"] for trial in update_trials}, {4, 8})
        self.assertEqual({trial["examples"] for trial in example_trials}, {8})
        self.assertEqual({trial["updates"] for trial in example_trials}, {2, 4})
        self.assertTrue(all(row["matched_budget"] for row in result["comparisons"]))
        self.assertTrue(all(row["relative_val_loss_change"] == 0.0 for row in result["comparisons"]
                            if row["trial"].startswith("reference__")))
        self.assertEqual(len({trial["initial_state_sha256"] for trial in trials}), 1)
        self.assertEqual(len(training_sequences), 4)
        longest = max(training_sequences.values(), key=len)
        for sequence in training_sequences.values():
            self.assertEqual(sequence, longest[:len(sequence)], "Every trial must see the same ordered training prefix")
        self.assertEqual(protected, self.snapshot_sources())
        self.assertEqual(set(self.seen_splits), {"train", "val"})
        self.assertTrue((self.root / "matched/summary.json").is_file())
        with (self.root / "matched/metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows)
        self.assertTrue(all(float(row["grad_norm_post_max"]) <= 5.0 * (1 + 1e-6)
                            for row in rows if row.get("grad_norm_post_max")))
        with self.assertRaises(FileExistsError):
            experiments.run_suite(self.plan, self.pipeline, self.root / "matched", execute=True)

    def test_reference_checkpoint_is_evaluation_only_and_its_hash_is_preserved(self):
        torch.manual_seed(42)
        model = Autoencoder(vocab_size=3, sequence_length=5, embedding_dim=3, hidden_size=4, latent_dim=2)
        path = self.root / "historical_best.pt"
        torch.save({"state_dict": model.state_dict(), "model_hparams": model.hparams,
                    "vocab": self.pipeline.vocab, "data_contract": self.pipeline.contract(),
                    "model_type": "ae", "checkpoint_role": "best_inference_only",
                    "epoch": 8, "optimizer_state_dict": {"must_not_restore": True}}, path)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.object(torch.optim.Adam, "load_state_dict", side_effect=AssertionError("best optimizer restored")):
            self.run_small("reference_eval", reference_checkpoint=path)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_nonfinite_gradient_stops_without_any_adam_update(self):
        with patch.object(experiments, "stable_clip_grad_norm_",
                          side_effect=experiments.NonFiniteGradientError("injected NaN gradient")), \
             patch.object(torch.optim.Adam, "step", side_effect=AssertionError("invalid step")):
            result = self.run_small("invalid")
        self.assertTrue(result["trials"])
        self.assertEqual(result["trials"][0]["updates"], 0)
        self.assertNotEqual(result["trials"][0]["status"], "completed")
        self.assertTrue(any(word in json.dumps(result).lower() for word in ("nonfinite", "nan", "numerical")))

    def test_time_budget_stops_without_waiting_for_real_time(self):
        elapsed = 0.0
        def advancing_time():
            nonlocal elapsed
            elapsed += 20.0
            return elapsed
        with patch.object(experiments.time, "monotonic", side_effect=advancing_time), \
             patch.object(torch.optim.Adam, "step", side_effect=AssertionError("step after expired deadline")):
            result = self.run_small("timed")
        self.assertTrue(any(word in json.dumps(result).lower() for word in ("time", "deadline")))
        self.assertTrue(all(trial["updates"] <= 2 for trial in result.get("trials", [])))

    def test_extreme_validation_deterioration_stops_before_spending_the_budget(self):
        plan = copy.deepcopy(self.plan)
        plan["limits"]["deterioration_patience"] = 1
        original = experiments._evaluate
        evaluations = 0
        def deteriorated(*args, **kwargs):
            nonlocal evaluations
            report = original(*args, **kwargs)
            evaluations += 1
            # Controlled injected metric: the first snapshot is the baseline;
            # later snapshots deliberately violate the declared ratio guard.
            return {**report, "val_reconstruction": report["val_reconstruction"] * (1.0 if evaluations == 1 else 100.0)}
        with patch.object(experiments, "_evaluate", side_effect=deteriorated):
            result = experiments.run_suite(plan, self.pipeline, self.root / "deterioration",
                                           variants=["reference"], budget="updates", device="cpu", execute=True)
        trial = result["trials"][0]
        self.assertNotEqual(trial["status"], "completed")
        self.assertLess(trial["updates"], 2)
        self.assertTrue(any(word in json.dumps(trial).lower() for word in ("reconstruction", "deterioration")))

    def test_suite_update_limit_is_shared_across_variants(self):
        plan = copy.deepcopy(self.plan)
        plan["limits"]["max_total_updates"] = 2
        result = experiments.run_suite(plan, self.pipeline, self.root / "aggregate_limit",
                                       variants=["reference", "batch4"], budget="updates",
                                       device="cpu", execute=True)
        self.assertLessEqual(sum(trial["updates"] for trial in result["trials"]), 2)


if __name__ == "__main__":
    unittest.main()
