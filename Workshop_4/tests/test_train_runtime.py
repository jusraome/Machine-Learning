"""Bounded CPU regressions for opt-in data transport and sampled profiling."""
import hashlib
import importlib.util
import json
import random
from threading import Event
import unittest

HAS_DEPENDENCIES = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy", "pyarrow"))
if HAS_DEPENDENCIES:
    import torch
    from src import train_ae
    from src.runtime import iter_prefetched, validate_runtime
    import test_train_ae as fixture_module


@unittest.skipUnless(HAS_DEPENDENCIES, "runtime checks require torch, numpy and pyarrow")
class TrainingRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        # Reuse the tiny Parquet fixture without inheriting/repeating its tests.
        self.fixture = fixture_module.TrainingIntegrationTests("test_short_training_requires_explicit_smoke_mode")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.pipeline = self.fixture.pipeline
        self.config = self.fixture.config
        original_tensor_reader = self.pipeline.iter_tensor_batches

        def guarded_tensor_reader(split, batch_size=None):
            self.assertIn(split, ("train", "val"))
            self.fixture.seen_splits.append(split)
            yield from original_tensor_reader(split, batch_size=batch_size)

        self.pipeline.iter_tensor_batches = guarded_tensor_reader

    def test_tensor_loader_preserves_shuffle_order_filtering_and_global_rng(self):
        torch_state = torch.get_rng_state().clone()
        python_state = random.getstate()
        for split in ("train", "val"):
            for epoch in (1, 2):
                for normal_class in (None, 1):
                    config = {**self.config, "normal_class": normal_class}
                    with self.subTest(split=split, epoch=epoch, normal_class=normal_class):
                        expected = list(train_ae._batches(self.pipeline, split, config, epoch))
                        actual = list(iter_prefetched(
                            train_ae._batches(self.pipeline, split, config, epoch, tensor_loader=True), 2))
                        self.assertEqual(len(actual), len(expected))
                        for left, right in zip(expected, actual):
                            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(random.getstate(), python_state)

    def test_prefetch_preserves_order_propagates_errors_and_closes_on_cancellation(self):
        for depth in (0, 2):
            self.assertEqual(list(iter_prefetched(iter(range(7)), depth)), list(range(7)))
        closed = Event()

        def failing_source():
            try:
                yield "first"
                yield "second"
                raise ValueError("reader failure")
            finally:
                closed.set()

        reader = iter_prefetched(failing_source(), 2)
        self.assertEqual(next(reader), "first")
        self.assertEqual(next(reader), "second")
        with self.assertRaisesRegex(ValueError, "reader failure"):
            next(reader)
        self.assertTrue(closed.wait(1.0))
        cancelled = Event()

        def cancellable_source():
            try:
                yield from range(100)
            finally:
                cancelled.set()

        reader = iter_prefetched(cancellable_source(), 2)
        self.assertEqual(next(reader), 0)
        reader.close()
        self.assertTrue(cancelled.wait(1.0))

    def test_tiny_cpu_training_with_prefetch_matches_baseline_model_adam_metrics_and_rng(self):
        self.fixture.run_tiny("baseline")
        self.fixture.run_tiny("optimized", runtime={"tensor_loader": True, "prefetch_batches": 2})
        baseline = self.fixture.checkpoint("baseline")
        optimized = self.fixture.checkpoint("optimized")
        self.fixture.assert_nested_equal(baseline["state_dict"], optimized["state_dict"])
        self.fixture.assert_nested_equal(baseline["optimizer_state_dict"], optimized["optimizer_state_dict"])
        self.fixture.assert_nested_equal(baseline["rng_state"], optimized["rng_state"])
        self.assertEqual(baseline["training_config"], optimized["training_config"])
        for left, right in zip(baseline["history"], optimized["history"]):
            for key in left.keys() - {"seconds"}:
                self.assertEqual(left[key], right[key], key)
        self.assertNotIn("test", self.fixture.seen_splits)

    def test_profile_samples_only_first_batch_and_exports_all_training_stages(self):
        config = {**self.config, "epochs": 1}
        self.fixture.run_tiny("profiled", config=config, runtime={"profile_batches": 1})
        paths = list((self.root / "profiled/tiny/profiles").glob("*.json"))
        self.assertEqual(len(paths), 2)
        profiles = {item["split"]: item for item in
                    (json.loads(path.read_text(encoding="utf-8")) for path in paths)}
        self.assertEqual(set(profiles), {"train", "val"})
        self.assertEqual(profiles["train"]["batch_limit"], 1)
        for name in ("read_prepare", "h2d", "forward_loss", "backward", "clipping", "optimizer",
                     "diagnostics_logging", "metrics_logging"):
            with self.subTest(stage=name):
                stage = profiles["train"]["stages"][name]
                self.assertEqual(stage["calls"], 1)
                self.assertGreaterEqual(stage["wall_ms_total"], 0.0)
        self.assertEqual(profiles["val"]["stages"]["forward_loss"]["calls"], 1)
        self.assertNotIn("backward", profiles["val"]["stages"])

    def test_recommended_matrix_is_explicit_five_and_keeps_optional_experiments(self):
        path = train_ae.ROOT / "config/experiments.yaml"
        recommended = train_ae.recommended_experiments(path)
        self.assertEqual(recommended, ["ae_z32", "ae_z64", "vae_z32_b0p5", "vae_z32_b1", "vae_z32_b4"])
        all_experiments = train_ae.load_experiments(path)
        self.assertEqual(len(all_experiments), 12)
        self.assertEqual(len({item["name"] for item in all_experiments} - set(recommended)), 7)
        args = train_ae.build_parser().parse_args(["--recommended", "--dry-run"])
        self.assertTrue(args.recommended)
        self.assertFalse(args.all)
        self.assertIsNone(args.experiment)
        with self.assertRaises(ValueError):
            validate_runtime({"prefetch_batches": 9})
        with self.assertRaises(ValueError):
            validate_runtime({"profile_batches": True})

    def test_auto_resume_rejects_transport_change_but_separate_continuation_preserves_source(self):
        self.fixture.run_tiny("source", stop_after_epoch=1)
        source = self.root / "source/tiny/checkpoints/last.pt"
        original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        runtime = {"tensor_loader": True, "prefetch_batches": 2}
        with self.assertRaisesRegex(ValueError, "Transport options changed"):
            self.fixture.run_tiny("source", resume="auto", runtime=runtime)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)
        with self.assertWarnsRegex(RuntimeWarning, "Transport configuration changed"):
            self.fixture.run_tiny("continuation", resume=source, runtime=runtime)
        self.assertEqual(self.fixture.checkpoint("continuation")["epoch"], 2)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)


if __name__ == "__main__":
    unittest.main()
