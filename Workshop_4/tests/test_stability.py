"""Tiny CPU fault injection: rejected gradients never reach Adam.

These artificial NaN/Inf cases are numerical safety tests, not dataset metrics.
No Drive checkpoint or real training run is read, changed or resumed here.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "pyarrow"))
if AVAILABLE:
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.autoencoder import Autoencoder, reconstruction_loss
    from src.data import DataPipeline
    from src import train_ae


@unittest.skipUnless(AVAILABLE, "stability checks require torch and pyarrow")
class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="workshop4-stability-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.manual_seed(42)
        self.cfg = train_ae.validate_experiment({"name": "tiny", "model": "ae", "epochs": 2,
            "batch_size": 2, "hidden_size": 4, "embedding_dim": 3, "latent_dim": 3,
            "learning_rate": 0.001, "seed": 42, "shuffle_buffer_batches": 1}, smoke_test=True)
        self.batches = [torch.tensor([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]),
                        torch.tensor([[1, 2, 1, 2, 1], [2, 1, 2, 1, 2]])]

    def model_optimizer(self):
        model = Autoencoder(vocab_size=3, sequence_length=5, embedding_dim=3,
                            hidden_size=4, latent_dim=3)
        return model, torch.optim.Adam(model.parameters(), lr=0.001)

    def monitor(self, directory="events", **policy):
        return train_ae._StabilityMonitor(self.root / directory,
            policy={"max_skipped_fraction": 1.0, **policy})

    def run_epoch(self, model, optimizer, monitor, epoch=12):
        def batches(*args, **kwargs):
            yield from (x.clone() for x in self.batches)
        with patch.object(train_ae, "_batches", side_effect=batches):
            return train_ae._epoch(model, None, self.cfg, epoch, torch.device("cpu"),
                                   optimizer=optimizer, monitor=monitor)

    def inject_once(self, model, value):
        calls = 0
        def corrupt_first_gradient(gradient):
            nonlocal calls
            calls += 1
            return torch.full_like(gradient, value) if calls == 1 else gradient
        handle = model.output_projection.weight.register_hook(corrupt_first_gradient)
        self.addCleanup(handle.remove)
        return handle

    def events(self, directory="events"):
        return [json.loads(line) for line in
                (self.root / directory / "nonfinite_events.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_isolated_nan_and_inf_skip_adam_and_do_not_bias_accepted_metrics(self):
        for label, invalid in (("nan", float("nan")), ("inf", float("inf"))):
            with self.subTest(invalid=label):
                model, optimizer = self.model_optimizer()
                monitor = self.monitor(directory=label)
                before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                with torch.no_grad():
                    expected = reconstruction_loss(model(self.batches[1]).logits, self.batches[1]).item()
                self.inject_once(model, invalid)
                actual_step = optimizer.step
                def checked_step(*args, **kwargs):
                    # At the first accepted step, the rejected batch must have
                    # changed neither parameters nor Adam's initially empty state.
                    self.assertFalse(optimizer.state)
                    for name, parameter in model.named_parameters():
                        self.assertTrue(torch.equal(parameter, before[name]), name)
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                    return actual_step(*args, **kwargs)
                with patch.object(optimizer, "step", side_effect=checked_step) as step:
                    result = self.run_epoch(model, optimizer, monitor)
                self.assertEqual(step.call_count, 1)
                self.assertEqual(result["attempted_batches"], 2)
                self.assertEqual(result["skipped_batches"], 1)
                self.assertEqual(result["accepted_batches"], 1)
                self.assertEqual(result["attempted_examples"], 4)
                self.assertEqual(result["skipped_examples"], 2)
                self.assertEqual(result["examples"], 2)
                self.assertAlmostEqual(result["total"], expected, places=5)
                self.assertEqual(monitor.state["total_skipped_batches"], 1)
                event = next(e for e in self.events(label) if e.get("reason") == "nonfinite_gradient_elements")
                self.assertEqual(event["epoch"], 12)
                self.assertIn("output_projection.weight", json.dumps(event))

    def test_finite_elements_with_overflowed_fp32_norm_are_clipped_and_accepted(self):
        model, optimizer = self.model_optimizer()
        self.inject_once(model, 1e20)
        actual_step = optimizer.step
        def checked_step(*args, **kwargs):
            gradients = [p.grad.detach().double().reshape(-1) for p in model.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))
            self.assertLessEqual(torch.linalg.vector_norm(torch.cat(gradients)).item(), 5.0 * (1 + 1e-6))
            return actual_step(*args, **kwargs)
        monitor = self.monitor()
        with patch.object(optimizer, "step", side_effect=checked_step) as step:
            result = self.run_epoch(model, optimizer, monitor)
        self.assertEqual(step.call_count, 2)
        self.assertEqual(result["skipped_batches"], 0)
        self.assertEqual(monitor.state["total_skipped_batches"], 0)
        self.assertEqual(result["large_grad_batches"], 1)
        event = next(e for e in self.events() if e.get("reason") == "large_finite_gradient_clipped")
        self.assertGreater(event["norm_before"], 1e20)
        self.assertLessEqual(event["norm_after"], 5.0 * (1 + 1e-6))
        with (self.root / "events" / "gradient_norms.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual([int(row["batch_index"]) for row in rows], [1, 2])
        self.assertEqual(float(rows[0]["norm_before"]), event["norm_before"])
        self.assertTrue(all(float(row["norm_after"]) <= 5.0 * (1 + 1e-6) for row in rows))

    def test_unrelated_clipping_runtime_error_is_never_reinterpreted_as_recoverable(self):
        model, optimizer = self.model_optimizer()
        self.inject_once(model, float("nan"))
        monitor = self.monitor()
        with patch.object(train_ae, "stable_clip_grad_norm_", side_effect=RuntimeError("CUDA out of memory")), \
             patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                self.run_epoch(model, optimizer, monitor)
        self.assertEqual(step.call_count, 0)
        self.assertEqual(monitor.state["total_skipped_batches"], 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(self.events()[0]["reason"], "clipping_exception")

    def test_repeated_incidents_abort_before_step_and_clear_rejected_gradients(self):
        model, optimizer = self.model_optimizer()
        hook = model.output_projection.weight.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        self.addCleanup(hook.remove)
        monitor = self.monitor()
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaises(FloatingPointError):
                self.run_epoch(model, optimizer, monitor)
        self.assertEqual(step.call_count, 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(monitor.state["total_skipped_batches"], 2)
        rejected = [e for e in self.events() if e.get("reason") == "nonfinite_gradient_elements"]
        self.assertEqual(len(rejected), 2)

    def test_excessive_fraction_aborts_even_after_other_batches_are_valid(self):
        model, optimizer = self.model_optimizer()
        self.inject_once(model, float("nan"))
        with self.assertRaises(FloatingPointError):
            self.run_epoch(model, optimizer, self.monitor(max_skipped_fraction=0.1))
        # One out of two batches is 50%; it cannot be presented as a valid epoch.
        self.assertFalse((self.root / "events" / "checkpoints" / "last.pt").exists())

    def test_nonfinite_validation_loss_is_not_skipped(self):
        model, _ = self.model_optimizer()
        monitor = self.monitor()
        def batches(*args, **kwargs):
            yield self.batches[0]
        with patch.object(train_ae, "_batches", side_effect=batches), \
             patch.object(train_ae, "reconstruction_loss", return_value=torch.tensor(float("nan"))):
            with self.assertRaises(FloatingPointError):
                train_ae._epoch(model, None, self.cfg, 12, torch.device("cpu"), monitor=monitor)
        self.assertEqual(monitor.state["total_skipped_batches"], 0)
        self.assertTrue((self.root / "events" / "nonfinite_events.jsonl").is_file())

    def test_rejection_budget_survives_monitor_recreation(self):
        model, optimizer = self.model_optimizer()
        hook = self.inject_once(model, float("nan"))
        first = self.monitor(max_skipped_batches_total=1)
        self.run_epoch(model, optimizer, first)
        hook.remove()
        second = self.monitor(max_skipped_batches_total=1)
        self.assertEqual(second.state["total_skipped_batches"], 1)
        self.inject_once(model, float("inf"))
        with self.assertRaises(FloatingPointError):
            self.run_epoch(model, optimizer, second, epoch=13)
        self.assertEqual(second.state["total_skipped_batches"], 2)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def make_pipeline(self):
        data = self.root / "data"
        data.mkdir()
        vocab = {"itos": ["<UNK>", "a", "b"], "stoi": {"<UNK>": 0, "a": 1, "b": 2},
                 "size": 3, "unk_index": 0}
        splits = {"character_ids": {"train": ["A"], "val": ["B"], "test": ["C"]}}
        for name, payload in (("vocab", vocab), ("splits", splits)):
            (data / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
        for split, speaker in (("train", "A"), ("val", "B")):
            rows = [{"character_id": speaker, "x": "ababa", "y": "b", "y_id": 2} for _ in range(4)]
            pq.write_table(pa.Table.from_pylist(rows), data / f"{split}.parquet")
        return DataPipeline({"vocab_path": data / "vocab.json", "splits_path": data / "splits.json",
                             "processed_dir": data, "raw_path": data / "raw.parquet",
                             "window_size": 5, "batch_size": 2, "dataset": "synthetic", "revision": "test"})

    def test_legacy_checkpoint_resumes_to_new_output_without_changing_source(self):
        pipeline = self.make_pipeline()
        train_ae.run_experiment(self.cfg, pipeline, self.root / "original", device="cpu",
                               smoke_test=True, stop_after_epoch=1)
        source_dir = self.root / "original" / "tiny"
        source_path = source_dir / "checkpoints" / "last.pt"
        legacy = torch.load(source_path, map_location="cpu", weights_only=True)
        legacy.pop("stability_policy", None)
        legacy.pop("stability_state", None)
        legacy.pop("continuation_origin", None)
        legacy.pop("gradient_clipping", None)
        for row in legacy["history"]:
            for field in train_ae.HISTORY_FIELDS[11:]:
                row.pop(field, None)
        # This synthetic v1 checkpoint has no diagnostics, like the old trainer.
        torch.save(legacy, source_path)
        protected = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_dir.rglob("*") if path.is_file()}
        train_ae.run_experiment(self.cfg, pipeline, self.root / "recovered", device="cpu",
                               smoke_test=True, resume=source_path)
        restored = torch.load(self.root / "recovered/tiny/checkpoints/last.pt",
                              map_location="cpu", weights_only=True)
        self.assertEqual(restored["epoch"], 2)
        self.assertEqual(restored["training_config"], legacy["training_config"])
        self.assertEqual(restored["stability_state"]["total_skipped_batches"], 0)
        self.assertTrue(restored["optimizer_state_dict"]["state"])
        self.assertEqual(restored["gradient_clipping"]["algorithm"], "global_l2_fp64_v1")
        self.assertEqual(restored["training_config"]["grad_clip"], 5.0)
        self.assertEqual(restored["continuation_origin"]["sha256"], protected[source_path])
        self.assertNotIn("train_skipped_batches", restored["history"][0])
        for path, before in protected.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before, str(path))
        with self.assertRaises(FileExistsError):
            train_ae.run_experiment(self.cfg, pipeline, self.root / "original", device="cpu",
                                   smoke_test=True, resume=source_path)

    def test_readable_checkpoint_with_corrupt_adam_state_is_rejected(self):
        pipeline = self.make_pipeline()
        train_ae.run_experiment(self.cfg, pipeline, self.root / "source", device="cpu",
                               smoke_test=True, stop_after_epoch=1)
        source_path = self.root / "source/tiny/checkpoints/last.pt"
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
        state = next(iter(checkpoint["optimizer_state_dict"]["state"].values()))
        state["exp_avg"].fill_(float("nan"))
        torch.save(checkpoint, source_path)
        before = hashlib.sha256(source_path.read_bytes()).hexdigest()
        with self.assertRaises(FloatingPointError):
            train_ae.run_experiment(self.cfg, pipeline, self.root / "bad-recovery", device="cpu",
                                   smoke_test=True, resume=source_path)
        self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), before)
        self.assertFalse((self.root / "bad-recovery/tiny/checkpoints/last.pt").exists())


if __name__ == "__main__":
    unittest.main()
