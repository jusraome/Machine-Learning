"""Bounded Phase 3 orchestration tests; synthetic data, at most two tiny updates.

All temporary artifacts live under Workshop_4/tmp. No production checkpoint,
Shakespeare data, full trainer or experiment matrix is opened by these tests.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if AVAILABLE:
    import torch
    from src import latent_diagnostic_runner as runner
    from src.autoencoder import AEOutput, Autoencoder


class SyntheticPipeline:
    def __init__(self):
        self.vocab = {"itos": ["<UNK>", "a", "b"],
                      "stoi": {"<UNK>": 0, "a": 1, "b": 2},
                      "size": 3, "unk_index": 0}
        self.config = {"window_size": 5, "batch_size": 2}
        self.seen_splits = []
        self.values = {
            "train": [[0, 1, 2, 1, 2], [2, 2, 1, 1, 1], [1, 2, 1, 0, 1], [2, 1, 2, 1, 2],
                      [1, 1, 2, 2, 1], [2, 1, 0, 1, 2], [1, 2, 2, 1, 2], [2, 1, 1, 2, 1]],
            "val": [[1, 2, 1, 1, 2], [2, 1, 0, 1, 2], [1, 1, 1, 2, 2], [2, 2, 1, 2, 1]],
        }

    def iter_tensor_batches(self, split, batch_size=None):
        if split not in ("train", "val"):
            raise AssertionError("Phase 3 must never read test")
        self.seen_splits.append(split)
        values = self.values[split]
        for start in range(0, len(values), batch_size or 2):
            x = torch.tensor(values[start:start + (batch_size or 2)], dtype=torch.long)
            yield {"x": x, "y": torch.zeros(len(x), dtype=torch.long)}

    def contract(self):
        return {"window_size": 5, "dataset": "synthetic_phase3"}

    def validate_checkpoint(self, value):
        if value["vocab"] != self.vocab or value["data_contract"] != self.contract():
            raise ValueError("incompatible synthetic checkpoint")
        return True


@unittest.skipUnless(AVAILABLE, "Phase 3 runner tests require torch and numpy")
class LatentDiagnosticRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="phase3-unit-", dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pipeline = SyntheticPipeline()
        self.settings = {"max_updates": 2, "val_examples": 4, "batch_size": 2,
                         "embedding_dim": 3, "hidden_size": 4, "latent_dim": 2,
                         "max_seconds": 30, "max_seconds_per_variant": 20}

    def run_small(self, name="evaluation", **kwargs):
        return runner.run_diagnostics(self.pipeline, self.root / name, settings=self.settings,
                                      device="cpu", execute=True, **kwargs)

    def test_dry_run_does_not_read_data_construct_optimizer_or_create_files(self):
        destination = self.root / "dry"
        with patch.object(self.pipeline, "iter_tensor_batches", side_effect=AssertionError("dry data read")), \
             patch.object(torch.optim, "Adam", side_effect=AssertionError("dry optimizer")):
            plan = runner.run_diagnostics(self.pipeline, destination, settings=self.settings,
                                          short_probe=True, execute=False)
        self.assertFalse(destination.exists())
        self.assertEqual(plan["architectures"], ["identity", "tanh"])
        self.assertEqual(plan["max_total_updates"], 4)
        self.assertFalse(self.pipeline.seen_splits)

    def test_existing_output_is_refused_even_for_a_dry_plan(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("historical", encoding="utf-8")
        for execute in (False, True):
            with self.assertRaises(FileExistsError):
                runner.run_diagnostics(self.pipeline, output, execute=execute)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "historical")

    def test_eval_only_fits_full_train_without_constructing_an_optimizer(self):
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("evaluation optimizer")):
            result = self.run_small()
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["probes"])
        self.assertEqual(set(self.pipeline.seen_splits), {"train", "val"})
        baseline = json.loads((self.root / "evaluation/baseline.json").read_text(encoding="utf-8"))
        self.assertEqual(baseline["train_examples"], len(self.pipeline.values["train"]))
        self.assertEqual(baseline["train_tokens"], 5 * len(self.pipeline.values["train"]))
        self.assertEqual(baseline["scope"], "full_train")

    def test_checkpoint_is_read_only_and_optimizer_state_is_never_restored(self):
        torch.manual_seed(42)
        model = Autoencoder(vocab_size=3, sequence_length=5, embedding_dim=3, hidden_size=4, latent_dim=2)
        checkpoint = self.root / "historical_best.pt"
        torch.save({"state_dict": model.state_dict(), "model_hparams": model.hparams,
                    "vocab": self.pipeline.vocab, "data_contract": self.pipeline.contract(),
                    "model_type": "ae", "epoch": 8,
                    "optimizer_state_dict": {"must_not_restore": True}}, checkpoint)
        before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        with patch.object(torch.optim.Adam, "load_state_dict", side_effect=AssertionError("restored Adam")), \
             patch.object(torch.optim, "Adam", side_effect=AssertionError("evaluation optimizer")):
            result = self.run_small("historical_evaluation", checkpoints={"best8": checkpoint})
        self.assertEqual(result["status"], "completed", result.get("reason"))
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)
        self.assertEqual(result["checkpoints"][0]["sha256"], before)
        self.assertEqual(result["checkpoints"][0]["optimizer_steps"], 0)

    def test_two_architectures_share_weights_and_ordered_data_and_call_fp64_clipping(self):
        training = {}
        original_forward = Autoencoder.forward
        def record(model, x):
            if model.training and torch.is_grad_enabled():
                training.setdefault(model.latent_activation, []).extend(x.detach().cpu().tolist())
            return original_forward(model, x)
        with patch.object(Autoencoder, "forward", new=record), \
             patch.object(runner, "stable_clip_grad_norm_", wraps=runner.stable_clip_grad_norm_) as clipping:
            result = self.run_small("paired", short_probe=True)
        self.assertEqual(result["status"], "completed", result.get("reason"))
        self.assertEqual(len(result["probes"]), 2)
        identity, tanh = result["probes"]
        self.assertEqual(identity["initial_state_sha256"], tanh["initial_state_sha256"])
        self.assertEqual(training["identity"], training["tanh"])
        self.assertEqual(len(training["identity"]), 4)
        self.assertEqual(set(self.pipeline.seen_splits), {"train", "val"})
        self.assertEqual(clipping.call_count, 4)
        for entry in result["probes"]:
            self.assertEqual(entry["updates"], 2)
            self.assertEqual(entry["examples"], 4)
            self.assertEqual([item["update"] for item in entry["snapshots"]], [0, 1, 2])
            self.assertTrue(all(row["norm_after"] <= 5.0 * (1 + 1e-6) for row in entry["training"]))

    def test_nonfinite_training_loss_prevents_every_optimizer_step(self):
        original_forward = Autoencoder.forward
        def nonfinite_training(model, x):
            output = original_forward(model, x)
            if model.training and torch.is_grad_enabled():
                return AEOutput(logits=output.logits * float("nan"), z=output.z)
            return output
        with patch.object(Autoencoder, "forward", new=nonfinite_training), \
             patch.object(torch.optim.Adam, "step", side_effect=AssertionError("nonfinite Adam step")):
            result = self.run_small("nonfinite", short_probe=True)
        self.assertEqual(result["status"], "partial")
        self.assertTrue(all(entry["updates"] == 0 for entry in result["probes"]))
        self.assertTrue(all("nonfinite" in entry["reason"].lower() for entry in result["probes"]))

    def test_rejected_gradient_prevents_every_optimizer_step(self):
        with patch.object(runner, "stable_clip_grad_norm_", side_effect=runner.NonFiniteGradientError("injected invalid gradient")), \
             patch.object(torch.optim.Adam, "step", side_effect=AssertionError("invalid Adam step")):
            result = self.run_small("gradient", short_probe=True)
        self.assertEqual(result["status"], "partial")
        self.assertTrue(all(entry["updates"] == 0 for entry in result["probes"]))
        self.assertTrue(all("gradient" in entry["reason"] for entry in result["probes"]))

    def test_deadline_stops_before_optimizer_step_and_writes_partial_summary(self):
        def expired(deadline):
            raise TimeoutError("injected time budget exhausted")
        with patch.object(runner, "_deadline", side_effect=expired), \
             patch.object(torch.optim.Adam, "step", side_effect=AssertionError("expired step")):
            result = self.run_small("timeout", short_probe=True)
        self.assertEqual(result["status"], "failed")
        self.assertIn("time budget", result["reason"])
        self.assertTrue((self.root / "timeout/summary.json").is_file())
        self.assertFalse(result["probes"])

    def test_hard_caps_cannot_expand_the_two_probe_protocol(self):
        for key, invalid in (("max_updates", 25), ("val_examples", 129), ("batch_size", 129),
                             ("max_seconds", 601), ("max_seconds_per_variant", 181),
                             ("hidden_size", 129), ("embedding_dim", 33), ("latent_dim", 33),
                             ("num_layers", 2), ("learning_rate", 0.0011), ("dropout", 0.1),
                             ("max_updates", True), ("learning_rate", float("nan")),
                             ("epochs", 30)):
            with self.subTest(key=key, value=invalid), self.assertRaises(ValueError):
                runner.validate_settings({key: invalid})


if __name__ == "__main__":
    unittest.main()
