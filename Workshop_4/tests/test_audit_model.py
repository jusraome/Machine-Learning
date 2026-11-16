"""Bounded read-only probes; constructed checkpoints are not training results."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.audit_model import compare_checkpoints, probe_model, recurrent_spectra
from src.autoencoder import Autoencoder, VariationalAutoencoder, reconstruction_loss
from src.data import DataPipeline


class ModelAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_probe_preserves_parameters_gradients_mode_rng_and_loss_definition(self):
        torch.manual_seed(42)
        x = torch.tensor([[0, 1, 2, 1, 0], [2, 1, 0, 1, 2]])
        for model_type in (Autoencoder, VariationalAutoencoder):
            with self.subTest(model=model_type.__name__):
                model = model_type(vocab_size=3, sequence_length=5, embedding_dim=3, hidden_size=4, latent_dim=2)
                model.train()
                model.encoder.eval()
                modes = {name: module.training for name, module in model.named_modules()}
                for parameter in model.parameters():
                    parameter.grad = torch.ones_like(parameter)
                parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
                gradients = {name: value.grad.clone() for name, value in model.named_parameters()}
                rng = torch.get_rng_state().clone()
                report = probe_model(model, x, horizons=(1, 3, 5))
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(model.training)
                self.assertEqual({name: module.training for name, module in model.named_modules()}, modes)
                for name, value in model.named_parameters():
                    self.assertTrue(torch.equal(value, parameters[name]), name)
                    self.assertTrue(torch.equal(value.grad, gradients[name]), name)
                model.eval()
                with torch.no_grad():
                    output = model(x, sample=False) if model_type is VariationalAutoencoder else model(x)
                    expected = reconstruction_loss(output.logits, x).item()
                self.assertAlmostEqual(report["reconstruction_nats_per_sequence"], expected, places=6)
                self.assertLessEqual(report["logit_gradient"]["max_abs_finite"], 1 / len(x) + 1e-7)
                self.assertEqual(set(report["decoder_position_vjp"]), {"1", "3", "5"})
                self.assertEqual(report["decoder_input_stride"][1], 0)
                self.assertTrue(report["latent_gradient"]["finite"])
                self.assertIn("decoder", report["gradient_blocks"])

    def test_gate_spectra_report_each_recurrent_gate_without_claiming_jacobian_bound(self):
        model = Autoencoder(vocab_size=3, sequence_length=5, embedding_dim=3, hidden_size=4, latent_dim=2)
        spectra = recurrent_spectra(model)
        self.assertEqual(set(spectra), {"encoder.weight_hh_l0", "decoder.weight_hh_l0"})
        self.assertEqual(set(spectra["encoder.weight_hh_l0"]), {"reset", "update", "candidate"})
        for gates in spectra.values():
            self.assertTrue(all(item["finite"] and item["spectral_norm"] >= 0 for item in gates.values()))

    def test_input_dependence_detects_constructed_constant_latent_without_training(self):
        model = Autoencoder(vocab_size=3, sequence_length=5, embedding_dim=3, hidden_size=4, latent_dim=2)
        with torch.no_grad():
            model.to_latent.weight.zero_()
            model.to_latent.bias.zero_()
        x = torch.tensor([[0, 1, 2, 1, 0], [2, 1, 0, 1, 2]])
        report = probe_model(model, x, horizons=(1, 5))
        dependence = report["input_dependence"]
        self.assertTrue(dependence["permutation_meaningful"])
        for name in ("original_latent", "permuted_latent", "zero_latent"):
            self.assertEqual(dependence[name]["relative_logits_l2_change"], 0.0)
            self.assertEqual(dependence[name]["mean_probability_variance_between_examples"], 0.0)
        self.assertEqual(dependence["zero_latent"]["reconstruction_nats_per_sequence"],
                         report["reconstruction_nats_per_sequence"])

    def test_checkpoint_comparison_is_read_only_and_never_reads_test(self):
        with tempfile.TemporaryDirectory(prefix="workshop4-model-audit-") as temporary:
            root = Path(temporary)
            vocab = {"itos": ["<UNK>", "a", "b"], "stoi": {"<UNK>": 0, "a": 1, "b": 2},
                     "size": 3, "unk_index": 0}
            splits = {"character_ids": {"train": ["A"], "val": ["B"], "test": ["C"]}}
            for name, payload in (("vocab", vocab), ("splits", splits)):
                (root / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
            pq.write_table(pa.Table.from_pylist([{"character_id": "B", "x": "ababa", "y": "b", "y_id": 2}]),
                           root / "val.parquet")
            pipeline = DataPipeline({"vocab_path": root / "vocab.json", "splits_path": root / "splits.json",
                                     "processed_dir": root, "raw_path": root / "raw.parquet", "window_size": 5,
                                     "batch_size": 1, "dataset": "synthetic", "revision": "test"})
            torch.manual_seed(42)
            model = Autoencoder(vocab_size=3, sequence_length=5, embedding_dim=3, hidden_size=4, latent_dim=2)
            paths = {}
            for label in ("best", "last"):
                paths[label] = root / f"{label}.pt"
                torch.save({"state_dict": model.state_dict(), "model_hparams": model.hparams,
                            "model_type": "ae", "vocab": vocab, "data_contract": pipeline.contract(),
                            "epoch": 1, "best_epoch": 1, "training_config": {"beta": 1.0}}, paths[label])
            before = {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in paths.items()}
            rng = torch.get_rng_state().clone()
            report = compare_checkpoints(paths, pipeline, batch_size=1, horizons=(1, 5))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(report["optimizer_steps"], 0)
            first, second = report["checkpoints"]
            self.assertEqual(first["probe"], second["probe"])
            for label, path in paths.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before[label])
            with self.assertRaises(ValueError):
                compare_checkpoints(paths, pipeline, split="test")
            json.dumps(report, allow_nan=False)

    def test_missing_checkpoint_is_explicit_and_does_not_claim_comparison(self):
        class MissingPipeline:
            def contract(self):
                return {"dataset": "synthetic"}
            def iter_batches(self, *args, **kwargs):
                raise AssertionError("No dataset read when no checkpoints are available")
        with tempfile.TemporaryDirectory() as temporary:
            report = compare_checkpoints({"best": Path(temporary) / "missing.pt"}, MissingPipeline())
            self.assertFalse(report["checkpoints"][0]["available"])
            self.assertTrue(any("fewer than two" in item for item in report["limitations"]))


if __name__ == "__main__":
    unittest.main()
