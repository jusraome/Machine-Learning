"""Read-only anomaly scoring/calibration tests on synthetic labelled windows.

No production checkpoint, Shakespeare training or test partition is used.
The labels are the existing next-character y class, not semantic anomalies.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import csv
import tempfile
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if AVAILABLE:
    import torch
    from src import anomaly_evaluation as anomalies
    from src.autoencoder import AEOutput, Autoencoder


class SyntheticPipeline:
    def __init__(self):
        self.vocab = {"itos": ["<UNK>", "a", "b"],
                      "stoi": {"<UNK>": 0, "a": 1, "b": 2}, "size": 3, "unk_index": 0}
        self.config = {"window_size": 5, "batch_size": 4}
        self.groups = {"train": {"TRAIN"}, "val": {"V1", "V2", "V3", "V4"}, "test": {"TEST"}}
        self.allow_test = False
        self.rows = {"train": [{"character_id": "TRAIN", "x": "ababa", "y": "a" if i < 11 else "b",
                                "y_id": 1 if i < 11 else 2} for i in range(16)],
                     "val": [{"character_id": speaker, "x": "ababa" if i % 2 else "baZab",
                              "y": "a" if i % 2 else "b", "y_id": 1 if i % 2 else 2}
                             for speaker in sorted(self.groups["val"]) for i in range(16)],
                     "test": [{"character_id": "TEST", "x": "abbba" if i % 2 else "abZbb",
                               "y": "a" if i % 2 else "b", "y_id": 1 if i % 2 else 2} for i in range(16)]}
        self.splits = {"character_ids": {name: sorted(group) for name, group in self.groups.items()},
                       "counts": {name: {"n_rows": len(self.rows.get(name, [])), "n_characters": len(group)}
                                  for name, group in self.groups.items()}}
        self.seen_splits = []

    def _iter_rows_batches(self, split, batch_size=None):
        if split not in ("train", "val") and not (split == "test" and self.allow_test):
            raise AssertionError("Anomaly preparation must not read test")
        self.seen_splits.append(split)
        for start in range(0, len(self.rows[split]), batch_size or 4):
            yield self.rows[split][start:start + (batch_size or 4)]

    def encode_tensor_rows(self, rows, split):
        if split not in ("train", "val") and not (split == "test" and self.allow_test):
            raise AssertionError("Test is not part of calibration")
        if any(row["character_id"] not in self.groups[split] for row in rows):
            raise ValueError("speaker outside split")
        return {"x": torch.tensor([[self.vocab["stoi"].get(c, 0) for c in row["x"]] for row in rows], dtype=torch.long),
                "y": torch.tensor([self.vocab["stoi"].get(row["y"], 0) for row in rows], dtype=torch.long)}

    def _validate_rows(self, rows, split):
        if any(row["character_id"] not in self.groups[split] for row in rows):
            raise ValueError("speaker outside split")
        return [row["x"] for row in rows], [self.vocab["stoi"].get(row["y"], 0) for row in rows]

    def iter_tensor_batches(self, split, batch_size=None):
        for rows in self._iter_rows_batches(split, batch_size):
            yield self.encode_tensor_rows(rows, split)

    def iter_batches(self, split, batch_size=None):
        for batch in self.iter_tensor_batches(split, batch_size):
            yield {key: value.tolist() for key, value in batch.items()}

    def contract(self):
        return {"window_size": 5, "dataset": "synthetic_anomaly", "revision": "unit"}

    def validate_checkpoint(self, checkpoint):
        if checkpoint.get("vocab") != self.vocab or checkpoint.get("data_contract") != self.contract():
            raise ValueError("incompatible anomaly fixture checkpoint")
        return True


if AVAILABLE:
    class FixedLogits(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.register_buffer("fixed", logits)

        def forward(self, x):
            logits = self.fixed.expand(len(x), x.shape[1], -1)
            return AEOutput(logits=logits, z=logits.new_zeros((len(x), 32)))


@unittest.skipUnless(AVAILABLE, "Anomaly evaluation requires torch and numpy")
class AnomalyEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / "tmp"
        parent.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="anomaly-unit-", dir=parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pipeline = SyntheticPipeline()
        self.config = {"max_calibration": 16, "max_evaluation": 16, "per_speaker_cap": 8,
                       "batch_size": 4, "read_batch_size": 8, "max_seconds": 30, "alpha": 0.2}

    def checkpoint(self, name="best8.pt", *, model_type="ae", epoch=8, latent_dim=32,
                   latent_activation="identity"):
        torch.manual_seed(42)
        model = Autoencoder(vocab_size=3, sequence_length=5, embedding_dim=32, hidden_size=128,
                            latent_dim=latent_dim, latent_activation=latent_activation)
        payload = dict(model_type=model_type, model_hparams=model.hparams, state_dict=model.state_dict(),
                       vocab=self.pipeline.vocab, data_contract=self.pipeline.contract(), epoch=epoch,
                       checkpoint_role="best_inference_only", training_config={"normal_class": None},
                       optimizer_state_dict={"must_not_restore": True})
        path = self.root / name
        torch.save(payload, path)
        return path

    def run_small(self, name="evaluation", **kwargs):
        checkpoint = kwargs.pop("checkpoint", None) or self.checkpoint()
        return anomalies.run_anomaly_evaluation(self.pipeline, checkpoint, self.root/name,
                                                config=self.config, device="cpu", execute=True, **kwargs)

    def test_uniform_scores_use_nats_per_sequence_and_probability_mse(self):
        model = FixedLogits(torch.zeros(1, 1, 3, dtype=torch.float32))
        x = torch.tensor([[0, 1, 2, 1, 0], [2, 2, 1, 0, 2]], dtype=torch.long)
        scores = anomalies.reconstruction_scores(model, x)
        self.assertEqual(scores["nll_nats_per_sequence"].dtype, torch.float64)
        self.assertTrue(torch.allclose(scores["nll_nats_per_sequence"], torch.full((2,), 5*math.log(3), dtype=torch.float64)))
        self.assertTrue(torch.allclose(scores["brier_probability_mse"], torch.full((2,), 2/9, dtype=torch.float64)))

    def test_scores_are_invariant_to_joint_category_relabelling(self):
        logits = torch.tensor([[[1.0, -1.0, 0.25]]])
        targets = torch.tensor([[0, 1, 2, 1, 0]], dtype=torch.long)
        first = anomalies.reconstruction_scores(FixedLogits(logits), targets)
        permutation = torch.tensor([2, 0, 1])
        inverse = torch.argsort(permutation)
        second = anomalies.reconstruction_scores(FixedLogits(logits[..., permutation]), inverse[targets])
        for key in ("nll_nats_per_sequence", "brier_probability_mse"):
            self.assertTrue(torch.allclose(first[key], second[key], atol=1e-12))

    def test_scoring_preserves_modes_rng_gradients_and_parameters(self):
        torch.manual_seed(50)
        model = Autoencoder(vocab_size=3, sequence_length=5, embedding_dim=3, hidden_size=4, latent_dim=32)
        model.train()
        model.encoder.eval()
        modes = {m: m.training for m in model.modules()}
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
        rng = torch.get_rng_state().clone()
        anomalies.reconstruction_scores(model, torch.tensor([[0, 1, 2, 1, 0]], dtype=torch.long))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(modes, {m: m.training for m in model.modules()})
        for name, p in model.named_parameters():
            self.assertTrue(torch.equal(before[name], p))
            self.assertTrue(torch.equal(gradients[name], p.grad))

    def test_normal_class_is_fitted_exclusively_from_train_y(self):
        # Train x contains more b/a alternations than the next-character labels;
        # the protocol's normal class is specifically y, never x or held-out y.
        result = anomalies.fit_normal_class(self.pipeline, batch_size=3)
        self.assertEqual(result["normal_class"], 1)
        self.assertEqual(result["train_label_counts"], [0, 11, 5])
        self.assertEqual(result["train_examples"], 16)
        self.assertEqual(result["source"], "most_frequent_train_y")
        self.assertEqual(set(self.pipeline.seen_splits), {"train"})

    def test_explicit_normal_class_is_validated_without_fitting_val(self):
        result = anomalies.fit_normal_class(self.pipeline, normal_class=2)
        self.assertEqual(result["normal_class"], 2)
        self.assertEqual(result["source"], "explicit_train_supported_class")
        self.assertNotIn("val", self.pipeline.seen_splits)
        for value in (-1, 3, True):
            with self.subTest(value=value):
                with self.assertRaises((ValueError, TypeError)):
                    anomalies.fit_normal_class(self.pipeline, normal_class=value)

    def test_validation_partitions_are_deterministic_disjoint_speakers(self):
        first = anomalies.split_validation_speakers(self.pipeline, seed=42)
        second = anomalies.split_validation_speakers(self.pipeline, seed=42)
        self.assertEqual(first, second)
        calibration = set(first["calibration_speakers"])
        evaluation = set(first["evaluation_speakers"])
        self.assertTrue(calibration and evaluation)
        self.assertFalse(calibration & evaluation)
        self.assertEqual(calibration | evaluation, self.pipeline.groups["val"])
        self.assertFalse((calibration | evaluation) & self.pipeline.groups["train"])
        self.assertFalse((calibration | evaluation) & self.pipeline.groups["test"])
        self.assertFalse(self.pipeline.seen_splits)

    def test_threshold_uses_finite_sample_upper_order_statistic(self):
        result = anomalies.conformal_threshold(list(range(1, 21)), alpha=0.1)
        self.assertEqual(result["rank"], 19)
        self.assertEqual(result["threshold"], 19)
        self.assertEqual(result["n_normal"], 20)
        extreme = anomalies.conformal_threshold(list(range(1, 20)), alpha=0.05)
        self.assertEqual(extreme["rank"], 19)
        self.assertEqual(extreme["threshold"], 19)

    def test_insufficient_calibration_does_not_emit_infinity(self):
        result = anomalies.conformal_threshold([1.0, 2.0], alpha=0.05)
        self.assertIsNone(result["threshold"])
        json.dumps(result, allow_nan=False)
        for scores in ([1.0, float("nan")], [float("inf")]):
            with self.subTest(scores=scores):
                with self.assertRaises((ValueError, FloatingPointError)):
                    anomalies.conformal_threshold(scores, alpha=0.05)

    def test_tied_auc_and_average_precision_are_grouped(self):
        result = anomalies.detection_metrics([0.1, 0.5, 0.5, 0.9], [0, 1, 0, 1], threshold=0.5)
        self.assertAlmostEqual(result["roc_auc"], 0.875)
        self.assertAlmostEqual(result["average_precision"], 5/6)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 2/3)

    def test_all_tied_scores_give_chance_auc_and_prevalence_ap(self):
        result = anomalies.detection_metrics([1, 1, 1, 1], [0, 1, 0, 1], threshold=1.0)
        self.assertAlmostEqual(result["roc_auc"], 0.5)
        self.assertAlmostEqual(result["average_precision"], 0.5)
        # Threshold is strict score>t, so equality never marks an anomaly.
        self.assertEqual(result["recall"], 0.0)

    def test_single_class_auc_is_explicitly_undefined(self):
        result = anomalies.detection_metrics([1, 2, 3], [0, 0, 0], threshold=2)
        self.assertIsNone(result["roc_auc"])
        json.dumps(result, allow_nan=False)

    def test_empty_normal_calibration_never_invents_a_threshold(self):
        result = anomalies.conformal_threshold([], alpha=0.05)
        self.assertIsNone(result["threshold"])
        self.assertEqual(result["n_normal"], 0)
        json.dumps(result, allow_nan=False)

    def test_nonfinite_logits_are_rejected_before_metric_reporting(self):
        model = FixedLogits(torch.tensor([[[0.0, float("nan"), 0.0]]]))
        with self.assertRaises((ValueError, FloatingPointError)):
            anomalies.reconstruction_scores(model, torch.tensor([[0, 1, 2, 1, 0]], dtype=torch.long))

    def test_configuration_is_bounded_and_never_enables_training(self):
        cfg = anomalies.validate_config(self.config)
        self.assertEqual(cfg["primary_score"], "brier_mse")
        for update in ({"epochs": 30}, {"model": "vae"}, {"max_evaluation": 1000000},
                       {"max_calibration": 1000000}, {"alpha": 0}, {"alpha": 1},
                       {"max_seconds": 601}, {"normal_class": True}, {"evaluate_test": "yes"}):
            with self.subTest(update=update):
                with self.assertRaises((ValueError, TypeError)):
                    anomalies.validate_config(update)

    def test_default_plan_has_no_data_checkpoint_or_optimizer_side_effects(self):
        checkpoint = self.root / "not_loaded.pt"
        with patch.object(torch, "load", side_effect=AssertionError("dry checkpoint load")), \
             patch.object(torch.optim, "Adam", side_effect=AssertionError("dry Adam")), \
             patch.object(self.pipeline, "_iter_rows_batches", side_effect=AssertionError("dry data")):
            result = anomalies.run_anomaly_evaluation(self.pipeline, checkpoint, self.root/"plan", config=self.config)
        self.assertIsInstance(result, dict)
        self.assertFalse((self.root/"plan").exists())

    def test_output_reuse_is_refused(self):
        destination = self.root/"previous"
        destination.mkdir()
        sentinel = destination/"keep.txt"
        sentinel.write_text("previous", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            anomalies.run_anomaly_evaluation(self.pipeline, self.root/"dummy.pt", destination, config=self.config)
        self.assertEqual(sentinel.read_text(), "previous")

    def test_frozen_best8_evaluation_never_constructs_adam_or_reads_test(self):
        checkpoint = self.checkpoint()
        before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        torch_state = torch.get_rng_state().clone()
        python_state = random.getstate()
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("inference restored Adam")), \
             patch.object(torch.optim.Adam, "step", side_effect=AssertionError("inference optimizer step")):
            result = self.run_small(checkpoint=checkpoint)
        self.assertEqual(result["status"], "completed", result.get("reason"))
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertFalse(result["checkpoint_modified"])
        self.assertTrue(result["model_unchanged"])
        self.assertEqual(result["detector_status"], "exploratory_not_normal_only_trained")
        self.assertFalse(result["test_evaluated"])
        self.assertNotIn("test", self.pipeline.seen_splits)
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        rows = list(csv.DictReader((self.root/"evaluation/scores.csv").open(encoding="utf-8", newline="")))
        calibration = [r for r in rows if r["population"] == "calibration"]
        evaluation = [r for r in rows if r["population"] == "evaluation"]
        self.assertTrue(calibration and evaluation)
        self.assertTrue(all(int(r["y_id"]) == result["normal_definition"]["normal_class"] for r in calibration))
        self.assertFalse({r["character_id"] for r in calibration} & {r["character_id"] for r in evaluation})
        self.assertLessEqual(len(calibration), self.config["max_calibration"])
        self.assertLessEqual(len(evaluation), self.config["max_evaluation"])
        self.assertIn("brier_mse", result["thresholds"])
        self.assertEqual(result["plan"]["primary_score"], "brier_mse")
        json.dumps(result, allow_nan=False)

    def test_optional_test_uses_threshold_frozen_before_test_is_read(self):
        self.pipeline.allow_test = True
        original = self.pipeline._iter_rows_batches
        observed = []
        def guarded(split, batch_size=None):
            if split == "test":
                path = self.root/"mixed/thresholds.json"
                self.assertTrue(path.is_file(), "threshold must be committed before test access")
                observed.append(path.read_bytes())
            yield from original(split, batch_size)
        self.pipeline._iter_rows_batches = guarded
        self.config.update(evaluate_test=True, max_test=8)
        result = self.run_small(name="mixed")
        self.assertEqual(result["status"], "completed", result.get("reason"))
        self.assertTrue(result["test_evaluated"])
        self.assertTrue(observed)
        self.assertTrue(all(content == (self.root/"mixed/thresholds.json").read_bytes() for content in observed))
        self.assertIn("test", result["metrics"])
        for name in ("brier_mse", "nll_nats_per_sequence"):
            self.assertEqual(result["metrics"]["test"][name]["threshold"], result["thresholds"][name]["threshold"])
            self.assertEqual(result["metrics"]["evaluation"][name]["threshold"], result["thresholds"][name]["threshold"])

    def test_wrong_epoch_and_other_architectures_are_refused_without_anomaly_scores(self):
        for number, overrides in enumerate(({"epoch": 30}, {"model_type": "vae"},
                                             {"latent_dim": 64}, {"latent_activation": "tanh"})):
            with self.subTest(overrides=overrides):
                checkpoint = self.checkpoint(name=f"wrong_{number}.pt", **overrides)
                before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                result = self.run_small(name=f"refused_{number}", checkpoint=checkpoint)
                self.assertEqual(result["status"], "failed")
                self.assertFalse((self.root/f"refused_{number}/scores.csv").exists())
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)

    def test_declared_normal_only_checkpoint_class_cannot_be_silently_relabelled(self):
        checkpoint = self.checkpoint()
        payload = torch.load(checkpoint, weights_only=True)
        payload["training_config"]["normal_class"] = 2
        torch.save(payload, checkpoint)
        result = self.run_small(name="normal_mismatch", checkpoint=checkpoint)
        self.assertEqual(result["status"], "failed")
        self.assertIn("normal", result["reason"].lower())
        self.assertFalse((self.root/"normal_mismatch/scores.csv").exists())

    def test_missing_training_regime_is_unverified_rather_than_normal_only_or_general_proof(self):
        checkpoint = self.checkpoint()
        payload = torch.load(checkpoint, weights_only=True)
        payload.pop("training_config")
        torch.save(payload, checkpoint)
        result = self.run_small(name="unknown_regime", checkpoint=checkpoint)
        self.assertEqual(result["status"], "completed", result.get("reason"))
        self.assertEqual(result["detector_status"], "exploratory_training_regime_unverified")
        self.assertIn("unverified", result["normal_only_training_evidence"])


if __name__ == "__main__":
    unittest.main()
