"""Synthetic controls for sequence information diagnostics, without training."""
import copy
import json
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from src.autoencoder import Autoencoder, reconstruction_loss
from src.latent_diagnostics import (
    compare_logits,
    evaluate_baseline,
    evaluate_latent_dependence,
    fit_character_baseline,
    summarize_prediction,
)


class _TrainOnlyPipeline:
    """Any attempt to fit on val/test is an immediate test failure."""

    def __init__(self, rows=None):
        self.vocab = {"itos": ["<UNK>", "a", "b"]}
        self.rows = torch.tensor(rows if rows is not None else [
            [0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ], dtype=torch.long)
        self.config = {"window_size": self.rows.shape[1]}
        self.requested_splits = []

    def iter_tensor_batches(self, split, batch_size=4096):
        if split != "train":
            raise AssertionError("Baseline fitting may only read train")
        self.requested_splits.append(split)
        for start in range(0, len(self.rows), batch_size):
            x = self.rows[start:start + batch_size]
            # Deliberately absent from x. Counting y would corrupt the baseline.
            yield {"x": x, "y": torch.full((len(x),), 2, dtype=torch.long)}


class _SequenceCopyDecoder(nn.Module):
    """A positive control whose z contains exactly the reconstructable tokens."""

    def __init__(self, length=6, invariant=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.child = nn.Dropout(0.5)
        self.length = length
        self.invariant = invariant

    def encode(self, x):
        # A read-only diagnostic must preserve RNG even for a stochastic module.
        return x.float() + torch.rand((), device=x.device) * 0

    def decode(self, z):
        tokens = torch.zeros_like(z, dtype=torch.long) if self.invariant else z.long()
        return F.one_hot(tokens, num_classes=3).float() * self.scale


class LatentDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.pipeline = _TrainOnlyPipeline()
        self.baseline = fit_character_baseline(self.pipeline, batch_size=2)
        self.x = torch.tensor([
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
        ], dtype=torch.long)

    def test_baseline_counts_train_x_all_positions_and_unk_only(self):
        result = self.baseline
        self.assertEqual(self.pipeline.requested_splits, ["train"])
        self.assertEqual(result["fitted_split"], "train")
        self.assertEqual(result["scope"], "full_train")
        self.assertEqual(result["counts"], [3, 15, 0])
        self.assertEqual(result["train_examples"], 3)
        self.assertEqual(result["train_tokens"], 18)
        expected = [(count + 0.5) / 19.5 for count in (3, 15, 0)]
        self.assertEqual(result["probabilities"], expected)
        # The partial last batch must receive its actual token count.
        single_batches = fit_character_baseline(self.pipeline, batch_size=1)
        self.assertEqual(result["counts"], single_batches["counts"])
        self.assertEqual(result["position_counts"], single_batches["position_counts"])
        self.assertEqual(result["train_x_sha256"], single_batches["train_x_sha256"])

    def test_prefix_approximation_is_explicit_and_does_not_round_to_batches(self):
        result = fit_character_baseline(self.pipeline, batch_size=3, max_examples=2)
        self.assertEqual(result["scope"], "explicit_train_prefix")
        self.assertEqual(result["train_examples"], 2)
        self.assertEqual(result["train_tokens"], 12)
        self.assertEqual(result["counts"], [3, 9, 0])

    def test_baseline_and_ae_use_identical_units_targets_and_positions(self):
        logits = torch.tensor(self.baseline["probabilities"], dtype=torch.float64).log()
        logits = logits.expand(len(self.x), self.x.shape[1], -1)
        summary = summarize_prediction(logits, self.x)
        baseline = evaluate_baseline(self.baseline, self.x)["marginal"]
        expected = reconstruction_loss(logits, self.x).item()
        self.assertAlmostEqual(summary["nats_per_sequence"], expected, places=12)
        self.assertAlmostEqual(baseline["nats_per_sequence"], expected, places=12)
        self.assertAlmostEqual(baseline["nats_per_character"] * 6, expected, places=12)
        self.assertEqual(len(baseline["per_position_nats_per_character"]), 6)
        self.assertAlmostEqual(sum(baseline["per_position_nats_per_character"]), expected, places=12)
        manual = sum(-math.log(self.baseline["probabilities"][int(token)])
                     for token in self.x.flatten()) / len(self.x)
        self.assertAlmostEqual(manual, expected, places=12)

    def test_unseen_category_smoothing_and_unsmoothed_support_are_explicit(self):
        smooth = evaluate_baseline(self.baseline, self.x)["marginal"]
        self.assertTrue(math.isfinite(smooth["nats_per_sequence"]))
        self.assertEqual(smooth["unsupported_target_tokens"], 0)
        unsmoothed = fit_character_baseline(self.pipeline, alpha=0)
        result = evaluate_baseline(unsmoothed, self.x)
        self.assertEqual(result["marginal"]["status"], "undefined_unsmoothed_support")
        self.assertIsNone(result["marginal"]["nats_per_sequence"])
        self.assertEqual(result["marginal"]["unsupported_target_tokens"], 6)
        json.dumps(result, allow_nan=False)

    def test_train_position_baseline_captures_position_only_patterns(self):
        pipeline = _TrainOnlyPipeline([[0, 0, 1, 1, 2, 2]] * 3)
        baseline = fit_character_baseline(pipeline)
        result = evaluate_baseline(baseline, pipeline.rows)
        self.assertLess(result["position_marginal"]["nats_per_sequence"],
                        result["marginal"]["nats_per_sequence"])
        positions = [i for region in result["marginal"]["regions"].values()
                     for i in region["positions_0based"]]
        self.assertEqual(positions, list(range(6)))

    def test_latent_invariant_decoder_has_zero_intervention_and_diversity(self):
        result = evaluate_latent_dependence(_SequenceCopyDecoder(invariant=True), self.x, self.baseline)
        for comparison in result["comparisons"].values():
            self.assertEqual(comparison["delta_nats_per_sequence"]["mean"], 0.0)
            self.assertEqual(comparison["centered_logits_rms_change"], 0.0)
            self.assertEqual(comparison["probability_tv_mean"], 0.0)
            self.assertEqual(comparison["top1_change_fraction"], 0.0)
        self.assertEqual(result["historical_sensitivity"]["relative_raw_logits_l2_reverse"], 0.0)
        self.assertEqual(result["modes"]["original"]["between_examples"]["probability_variance_mean"], 0.0)
        # z still differs across examples, separating encoder spread from use by decoder.
        self.assertGreater(result["latent"]["between_example_std_rms"], 0)

    def test_informative_decoder_has_positive_swap_damage_and_predictions_change(self):
        result = evaluate_latent_dependence(_SequenceCopyDecoder(), self.x, self.baseline)
        comparison = result["comparisons"]["swapped_minus_original"]
        self.assertLess(result["modes"]["original"]["nats_per_sequence"], 0.01)
        self.assertGreater(comparison["delta_nats_per_sequence"]["mean"], 40)
        self.assertGreater(comparison["probability_tv_mean"], 0.99)
        self.assertEqual(comparison["top1_change_fraction"], 1.0)
        self.assertEqual(result["permutation_fixed_points"], 0)
        self.assertEqual(result["swapped_source_target_exact_sequence_agreement"], 0.0)
        self.assertEqual(result["historical_sensitivity"]["reverse_fixed_points"], 1)
        self.assertEqual(result["metrics_dtype"], "float64")
        json.dumps(result, allow_nan=False)

    def test_evaluation_preserves_rng_weights_existing_gradients_and_mixed_modes(self):
        model = _SequenceCopyDecoder()
        model.train()
        model.child.eval()
        model.scale.grad = torch.tensor(12.5)
        state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        gradient = model.scale.grad.clone()
        rng = torch.get_rng_state().clone()
        evaluate_latent_dependence(model, self.x, self.baseline)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        self.assertTrue(torch.equal(gradient, model.scale.grad))
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, state[name]), name)

    def test_real_ae_modes_both_architectures_are_supported_without_gradients(self):
        for activation in ("identity", "tanh"):
            with self.subTest(activation=activation):
                model = Autoencoder(vocab_size=3, sequence_length=6, embedding_dim=3,
                                    hidden_size=4, latent_dim=2, latent_activation=activation)
                before = copy.deepcopy(model.state_dict())
                result = evaluate_latent_dependence(model, self.x, self.baseline)
                self.assertEqual(set(result["modes"]), {"original", "swapped", "zero"})
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, before[name]), name)

    def test_common_logit_offset_changes_raw_metric_without_information_change(self):
        original = torch.tensor([[[1., 3., -2.]], [[2., 0., 1.]]], dtype=torch.float64)
        targets = torch.tensor([[1], [0]])
        offset_only = compare_logits(original, original + 50, targets)
        self.assertGreater(offset_only["relative_raw_logits_l2_change"], 1)
        self.assertAlmostEqual(offset_only["centered_logits_rms_change"], 0, places=12)
        self.assertAlmostEqual(offset_only["probability_tv_mean"], 0, places=12)
        self.assertAlmostEqual(offset_only["delta_nats_per_sequence"]["mean"], 0, places=12)
        altered = original.flip(0)
        reference = compare_logits(original, altered, targets)
        shifted = compare_logits(original + 1000, altered + 1000, targets)
        self.assertLess(shifted["relative_raw_logits_l2_change"], reference["relative_raw_logits_l2_change"] / 100)
        self.assertAlmostEqual(shifted["centered_logits_rms_change"], reference["centered_logits_rms_change"], places=12)
        self.assertAlmostEqual(shifted["probability_tv_mean"], reference["probability_tv_mean"], places=12)

    def test_small_relative_raw_change_can_hide_a_reconstruction_decision(self):
        # Huge shared logits make the historical denominator misleading.
        original = torch.tensor([[[1000., 999.]]], dtype=torch.float64)
        altered = torch.tensor([[[999., 1000.]]], dtype=torch.float64)
        comparison = compare_logits(original, altered, torch.tensor([[0]]))
        self.assertLess(comparison["relative_raw_logits_l2_change"], 0.002)
        self.assertGreater(comparison["probability_tv_mean"], 0.45)
        self.assertGreater(comparison["delta_nats_per_sequence"]["mean"], 0.99)
        self.assertEqual(comparison["top1_change_fraction"], 1.0)

    def test_top1_agreement_does_not_imply_equal_distributions(self):
        comparison = compare_logits(torch.tensor([[[5., 1., 0.]]]),
                                    torch.tensor([[[5., 3., 0.]]]), torch.tensor([[1]]))
        self.assertEqual(comparison["top1_change_fraction"], 0)
        self.assertGreater(comparison["probability_tv_mean"], 0.05)
        self.assertGreater(comparison["probability_js_nats_mean"], 0)
        self.assertLess(comparison["delta_nats_per_sequence"]["mean"], 0)

    def test_position_diagnostics_localize_late_loss_of_information(self):
        original = torch.zeros(2, 6, 3, dtype=torch.float64)
        altered = original.clone()
        altered[:, 4:, 1] = 5
        comparison = compare_logits(original, altered, torch.zeros(2, 6, dtype=torch.long))
        self.assertEqual(comparison["per_position"]["delta_nats_per_character"][:4], [0] * 4)
        self.assertEqual(comparison["regions"]["initial"]["delta_nats_per_character"], 0)
        self.assertEqual(comparison["regions"]["middle"]["delta_nats_per_character"], 0)
        self.assertGreater(comparison["regions"]["final"]["delta_nats_per_character"], 3)
        self.assertAlmostEqual(sum(comparison["per_position"]["delta_nats_per_character"]),
                               comparison["delta_nats_per_sequence"]["mean"], places=12)

    def test_permutations_must_be_complete_derangements_and_batches_nontrivial(self):
        model = _SequenceCopyDecoder()
        for invalid in ([0, 1, 2], [1, 1, 0], [1, 2], [1., 2., 0.], [-1, 0, 1]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluate_latent_dependence(model, self.x, self.baseline, permutation=invalid)
        with self.assertRaises(ValueError):
            evaluate_latent_dependence(model, self.x[:1], self.baseline)
        result = evaluate_latent_dependence(model, self.x, self.baseline, permutation=[1, 2, 0])
        self.assertEqual(result["permutation"], [1, 2, 0])

    def test_nonfinite_logits_latent_and_invalid_baseline_fail_closed(self):
        logits = torch.zeros(2, 6, 3)
        for invalid in (float("nan"), float("inf")):
            broken = logits.clone()
            broken[0, 0, 0] = invalid
            with self.assertRaises(FloatingPointError):
                summarize_prediction(broken, self.x[:2])
        baseline = copy.deepcopy(self.baseline)
        baseline["probabilities"][0] = float("nan")
        with self.assertRaises(FloatingPointError):
            evaluate_baseline(baseline, self.x)
        model = _SequenceCopyDecoder()
        model.scale.data.fill_(float("inf"))
        with self.assertRaises(FloatingPointError):
            evaluate_latent_dependence(model, self.x, self.baseline)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
