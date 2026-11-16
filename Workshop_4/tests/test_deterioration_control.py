"""Phase 4 controls, immutable recovery and tiny synthetic CPU windows only.

The fixtures never open Shakespeare or production checkpoints. Training tests
use two to four updates with a small GRU while retaining the original z32 AE.
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
    from src import deterioration_control as control
    from src.autoencoder import AEOutput, Autoencoder
    from src.train_ae import NonFiniteGradientError


class SyntheticPipeline:
    def __init__(self):
        self.vocab = {"itos": ["<UNK>", "a", "b"],
                      "stoi": {"<UNK>": 0, "a": 1, "b": 2},
                      "size": 3, "unk_index": 0}
        self.config = {"window_size": 5, "batch_size": 2}
        self.splits = {"counts": {"train": {"n_rows": 1024}, "val": {"n_rows": 4}}}
        self.seen_splits = []
        patterns = [[0, 1, 2, 1, 2], [2, 2, 1, 1, 1], [1, 2, 1, 0, 1], [2, 1, 2, 1, 2]]
        self.values = {"train": patterns * 256, "val": patterns}

    def iter_tensor_batches(self, split, batch_size=None):
        if split not in ("train", "val"):
            raise AssertionError("Phase 4 must never read test")
        self.seen_splits.append(split)
        for start in range(0, len(self.values[split]), batch_size or 2):
            x = torch.tensor(self.values[split][start:start+(batch_size or 2)], dtype=torch.long)
            yield {"x": x, "y": torch.zeros(len(x), dtype=torch.long)}

    def contract(self):
        return {"window_size": 5, "dataset": "synthetic_phase4"}

    def validate_checkpoint(self, value):
        if value["vocab"] != self.vocab or value["data_contract"] != self.contract():
            raise ValueError("incompatible synthetic checkpoint")
        return True


@unittest.skipUnless(AVAILABLE, "Phase 4 control tests require torch and numpy")
class DeteriorationControlTests(unittest.TestCase):
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
        self.temporary = tempfile.TemporaryDirectory(prefix="phase4-unit-", dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pipeline = SyntheticPipeline()
        self.config = {"max_updates": 2, "evaluate_every": 1, "batch_size": 2,
                       "val_examples": 4, "embedding_dim": 3, "hidden_size": 4,
                       "max_seconds": 30}

    def run_small(self, name="window", **kwargs):
        return control.run_controlled(self.pipeline, self.root / name, config=self.config,
                                      device="cpu", execute=True, **kwargs)

    def historical_checkpoint(self, name="best8.pt", **overrides):
        torch.manual_seed(42)
        model = Autoencoder(vocab_size=3, sequence_length=5, embedding_dim=3,
                            hidden_size=4, latent_dim=32)
        value = {"model_type": "ae", "model_hparams": model.hparams,
                 "state_dict": model.state_dict(), "vocab": self.pipeline.vocab,
                 "data_contract": self.pipeline.contract(), "epoch": 8,
                 "checkpoint_role": "best_inference_only",
                 "optimizer_state_dict": {"must_not_restore": True}, **overrides}
        destination = self.root / name
        torch.save(value, destination)
        return destination

    def state(self, reconstruction=100.0, utility=50.0):
        return {"best_reconstruction": reconstruction, "best_swap_damage": utility,
                "initial_reconstruction": reconstruction, "initial_swap_damage": utility,
                "ce_bad_count": 0, "swap_bad_count": 0}

    def test_default_is_a_plan_without_data_optimizer_or_outputs(self):
        output = self.root / "dry"
        with patch.object(self.pipeline, "iter_tensor_batches", side_effect=AssertionError("dry data")), \
             patch.object(torch.optim, "Adam", side_effect=AssertionError("dry Adam")):
            plan = control.run_controlled(self.pipeline, output, config=self.config)
        self.assertFalse(output.exists())
        self.assertFalse(self.pipeline.seen_splits)
        self.assertIsInstance(plan, dict)

    def test_existing_output_is_refused_and_preserved(self):
        output = self.root / "historical"
        output.mkdir()
        sentinel = output / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        for execute in (False, True):
            with self.assertRaises(FileExistsError):
                control.run_controlled(self.pipeline, output, config=self.config, execute=execute)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_caps_and_architecture_cannot_start_other_experiments(self):
        for key, invalid in (("max_updates", 257), ("max_updates", True), ("latent_dim", 64),
                             ("latent_dim", 31), ("num_layers", 2), ("dropout", 0.2),
                             ("learning_rate", float("nan")), ("grad_clip", 0),
                             ("latent_activation", "tanh"), ("model", "vae"), ("epochs", 30)):
            with self.subTest(key=key, invalid=invalid):
                with self.assertRaises(ValueError):
                    control.validate_config({key: invalid})

    def test_severe_reconstruction_damage_stops_immediately(self):
        _, decision = control.control_decision({"reconstruction": 201.0, "swap_damage": 50.0},
                                                self.state(), control.validate_config())
        self.assertTrue(decision["stop"])
        self.assertTrue(decision["reasons"])

    def test_sustained_ce_damage_requires_patience_and_resets_on_recovery(self):
        config = control.validate_config()
        state, first = control.control_decision({"reconstruction": 125.0, "swap_damage": 50.0},
                                                self.state(), config)
        self.assertFalse(first["stop"])
        _, second = control.control_decision({"reconstruction": 125.0, "swap_damage": 50.0}, state, config)
        self.assertTrue(second["stop"])
        recovered, clear = control.control_decision({"reconstruction": 100.0, "swap_damage": 50.0}, state, config)
        self.assertFalse(clear["stop"])
        _, new_first = control.control_decision({"reconstruction": 125.0, "swap_damage": 50.0}, recovered, config)
        self.assertFalse(new_first["stop"])

    def test_loss_of_learned_swap_utility_stops_and_blocks_best_replacement(self):
        config = control.validate_config()
        state, first = control.control_decision({"reconstruction": 90.0, "swap_damage": 20.0},
                                                self.state(), config)
        self.assertFalse(first["stop"])
        self.assertFalse(first["eligible_for_best"])
        _, second = control.control_decision({"reconstruction": 90.0, "swap_damage": 20.0}, state, config)
        self.assertTrue(second["stop"])

    def test_random_initial_latent_response_does_not_arm_utility_stop(self):
        config = control.validate_config()
        state = self.state(100.0, 0.01)
        for _ in range(5):
            state, decision = control.control_decision({"reconstruction": 99.0, "swap_damage": 0.0}, state, config)
            self.assertFalse(decision["stop"])

    def test_swap_guard_arms_only_after_meaningful_utility_is_observed(self):
        config = control.validate_config()
        state, _ = control.control_decision({"reconstruction": 90.0, "swap_damage": 10.0},
                                            self.state(100.0, 0.01), config)
        state, first = control.control_decision({"reconstruction": 90.0, "swap_damage": 1.0}, state, config)
        self.assertTrue(first["swap_guard_applicable"])
        self.assertFalse(first["stop"])
        _, second = control.control_decision({"reconstruction": 90.0, "swap_damage": 1.0}, state, config)
        self.assertTrue(second["stop"])

    def test_weights_only_never_restores_historical_adam_and_preserves_source(self):
        source = self.historical_checkpoint()
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        with patch.object(torch.optim.Adam, "load_state_dict", side_effect=AssertionError("historical Adam restored")):
            result = self.run_small("weights", source_checkpoint=source, initialization="weights_only")
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(set(self.pipeline.seen_splits), {"train", "val"})

    def test_inference_best_is_never_used_for_optimizer_continuation(self):
        source = self.historical_checkpoint()
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        for initialization in ("continue", "reset_adam"):
            with self.subTest(initialization=initialization):
                with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("invalid continuation step")):
                    try:
                        result = self.run_small("wrong_"+initialization, source_checkpoint=source,
                                                initialization=initialization)
                    except ValueError:
                        result = {"status": "failed"}
                self.assertNotEqual(result["status"], "completed")
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_invalid_loss_and_gradients_never_reach_adam(self):
        original = Autoencoder.forward
        def nonfinite(model, x):
            output = original(model, x)
            if model.training and torch.is_grad_enabled():
                return AEOutput(output.logits * float("nan"), output.z)
            return output
        with patch.object(Autoencoder, "forward", new=nonfinite), \
             patch.object(torch.optim.Adam, "step", side_effect=AssertionError("nonfinite Adam step")):
            result = self.run_small("loss_invalid")
        self.assertNotEqual(result["status"], "completed")
        with patch.object(control, "stable_clip_grad_norm_", side_effect=NonFiniteGradientError("invalid gradient")), \
             patch.object(torch.optim.Adam, "step", side_effect=AssertionError("invalid gradient Adam step")):
            result = self.run_small("gradient_invalid")
        self.assertNotEqual(result["status"], "completed")

    def test_zero_grad_prevents_accumulation_and_fp64_clipping_is_called(self):
        original = Autoencoder.forward
        checked = []
        def record(model, x):
            if model.training and torch.is_grad_enabled():
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                checked.append(len(x))
            return original(model, x)
        with patch.object(Autoencoder, "forward", new=record), \
             patch.object(control, "stable_clip_grad_norm_", wraps=control.stable_clip_grad_norm_) as clipping:
            result = self.run_small("gradient_clear")
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(checked, [2, 2])
        self.assertEqual(clipping.call_count, 2)

    def test_periodic_checkpoints_are_immutable_and_deterioration_preserves_healthy_best(self):
        self.config["max_updates"] = 4
        original_diagnostic = control.evaluate_latent_dependence
        original_save = control._save_checkpoint
        measured = 0
        hashes = {}
        # These injected metrics exercise policy, not reconstruction accuracy.
        trajectory = ((100.0, 50.0), (90.0, 50.0), (135.0, 1.0), (135.0, 1.0))
        def synthetic_metrics(*args, **kwargs):
            nonlocal measured
            value = original_diagnostic(*args, **kwargs)
            ce, utility = trajectory[min(measured, len(trajectory)-1)]
            measured += 1
            value["modes"]["original"]["nats_per_sequence"] = ce
            value["comparisons"]["swapped_minus_original"]["delta_nats_per_sequence"]["mean"] = utility
            return value
        def capture_saved(path, value):
            original_save(path, value)
            hashes[Path(path)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with patch.object(control, "evaluate_latent_dependence", side_effect=synthetic_metrics), \
             patch.object(control, "_save_checkpoint", side_effect=capture_saved):
            result = self.run_small("deterioration")
        self.assertEqual(result["status"], "stopped", result)
        self.assertEqual(result["updates"], 3)
        self.assertEqual(result["best_prefix"]["cumulative_update"], 1)
        self.assertEqual(result["best_prefix"]["reconstruction"], 90.0)
        self.assertEqual(len(hashes), 4)
        for path, expected in hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        best_path = Path(result["best_prefix"]["path"])
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        self.assertEqual(best["cursor_updates"], 1)
        with self.assertRaises(FileExistsError):
            control._save_checkpoint(best_path, {"never": "replace"})
        stopped_source = self.root / "deterioration" / result["checkpoints"][-1]
        before = hashlib.sha256(stopped_source.read_bytes()).hexdigest()
        for initialization in ("continue", "reset_adam"):
            with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("bypassed deteriorated stop")):
                rejected = self.run_small("stopped_"+initialization, source_checkpoint=stopped_source,
                                          initialization=initialization)
            self.assertEqual(rejected["status"], "failed")
            self.assertIn("stop", rejected["reason"].lower())
        self.assertEqual(hashlib.sha256(stopped_source.read_bytes()).hexdigest(), before)

    def test_failed_candidate_checkpoint_write_cannot_promote_or_destroy_best(self):
        original_diagnostic = control.evaluate_latent_dependence
        original_save = control._save_checkpoint
        measured = saved = 0
        def improving(*args, **kwargs):
            nonlocal measured
            value = original_diagnostic(*args, **kwargs)
            value["modes"]["original"]["nats_per_sequence"] = 100.0 if measured == 0 else 90.0
            value["comparisons"]["swapped_minus_original"]["delta_nats_per_sequence"]["mean"] = 50.0
            measured += 1
            return value
        def disk_failure(path, value):
            nonlocal saved
            saved += 1
            if saved == 2:
                raise OSError("injected checkpoint write failure")
            original_save(path, value)
        with patch.object(control, "evaluate_latent_dependence", side_effect=improving), \
             patch.object(control, "_save_checkpoint", side_effect=disk_failure):
            result = self.run_small("write_failure")
        self.assertEqual(result["status"], "stopped", result)
        self.assertEqual(result["best_prefix"]["cumulative_update"], 0)
        pointer = json.loads((self.root / "write_failure/best_prefix.json").read_text())
        self.assertEqual(pointer, result["best_prefix"])
        self.assertTrue(Path(pointer["path"]).is_file())

    def test_continuation_refuses_a_missing_protected_best_reference(self):
        original_diagnostic = control.evaluate_latent_dependence
        measured = 0
        trajectory = (100.0, 90.0, 95.0)
        def known_best(*args, **kwargs):
            nonlocal measured
            value = original_diagnostic(*args, **kwargs)
            value["modes"]["original"]["nats_per_sequence"] = trajectory[min(measured, 2)]
            value["comparisons"]["swapped_minus_original"]["delta_nats_per_sequence"]["mean"] = 50.0
            measured += 1
            return value
        with patch.object(control, "evaluate_latent_dependence", side_effect=known_best):
            origin = self.run_small("best_reference_origin")
        self.assertEqual(origin["status"], "completed", origin)
        source = self.root / "best_reference_origin" / origin["checkpoints"][-1]
        best = Path(origin["best_prefix"]["path"])
        self.assertNotEqual(source, best)
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        # Delete only this test's generated temporary best, simulating a user
        # copying the periodic file without its referenced protected checkpoint.
        self.assertTrue(best.is_relative_to(self.root))
        best.unlink()
        for initialization in ("continue", "reset_adam"):
            with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("missing-best Adam step")):
                rejected = self.run_small("missing_best_"+initialization, source_checkpoint=source,
                                          initialization=initialization)
            self.assertEqual(rejected["status"], "failed", rejected)
            self.assertIn("best", rejected["reason"].lower())
            self.assertIn("missing", rejected["reason"].lower())
            self.assertEqual(rejected["updates"], 0)
            self.assertFalse(rejected["checkpoints"])
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_exact_continuation_preserves_adam_rng_and_order(self):
        self.config["max_updates"] = 1
        first = self.run_small("first")
        self.assertEqual(first["status"], "completed", first)
        source = self.root / "first" / first["checkpoints"][-1]
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        continued = self.run_small("continued", source_checkpoint=source, initialization="continue")
        self.config["max_updates"] = 2
        uninterrupted = self.run_small("uninterrupted")
        self.assertEqual(continued["status"], "completed", continued)
        self.assertEqual(uninterrupted["status"], "completed", uninterrupted)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        left = torch.load(self.root / "continued" / continued["checkpoints"][-1], weights_only=True)
        right = torch.load(self.root / "uninterrupted" / uninterrupted["checkpoints"][-1], weights_only=True)
        self.assertEqual(left["cursor_updates"], 2)
        self.assertEqual(left["diagnostic_data_state"], right["diagnostic_data_state"])
        self.assertEqual(left["control_state"], right["control_state"])
        self.assertTrue(torch.equal(left["rng_state"]["torch"], right["rng_state"]["torch"]))
        for name in left["state_dict"]:
            self.assertTrue(torch.equal(left["state_dict"][name], right["state_dict"][name]), name)
        left_adam, right_adam = left["optimizer_state_dict"], right["optimizer_state_dict"]
        self.assertEqual(left_adam["param_groups"], right_adam["param_groups"])
        for parameter, values in left_adam["state"].items():
            for name, value in values.items():
                self.assertTrue(torch.equal(value, right_adam["state"][parameter][name]), name)

    def test_reset_adam_is_a_matched_counterfactual_with_identical_next_batch(self):
        self.config["max_updates"] = 1
        first = self.run_small("optimizer_origin")
        self.assertEqual(first["status"], "completed", first)
        source = self.root / "optimizer_origin" / first["checkpoints"][-1]
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        original = Autoencoder.forward
        batches = {}
        mode = None
        def record(model, x):
            if model.training and torch.is_grad_enabled():
                batches.setdefault(mode, []).append(x.detach().clone())
            return original(model, x)
        results = {}
        for mode in ("continue", "reset_adam"):
            with patch.object(Autoencoder, "forward", new=record):
                results[mode] = self.run_small("optimizer_"+mode, source_checkpoint=source, initialization=mode)
            self.assertEqual(results[mode]["status"], "completed", results[mode])
        self.assertEqual(len(batches["continue"]), 1)
        self.assertTrue(torch.equal(batches["continue"][0], batches["reset_adam"][0]))
        initial, final = {}, {}
        for mode, result in results.items():
            initial[mode] = torch.load(self.root / ("optimizer_"+mode) / result["checkpoints"][0], weights_only=True)
            final[mode] = torch.load(self.root / ("optimizer_"+mode) / result["checkpoints"][-1], weights_only=True)
        left, right = initial["continue"], initial["reset_adam"]
        self.assertEqual(left["cursor_updates"], right["cursor_updates"])
        self.assertEqual(left["diagnostic_data_state"], right["diagnostic_data_state"])
        self.assertEqual(left["control_state"], right["control_state"])
        self.assertTrue(torch.equal(left["rng_state"]["torch"], right["rng_state"]["torch"]))
        for name in left["state_dict"]:
            self.assertTrue(torch.equal(left["state_dict"][name], right["state_dict"][name]), name)
        self.assertTrue(left["optimizer_state_dict"]["state"])
        self.assertEqual(right["optimizer_state_dict"]["state"], {})
        for mode, expected_step in (("continue", 2), ("reset_adam", 1)):
            self.assertTrue(all(float(v["step"]) == expected_step
                                for v in final[mode]["optimizer_state_dict"]["state"].values()))
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_post_adam_corruption_is_reported_as_post_update_and_keeps_earlier_checkpoint(self):
        original = torch.optim.Adam.step
        def corrupt_after_step(optimizer, *args, **kwargs):
            value = original(optimizer, *args, **kwargs)
            next(iter(optimizer.state.values()))["exp_avg"].fill_(float("inf"))
            return value
        with patch.object(torch.optim.Adam, "step", new=corrupt_after_step):
            result = self.run_small("post_adam_invalid")
        self.assertEqual(result["status"], "stopped", result)
        self.assertEqual(result["updates"], 1)
        self.assertEqual(result["failure_stage"], "after_update")
        self.assertFalse(result["stopped_before_unsafe_step"])
        self.assertEqual(result["best_prefix"]["cumulative_update"], 0)
        self.assertTrue(Path(result["best_prefix"]["path"]).is_file())

    def test_full_validation_is_separate_and_weights_partial_batches(self):
        torch.manual_seed(42)
        model = Autoencoder(vocab_size=3, sequence_length=5, embedding_dim=3,
                            hidden_size=4, latent_dim=32)
        model.train()
        x = torch.tensor(self.pipeline.values["val"], dtype=torch.long)
        from src.autoencoder import reconstruction_loss
        with torch.no_grad():
            expected = reconstruction_loss(model(x).logits.double(), x).item()
        result = control.evaluate_full_validation(model, self.pipeline, batch_size=3)
        self.assertEqual(result["scope"], "full_validation")
        self.assertEqual(result["examples"], 4)
        self.assertAlmostEqual(result["nats_per_sequence"], expected, places=6)
        self.assertAlmostEqual(result["nats_per_character"], expected / 5, places=6)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
