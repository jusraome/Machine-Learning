"""Checkpoint inspection uses synthetic states; no optimizer step or real data."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec("torch") is not None
if AVAILABLE:
    import torch
    from src.autoencoder import Autoencoder
    from src import checkpoint_fidelity as fidelity
    from src.train_ae import _rng_state


@unittest.skipUnless(AVAILABLE, "Checkpoint inspection requires torch")
class CheckpointFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "tmp"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="checkpoint-fidelity-", dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = Autoencoder(vocab_size=3, sequence_length=5, embedding_dim=3,
                                 hidden_size=4, latent_dim=32)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        state = optimizer.state_dict()
        for identifier, parameter in zip(state["param_groups"][0]["params"], self.model.parameters()):
            state["state"][identifier] = {"exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.ones_like(parameter) * .01, "step": torch.tensor(8.)}
        cfg = dict(model="ae", embedding_dim=3, hidden_size=4, latent_dim=32, num_layers=1,
                   dropout=0., batch_size=2, seed=42, learning_rate=.001,
                   shuffle_buffer_batches=16, grad_clip=5., normal_class=None, epochs=30)
        self.checkpoint = dict(model_type="ae",model_hparams=self.model.hparams,
            state_dict=copy.deepcopy(self.model.state_dict()), optimizer_state_dict=state,
            epoch=8, checkpoint_role="best_inference_only", training_config=cfg,
            rng_state=_rng_state(), data_manifest={"files":{}}, data_contract={"synthetic":True},
            history=[dict(epoch=e,train_accepted_batches=1) for e in range(1,9)],
            runtime={"tensor_loader":False}, gradient_clipping={"algorithm":"global_l2_fp64_v1"})

    def inspect(self, checkpoint=None, **kwargs):
        return fidelity.inspect_checkpoint(self.checkpoint if checkpoint is None else checkpoint, **kwargs)

    def test_complete_adam_roundtrip_without_any_training_is_not_fidelity_proof(self):
        with patch.object(torch.optim.Adam,"step",side_effect=AssertionError("No training")):
            report = self.inspect()
        self.assertTrue(report["adam"]["structurally_complete"])
        self.assertTrue(report["disposable_restore"]["weights_verified"])
        self.assertTrue(report["disposable_restore"]["adam_verified"])
        self.assertEqual(report["temporal_alignment"]["status"],"unproven")
        self.assertFalse(report["historical_faithful_continuation_eligible"])
        self.assertFalse(report["disposable_restore"]["optimizer_step_called"])

    def test_recovered_best_detects_later_history_and_later_adam_steps(self):
        self.checkpoint["history"] = [dict(epoch=e,train_accepted_batches=1) for e in range(1,31)]
        for entry in self.checkpoint["optimizer_state_dict"]["state"].values():
            entry["step"] = torch.tensor(30.)
        report=self.inspect()
        self.assertEqual(report["temporal_alignment"]["status"],"inconsistent")
        self.assertEqual(report["temporal_alignment"]["weight_epoch"],8)
        self.assertEqual(report["temporal_alignment"]["final_history_epoch"],30)

    def test_missing_second_moment_is_incomplete_and_not_restored(self):
        self.checkpoint["optimizer_state_dict"]["state"][0].pop("exp_avg_sq")
        report=self.inspect()
        self.assertFalse(report["adam"]["structurally_complete"])
        self.assertFalse(report["disposable_restore"]["attempted"])

    def test_wrong_shape_negative_and_nonfinite_moments_are_detected(self):
        for invalid in (torch.zeros(1), torch.full_like(self.model.embedding.weight,-1),
                        torch.full_like(self.model.embedding.weight,float("nan"))):
            with self.subTest(invalid=str(invalid.shape)):
                fixture=copy.deepcopy(self.checkpoint)
                fixture["optimizer_state_dict"]["state"][0]["exp_avg_sq"]=invalid
                report=self.inspect(fixture)
                self.assertFalse(report["adam"]["structurally_complete"])

    def test_missing_parameter_state_is_detected(self):
        self.checkpoint["optimizer_state_dict"]["state"].pop(0)
        self.assertFalse(self.inspect()["adam"]["structurally_complete"])

    def test_fractional_step_and_history_step_disagreement_are_distinct(self):
        fixture=copy.deepcopy(self.checkpoint)
        fixture["optimizer_state_dict"]["state"][0]["step"]=torch.tensor(8.5)
        self.assertFalse(self.inspect(fixture)["adam"]["structurally_complete"])
        self.checkpoint["optimizer_state_dict"]["state"][0]["step"]=torch.tensor(9.)
        report=self.inspect()
        self.assertTrue(report["adam"]["structurally_complete"])
        self.assertEqual(report["temporal_alignment"]["status"],"inconsistent")

    def test_compare_later_checkpoint_identical_adam_rng_different_weights(self):
        last=copy.deepcopy(self.checkpoint)
        last["epoch"]=30
        last["state_dict"]["embedding.weight"].add_(1)
        report=self.inspect(comparison_checkpoint=last)
        comparison=report["temporal_alignment"]["comparison"]
        self.assertTrue(comparison["same_adam"])
        self.assertTrue(comparison["same_rng"])
        self.assertFalse(comparison["same_weights"])
        self.assertEqual(report["temporal_alignment"]["status"],"inconsistent")

    def test_group_learning_rate_and_external_metadata_mismatch_are_reported(self):
        self.checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]=.01
        report=self.inspect(metadata={"training_config":{"model":"something_else"}})
        self.assertGreaterEqual(len(report["configuration"]["issues"]),2)

    def test_missing_rng_is_not_equivalent_to_a_restorable_training_state(self):
        self.checkpoint.pop("rng_state")
        report=self.inspect()
        self.assertFalse(report["rng"]["complete"])
        self.assertFalse(report["epoch_boundary_state_restorable"])

    def test_inspection_preserves_process_rng_flags_and_checkpoint_values(self):
        torch_rng=torch.get_rng_state().clone()
        python_rng=random.getstate()
        flags=(torch.are_deterministic_algorithms_enabled(),torch.backends.cudnn.benchmark)
        before=fidelity.value_fingerprint(self.checkpoint)
        self.inspect()
        self.assertTrue(torch.equal(torch_rng,torch.get_rng_state()))
        self.assertEqual(python_rng,random.getstate())
        self.assertEqual(flags,(torch.are_deterministic_algorithms_enabled(),torch.backends.cudnn.benchmark))
        self.assertEqual(before,fidelity.value_fingerprint(self.checkpoint))

    def test_source_hash_preserved_and_only_readonly_report_written(self):
        source=self.root/"best.pt"
        torch.save(self.checkpoint,source)
        before=hashlib.sha256(source.read_bytes()).hexdigest()
        report=fidelity.write_checkpoint_audit(self.root/"new",checkpoint=source,execute=True)
        self.assertEqual(before,hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertTrue(report["sources_unchanged"])
        self.assertTrue((self.root/"new/report.json").is_file())
        with self.assertRaises(FileExistsError):
            fidelity.write_checkpoint_audit(self.root/"new",checkpoint=source,execute=True)

    def test_dry_plan_does_not_read_checkpoint_or_construct_adam(self):
        with patch.object(torch,"load",side_effect=AssertionError("No file read")), \
             patch.object(torch.optim,"Adam",side_effect=AssertionError("No optimizer")):
            report=fidelity.write_checkpoint_audit(self.root/"dry",checkpoint="missing.pt")
        self.assertFalse(report["execute"])
        self.assertFalse((self.root/"dry").exists())

    def test_missing_actual_checkpoint_does_not_invent_adam_inventory(self):
        report=fidelity.audit_checkpoint(self.root/"unavailable_drive_best.pt")
        self.assertEqual(report["availability"],"missing")
        self.assertFalse(report["inspected"])
        self.assertNotIn("adam",report)
        self.assertFalse(report["historical_faithful_continuation_eligible"])

    def test_consistent_training_envelope_can_restore_but_data_backend_fidelity_still_unverified(self):
        from src.runtime import DEFAULT_RUNTIME
        from src.train_ae import GRADIENT_CLIPPING, STABILITY_DEFAULTS
        self.checkpoint.update(checkpoint_role="last_resumable", runtime=dict(DEFAULT_RUNTIME),
            gradient_clipping={**GRADIENT_CLIPPING,"max_norm":5.0},
            stability_policy=dict(STABILITY_DEFAULTS),
            stability_state={"total_skipped_batches":0,"consecutive_skips":0})
        self.checkpoint["data_manifest"] = {"data_contract":self.checkpoint["data_contract"],
            "tokens":{"unk_index":0,"pad_index":None,"mask":"all_positions"},
            "files":{split:{"sha256":"a"*64,"rows":10,"size_bytes":100} for split in ("train","val")}}
        report=self.inspect()
        self.assertTrue(report["structural_restore_possible"])
        self.assertTrue(report["data_manifest_complete"])
        self.assertTrue(report["epoch_boundary_state_restorable"])
        self.assertFalse(report["data_bytes_reverified"])
        self.assertFalse(report["historical_faithful_continuation_eligible"])

    def test_missing_configuration_runtime_and_fp64_envelope_is_not_complete_resume(self):
        self.checkpoint["checkpoint_role"]="last_resumable"
        report=self.inspect()
        self.assertTrue(report["structural_restore_possible"])
        self.assertFalse(report["data_manifest_complete"])
        self.assertFalse(report["resumable_envelope"]["runtime_complete"])
        self.assertFalse(report["resumable_envelope"]["fp64_clipping_recorded"])
        self.assertFalse(report["epoch_boundary_state_restorable"])


if __name__=="__main__":
    unittest.main()
