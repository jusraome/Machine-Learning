"""Common development comparisons on synthetic frozen AE/VAE states only."""
import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE=all(importlib.util.find_spec(n) for n in ("torch","numpy"))
if AVAILABLE:
    import torch
    from src.autoencoder import Autoencoder,VariationalAutoencoder
    from src import evaluate_normal_models as compare
    from src import evaluate_normal_ae as previous
    from src.normal_only_protocol import role_partition,fit_normal_class,build_train_input_guard
    from test_evaluate_normal_ae import Pipeline


@unittest.skipUnless(AVAILABLE,"Common development comparison needs torch/numpy")
class NormalModelsComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def setUp(self):
        directory=Path(__file__).resolve().parents[1]/"tmp"
        directory.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix="normal-model-comparison-",dir=directory)
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.pipeline=Pipeline()
        self.partition=role_partition(self.pipeline,42)
        self.normal=fit_normal_class(self.pipeline,batch_size=8)
        self.guard=build_train_input_guard(self.pipeline,1,read_batch_size=8)
        self.config=dict(max_calibration=12,max_evaluation=12,per_speaker_cap=6,batch_size=4,
                         read_batch_size=8,max_seconds=60,diagnostic_examples=4,alpha=.2)
        self.paths={name:self.checkpoint(name) for name in ("ae_z32","ae_z64","vae_z32_b0p5","vae_z32_b1","vae_z32_b4","best8")}
        self.pipeline.read_splits.clear();self.pipeline.encodings.clear()
    def checkpoint(self,label,**changes):
        kind,latent,beta=compare.IDENTITIES[label]
        torch.manual_seed(42+latent+int(beta*10))
        cls=Autoencoder if kind=="ae" else VariationalAutoencoder
        model=cls(vocab_size=3,sequence_length=5,embedding_dim=32,hidden_size=128,latent_dim=latent)
        payload=dict(model_type=kind,model_hparams=model.hparams,state_dict=model.state_dict(),
            vocab=self.pipeline.vocab,data_contract=self.pipeline.contract(),epoch=8 if label=="best8" else 2,
            global_updates=8,experiment_id=label,
            checkpoint_role="best_inference_only" if label=="best8" else
                "normal_only_ae_best_inference" if label=="ae_z32" else "remaining_normal_best_inference",
            training_config=dict(from_scratch=True,normal_class=None if label=="best8" else 1,
                smoke_test=False,experiment_id=label,beta=beta),
            normal_only_protocol=dict(from_scratch=True,normal_class=1,role_digest=self.partition["role_digest"],
                experiment_scope="normal_only_training",experiment_id=label,beta=beta),
            normal_definition=self.normal,validation_partition=self.partition,train_input_guard=self.guard["metadata"],
            source_checkpoint=None,optimizer_state_dict={"must_not_restore":True})
        payload.update(changes)
        path=self.root/f"{label}.pt";torch.save(payload,path);return path
    def run_small(self,name="comparison",labels=("ae_z32","ae_z64","vae_z32_b1"),**kwargs):
        return compare.run_development_comparison(self.pipeline,{label:self.paths[label] for label in labels},
            self.root/name,config=self.config,execute=True,**kwargs)
    def make_reference(self):
        old_config={k:v for k,v in self.config.items() if k!="diagnostic_examples"}
        result=previous.run_normal_evaluation(self.pipeline,self.paths["ae_z32"],self.root/"old_reference",
            baseline_checkpoint=self.paths["best8"],config=old_config,execute=True)
        self.assertEqual(result["status"],"completed",result.get("reason"))
        return self.root/"old_reference/frozen_protocol.json"
    def test_dry_plan_has_no_checkpoint_or_data_side_effects_and_no_test_mode(self):
        with patch.object(torch,"load",side_effect=AssertionError("checkpoint in plan")),\
             patch.object(self.pipeline,"_iter_rows_batches",side_effect=AssertionError("data in plan")),\
             patch.object(torch.optim,"Adam",side_effect=AssertionError("optimizer")):
            plan=compare.run_development_comparison(self.pipeline,{"ae_z64":"missing.pt"},self.root/"dry")
        self.assertFalse(plan["execute"]);self.assertIn("closed",plan["test_access"])
        self.assertFalse((self.root/"dry").exists())
        with self.assertRaises(ValueError):compare.validate_config({"evaluate_test":True})
        with self.assertRaises(ValueError):compare.run_development_comparison(self.pipeline,{"best8":"x"},self.root/"only_old")
    def test_uniform_vae_scores_use_mu_and_separate_kl_with_common_units(self):
        model=VariationalAutoencoder(vocab_size=3,sequence_length=5,embedding_dim=32,hidden_size=128,latent_dim=32)
        with torch.no_grad():
            for parameter in model.parameters():parameter.zero_()
            model.mu_head.bias.fill_(2)
        model.train();x=torch.ones(2,5,dtype=torch.long)
        before=torch.get_rng_state().clone()
        scores,z,posterior=compare.reconstruction_and_kl_scores(model,x,beta=4)
        self.assertTrue(model.training);self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(torch.equal(z,torch.full((2,32),2.,dtype=torch.float64)))
        self.assertTrue(torch.allclose(scores["brier_mse"],torch.full((2,),2/9,dtype=torch.float64)))
        self.assertTrue(torch.allclose(scores["nll_nats_per_sequence"],torch.full((2,),5*math.log(3),dtype=torch.float64)))
        self.assertTrue(torch.equal(posterior["kl"],torch.full((2,),64.,dtype=torch.float64)))
        self.assertTrue(torch.allclose(scores["nll_plus_kl_nats_per_sequence"],scores["nll_nats_per_sequence"]+64))
        self.assertTrue(torch.allclose(scores["nll_plus_training_beta_kl_nats_per_sequence"],scores["nll_nats_per_sequence"]+256))
        self.assertNotIn("mse_plus_kl",scores)
    def test_all_supported_models_share_rows_and_never_construct_adam_or_read_test(self):
        original={label:hashlib.sha256(path.read_bytes()).hexdigest() for label,path in self.paths.items()}
        rng=torch.get_rng_state().clone();python_rng=random.getstate()
        with patch.object(torch.optim,"Adam",side_effect=AssertionError("optimizer created")):
            result=self.run_small(labels=tuple(self.paths))
        self.assertEqual(result["status"],"completed",result.get("reason"))
        self.assertFalse(result["test_accessed"]);self.assertFalse(result["test_ready"])
        self.assertFalse(result["winner_selected"]);self.assertNotIn("test",self.pipeline.read_splits)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(python_rng,random.getstate())
        self.assertTrue(all(hashlib.sha256(path.read_bytes()).hexdigest()==original[label] for label,path in self.paths.items()))
        with (self.root/"comparison/scores.csv").open(newline="",encoding="utf-8") as stream:rows=list(csv.DictReader(stream))
        sets={label:{r["row_index"] for r in rows if r["model"]==label and r["population"]=="evaluation"} for label in self.paths}
        self.assertTrue(all(value==sets["ae_z32"] for value in sets.values()))
        self.assertEqual(set(result["metrics"]["ae_z64"]),set(compare.COMMON_SCORES))
        self.assertEqual(set(result["metrics"]["vae_z32_b4"]),set((*compare.COMMON_SCORES,*compare.VAE_SCORES)))
        self.assertIn("mean_kl_nats_per_sequence",result["reconstruction_latent"]["vae_z32_b1"])
        json.dumps(result,allow_nan=False)
    def test_seal_exists_before_every_mixed_forward(self):
        original=compare._score_population
        observations=[]
        def checked(*args,**kwargs):
            if kwargs.get("diagnostics") or len(args)>7 and args[7]:
                path=self.root/"comparison/frozen_protocol.json"
                self.assertTrue(path.is_file());self.assertTrue(path.with_suffix(".sha256").is_file())
                frozen=json.loads(path.read_text());self.assertEqual(len(frozen["models"]),3)
                observations.append(True)
            return original(*args,**kwargs)
        with patch.object(compare,"_score_population",side_effect=checked):result=self.run_small()
        self.assertEqual(result["status"],"completed",result.get("reason"));self.assertEqual(len(observations),3)
    def test_existing_ae32_reference_preserves_exact_ids_order_content_and_counts(self):
        path=self.make_reference();reference=json.loads(path.read_text())
        result=self.run_small(reference_protocol=path,labels=("ae_z32","ae_z64","vae_z32_b0p5","best8"))
        self.assertEqual(result["status"],"completed",result.get("reason"))
        self.assertTrue(result["exact_reference_reused"])
        cohort=json.loads((self.root/"comparison/cohort.json").read_text())
        self.assertEqual(cohort["selected_rows"],reference["selected_rows"])
        self.assertEqual(result["evaluation_examples"],len(reference["selected_rows"]["evaluation"]))
    def test_rehashing_changed_reference_row_still_rejects_actual_content_before_forward(self):
        path=self.make_reference();value=json.loads(path.read_text())
        value["selected_rows"]["evaluation"][0]["row_sha256"]="0"*64
        for role,records in value["selected_rows"].items():value["selected_rows_sha256"][role]=hashlib.sha256(compare._canonical(records)).hexdigest()
        path.write_text(json.dumps(value));path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
        with patch.object(compare,"reconstruction_and_kl_scores",side_effect=AssertionError("forward happened")):
            result=self.run_small(reference_protocol=path)
        self.assertEqual(result["status"],"failed");self.assertIn("Reference row content",result["reason"])
    def test_reference_checksum_and_missing_row_are_not_silently_resampled(self):
        path=self.make_reference();raw=path.read_bytes();path.write_bytes(raw+b" ")
        result=self.run_small(name="bad_sha",reference_protocol=path)
        self.assertEqual(result["status"],"failed");self.assertIn("SHA-256",result["reason"])
        path.write_bytes(raw)
        self.pipeline.rows["val"].pop(0)
        result=self.run_small(name="missing_row",reference_protocol=path)
        self.assertEqual(result["status"],"failed")
        self.assertNotIn("scores.csv",[p.name for p in (self.root/"missing_row").iterdir()])
    def test_reference_alpha_class_or_source_model_changes_are_rejected(self):
        path=self.make_reference()
        cfg={**self.config,"alpha":.1}
        result=compare.run_development_comparison(self.pipeline,{"ae_z64":self.paths["ae_z64"]},self.root/"alpha",config=cfg,reference_protocol=path,execute=True)
        self.assertEqual(result["status"],"failed");self.assertIn("reference score",result["reason"])
        self.checkpoint("ae_z32",state_dict=copy.deepcopy(torch.load(self.paths["ae_z64"],weights_only=True)["state_dict"]))
        result=self.run_small(name="changed_ae32",reference_protocol=path)
        self.assertEqual(result["status"],"failed")

    def test_valid_but_changed_ae32_weights_fail_the_reference_source_sha_gate(self):
        path=self.make_reference()
        payload=torch.load(self.paths["ae_z32"],weights_only=True)
        payload["state_dict"]["output_projection.bias"]+=.01
        torch.save(payload,self.paths["ae_z32"])
        result=self.run_small(name="changed_valid_ae32",reference_protocol=path)
        self.assertEqual(result["status"],"failed")
        self.assertIn("AE z32 differs",result["reason"])

    def test_new_comparison_reference_also_fixes_the_ae32_source_hash(self):
        initial=self.run_small(name="first_comparison")
        self.assertEqual(initial["status"],"completed",initial.get("reason"))
        path=self.root/"first_comparison/frozen_protocol.json"
        repeated=self.run_small(name="same_reference",reference_protocol=path)
        self.assertEqual(repeated["status"],"completed",repeated.get("reason"))
        self.assertEqual(repeated["evaluation_examples"],initial["evaluation_examples"])
        payload=torch.load(self.paths["ae_z32"],weights_only=True)
        payload["state_dict"]["output_projection.bias"]+=.01
        torch.save(payload,self.paths["ae_z32"])
        changed=self.run_small(name="new_reference_changed_ae32",reference_protocol=path)
        self.assertEqual(changed["status"],"failed")
        self.assertIn("AE z32 differs",changed["reason"])
    def test_wrong_identity_beta_untrained_or_smoke_checkpoints_are_rejected(self):
        for index,changes in enumerate((dict(global_updates=0),dict(checkpoint_role="remaining_normal_periodic_state"),
                dict(experiment_id="vae_z32_b4"),dict(training_config=dict(from_scratch=True,normal_class=1,smoke_test=False,beta=4)))):
            path=self.checkpoint("vae_z32_b1",**changes)
            result=compare.run_development_comparison(self.pipeline,{"vae_z32_b1":path},self.root/f"bad_{index}",config=self.config,execute=True)
            self.assertEqual(result["status"],"failed")
            self.assertFalse((self.root/f"bad_{index}/scores.csv").exists())
    def test_changed_training_inputs_reject_cohort_before_scoring(self):
        self.pipeline.rows["train"][0]["x"]="aaaaa"
        result=self.run_small()
        self.assertEqual(result["status"],"failed");self.assertIn("training input guard",result["reason"])
    def test_insufficient_calibration_is_explicit_and_test_stays_closed(self):
        cfg={**self.config,"max_calibration":1,"alpha":.05}
        result=compare.run_development_comparison(self.pipeline,{"ae_z64":self.paths["ae_z64"]},self.root/"small_cal",config=cfg,execute=True)
        self.assertEqual(result["status"],"completed",result.get("reason"))
        self.assertEqual(result["calibration_status"],"insufficient_normal_calibration")
        self.assertFalse(result["test_ready"])
    def test_output_reuse_and_configuration_expansion_are_refused(self):
        destination=self.root/"old";destination.mkdir();(destination/"keep").write_text("unchanged")
        with self.assertRaises(FileExistsError):compare.run_development_comparison(self.pipeline,{"ae_z64":self.paths["ae_z64"]},destination)
        self.assertEqual((destination/"keep").read_text(),"unchanged")
        for bad in ({"max_evaluation":16385},{"max_calibration":8192,"max_evaluation":16384},{"beta":1},{"primary_score":"mse_plus_kl"}):
            with self.assertRaises(ValueError):compare.validate_config(bad)


if __name__=="__main__":unittest.main()
