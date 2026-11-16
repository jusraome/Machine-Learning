"""Tiny synthetic AE z64/VAE regressions; never a Shakespeare training run."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE=all(importlib.util.find_spec(name) is not None for name in ("torch","numpy"))
if AVAILABLE:
    import torch
    from src import train_remaining_normal as training
    from src import normal_only_protocol as protocol
    from src.autoencoder import (Autoencoder,VariationalAutoencoder,VAEOutput,
                                 reconstruction_loss,kl_divergence)
    from src.train_ae import NonFiniteGradientError


class Pipeline:
    def __init__(self):
        self.vocab={"itos":["<UNK>","a","b"],"stoi":{"<UNK>":0,"a":1,"b":2}}
        self.config={"window_size":5}
        self.groups={"train":{"TRAIN"},"val":{f"V{i}" for i in range(6)},"test":{"TEST"}}
        self.rows={"train":[dict(character_id="TRAIN",x="ababa" if i%3==0 else "bbbbb",
                    y="b" if i%3==0 else "a",y_id=2 if i%3==0 else 1) for i in range(17)],
                   "val":[dict(character_id=s,x="aabaa",y="a" if i%2 else "b",y_id=1 if i%2 else 2)
                          for s in sorted(self.groups["val"]) for i in range(12)]}
        self.read_splits=[]
        self.encoded=[]

    def contract(self):return {"dataset":"remaining-training-fixture","window_size":5,"revision":"unit"}

    def _iter_rows_batches(self,split,batch_size=4):
        if split not in ("train","val"):raise AssertionError("Training read test")
        self.read_splits.append(split)
        for start in range(0,len(self.rows[split]),batch_size):yield self.rows[split][start:start+batch_size]

    def _validate_rows(self,rows,split):
        if any(row["character_id"] not in self.groups[split] for row in rows):raise ValueError("speaker outside split")
        labels=[self.vocab["stoi"].get(row["y"],0) for row in rows]
        if any(row["y_id"]!=label for row,label in zip(rows,labels)):raise ValueError("wrong label")
        return [row["x"] for row in rows],labels

    def encode_tensor_rows(self,rows,split):
        texts,labels=self._validate_rows(rows,split)
        self.encoded.extend((split,row["character_id"],label) for row,label in zip(rows,labels))
        return {"x":torch.tensor([[self.vocab["stoi"].get(c,0) for c in x] for x in texts],dtype=torch.long),
                "y":torch.tensor(labels,dtype=torch.long)}


@unittest.skipUnless(AVAILABLE,"Remaining normal-only models need Torch and NumPy")
class RemainingNormalTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def setUp(self):
        parent=Path(__file__).resolve().parents[1]/"tmp"
        parent.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix="remaining-normal-trainer-",dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.pipeline=Pipeline()
        self.base=dict(epochs=1,min_epochs=1,batch_size=4,embedding_dim=3,hidden_size=4,
            model_selection_examples=8,diagnostic_examples=4,per_speaker_cap=8,read_batch_size=5,
            evaluate_every_updates=1,shuffle_buffer_batches=2,max_updates=2,max_seconds=30)

    def cfg(self,experiment="ae_z64",**overrides):
        model,latent,beta={"ae_z64":("ae",64,0.),"vae_z32_b0p5":("vae",32,.5),
                           "vae_z32_b1":("vae",32,1.),"vae_z32_b4":("vae",32,4.)}[experiment]
        return training.validate_config({**self.base,"experiment_id":experiment,"model":model,
                                         "latent_dim":latent,"beta":beta,**overrides})

    def metric_sequence(self,ces,swaps=None):
        original=training.evaluate_model_selection
        count=0
        def evaluate(*args,**kwargs):
            nonlocal count
            result=original(*args,**kwargs)
            index=min(count,len(ces)-1)
            result["reconstruction_nats_per_sequence"]=float(ces[index])
            result["reconstruction_nats_per_character"]=float(ces[index])/5
            if swaps is not None:result["swap_damage_nats_per_sequence"]=float(swaps[min(count,len(swaps)-1)])
            count+=1
            return result
        return evaluate

    def run_small(self,name="run",experiment="ae_z64",**kwargs):
        cfg=kwargs.pop("config",self.cfg(experiment))
        return training.run_remaining_normal(self.pipeline,self.root/name,config=cfg,device="cpu",
            execute=True,smoke_test=kwargs.pop("smoke_test",True),**kwargs)

    def test_whitelist_cannot_retrain_ae32_or_start_a_grid(self):
        for invalid in ([], "", ["ae_z64"]):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                training.validate_config(invalid)
        for change in ({"experiment_id":"ae_z32"},{"latent_dim":32},{"model":"vae"},
                       {"experiment_id":"vae_z32_b2","model":"vae","latent_dim":32,"beta":2.},
                       {"source_checkpoint":"best8.pt"},{"resume":"last.pt"},{"all":True},
                       {"latent_activation":"tanh"},{"epochs":31},{"grad_clip":4.}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                training.validate_config({**self.cfg(),**change})
        for name in ("ae_z64","vae_z32_b0p5","vae_z32_b1","vae_z32_b4"):
            self.assertEqual(self.cfg(name)["experiment_id"],name)

    def test_factory_changes_only_requested_bottleneck_and_posterior(self):
        ae=training.build_model(self.cfg(),vocab_size=3,sequence_length=5)
        vae=training.build_model(self.cfg("vae_z32_b1"),vocab_size=3,sequence_length=5)
        self.assertIsInstance(ae,Autoencoder)
        self.assertIsInstance(vae,VariationalAutoencoder)
        self.assertEqual(ae.latent_dim,64)
        self.assertEqual(vae.latent_dim,32)
        self.assertEqual(ae.hidden_size,vae.hidden_size)
        self.assertEqual(ae.num_layers,vae.num_layers)

    def test_dry_plan_reads_no_training_data_and_constructs_no_adam(self):
        with patch.object(self.pipeline,"_iter_rows_batches",side_effect=AssertionError("dry data")), \
             patch.object(torch.optim,"Adam",side_effect=AssertionError("dry Adam")):
            result=training.run_remaining_normal(self.pipeline,self.root/"dry",config=self.cfg())
        self.assertFalse(result["execute"])
        self.assertFalse((self.root/"dry").exists())
        self.assertEqual(result["test_access"],"none")

    def test_ae_terms_have_no_kl_and_match_original_reconstruction(self):
        cfg=self.cfg()
        model=training.build_model(cfg,vocab_size=3,sequence_length=5)
        model.train()
        x=torch.tensor([[1,2,1,2,1],[2,1,2,1,2]],dtype=torch.long)
        terms=training.training_terms(model,x,cfg)
        expected=reconstruction_loss(terms["output"].logits,x)
        self.assertTrue(torch.equal(terms["reconstruction"],expected))
        self.assertEqual(terms["kl"].item(),0)
        self.assertEqual(terms["weighted_kl"].item(),0)
        self.assertTrue(torch.equal(terms["total"],expected))

    def test_vae_training_is_sampled_and_beta_weights_kl_once(self):
        x=torch.tensor([[1,2,1,2,1],[2,1,2,1,2]],dtype=torch.long)
        for name,beta in (("vae_z32_b0p5",.5),("vae_z32_b1",1.),("vae_z32_b4",4.)):
            with self.subTest(beta=beta):
                torch.manual_seed(42)
                cfg=self.cfg(name)
                model=training.build_model(cfg,vocab_size=3,sequence_length=5)
                model.train()
                terms=training.training_terms(model,x,cfg)
                output=terms["output"]
                self.assertFalse(torch.equal(output.z,output.mu))
                self.assertTrue(output.z.requires_grad)
                self.assertTrue(torch.allclose(terms["reconstruction"],reconstruction_loss(output.logits,x)))
                self.assertTrue(torch.allclose(terms["kl"],kl_divergence(output.mu,output.logvar)))
                self.assertTrue(torch.allclose(terms["weighted_kl"],beta*terms["kl"]))
                self.assertTrue(torch.allclose(terms["total"],terms["reconstruction"]+beta*terms["kl"]))
                terms["total"].backward()
                self.assertTrue(torch.isfinite(model.mu_head.weight.grad).all().item())
                self.assertTrue(torch.isfinite(model.logvar_head.weight.grad).all().item())
                self.assertGreater(model.logvar_head.weight.grad.abs().sum().item(),0)

    def test_vae_selection_is_deterministic_mu_and_objective_is_not_named_elbo(self):
        torch.manual_seed(42)
        cfg=self.cfg("vae_z32_b4")
        model=training.build_model(cfg,vocab_size=3,sequence_length=5)
        model.train()
        x=torch.tensor([[1,2,1,2,1],[2,1,2,1,2],[1,1,2,2,1]],dtype=torch.long)
        with torch.no_grad():
            reference=model(x,sample=False)
            expected_ce=reconstruction_loss(reference.logits.double(),x).item()
            expected_kl=kl_divergence(reference.mu.double(),reference.logvar.double()).item()
        rng=torch.get_rng_state().clone()
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        flags=[]
        original=model.forward
        def observed(x,sample=None):
            flags.append(sample)
            return original(x,sample=sample)
        with patch.object(model,"forward",side_effect=observed):
            first=training.evaluate_model_selection(model,x,batch_size=2,diagnostic_examples=3,config=cfg)
            second=training.evaluate_model_selection(model,x,batch_size=2,diagnostic_examples=3,config=cfg)
        self.assertEqual(first,second)
        self.assertTrue(flags and all(flag is False for flag in flags))
        self.assertAlmostEqual(first["reconstruction_nats_per_sequence"],expected_ce,places=5)
        self.assertAlmostEqual(first["kl_nats_per_sequence"],expected_kl,places=5)
        self.assertAlmostEqual(first["weighted_kl_nats_per_sequence"],4*expected_kl,places=5)
        self.assertAlmostEqual(first["objective_at_mu_nats_per_sequence"],expected_ce+4*expected_kl,places=5)
        posterior=first["posterior"]
        self.assertEqual(len(posterior["kl_per_dimension"]),32)
        self.assertAlmostEqual(sum(posterior["kl_per_dimension"]),expected_kl,places=5)
        self.assertGreaterEqual(posterior["active_units"],0)
        self.assertLessEqual(posterior["active_units"],32)
        self.assertTrue(torch.isfinite(torch.tensor([posterior[key] for key in
            ("mu_rms","mu_between_example_std_rms","logvar_min","logvar_max","logvar_mean","mean_posterior_variance")])).all().item())
        self.assertFalse(any("elbo" in key.lower() for key in first))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(model.training)
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_existing_outputs_are_preserved(self):
        destination=self.root/"old"
        destination.mkdir()
        path=destination/"keep.pt"
        path.write_bytes(b"previous")
        with self.assertRaises(FileExistsError):self.run_small(name="old")
        self.assertEqual(path.read_bytes(),b"previous")

    def test_ae_smoke_is_fresh_normal_only_and_keeps_best8_untouched(self):
        historical=self.root/"best8.pt"
        historical.write_bytes(b"historical-not-a-source")
        digest=hashlib.sha256(historical.read_bytes()).hexdigest()
        with patch.object(torch.optim.Adam,"load_state_dict",side_effect=AssertionError("restored Adam")), \
             patch.object(training,"evaluate_model_selection",side_effect=self.metric_sequence([20,19,18],[0,.1,.2])):
            result=self.run_small()
        self.assertEqual(result["status"],"completed",result.get("reason"))
        self.assertEqual(result["global_updates"],2)
        initial=torch.load(self.root/"run/checkpoints/update_00000000.pt",weights_only=True)
        self.assertEqual(initial["optimizer_state_dict"]["state"],{})
        self.assertIsNone(initial["source_checkpoint"])
        frozen=torch.load(result["best_inference"]["path"],weights_only=True)
        self.assertEqual(frozen["model_type"],"ae")
        self.assertEqual(frozen["model_hparams"]["latent_dim"],64)
        self.assertEqual(frozen["checkpoint_role"],"remaining_normal_best_inference")
        self.assertEqual(frozen["accepted_updates"],2)
        self.assertNotIn("optimizer_state_dict",frozen)
        self.assertEqual(hashlib.sha256(historical.read_bytes()).hexdigest(),digest)
        selection=set(protocol.role_partition(self.pipeline,42)["roles"]["model_selection"])
        self.assertTrue(all(label==1 for _,_,label in self.pipeline.encoded))
        self.assertTrue(all(speaker in selection for split,speaker,_ in self.pipeline.encoded if split=="val"))
        self.assertNotIn("test",self.pipeline.read_splits)

    def test_each_beta_has_identical_fresh_initial_weights_and_independent_adam(self):
        snapshots=[]
        for name in ("vae_z32_b0p5","vae_z32_b1","vae_z32_b4"):
            with patch.object(torch.optim.Adam,"load_state_dict",side_effect=AssertionError("restored Adam")), \
                 patch.object(training,"evaluate_model_selection",side_effect=self.metric_sequence([30,29,28],[0,0,0])):
                result=self.run_small(name=name,experiment=name)
            self.assertEqual(result["status"],"completed",result.get("reason"))
            initial=torch.load(self.root/name/"checkpoints/update_00000000.pt",weights_only=True)
            self.assertEqual(initial["optimizer_state_dict"]["state"],{})
            snapshots.append(initial["state_dict"])
            self.assertEqual(result["training_config"]["experiment_id"],name)
        self.assertTrue(all(torch.equal(snapshots[0][key],other[key]) for other in snapshots[1:] for key in snapshots[0]))

    def test_invalid_loss_gradient_or_posterior_never_reaches_adam(self):
        original=training.training_terms
        def invalid_terms(model,x,config):
            terms=original(model,x,config)
            terms["total"]=terms["total"]*float("nan")
            return terms
        original_forward=VariationalAutoencoder.forward
        def bad_posterior(model,x,sample=None):
            output=original_forward(model,x,sample=sample)
            if model.training:
                return VAEOutput(output.logits,output.z,output.mu,torch.full_like(output.logvar,float("inf")))
            return output
        contexts=(("loss",patch.object(training,"training_terms",side_effect=invalid_terms),"ae_z64"),
                  ("gradient",patch.object(training,"stable_clip_grad_norm_",side_effect=NonFiniteGradientError("invalid")),"ae_z64"),
                  ("posterior",patch.object(VariationalAutoencoder,"forward",new=bad_posterior),"vae_z32_b1"))
        for name,context,experiment in contexts:
            with self.subTest(name=name),context,patch.object(torch.optim.Adam,"step",side_effect=AssertionError("unsafe Adam")):
                result=self.run_small(name=name,experiment=experiment)
            self.assertEqual(result["global_updates"],0)
            self.assertEqual(result["status"],"stopped",result.get("reason"))
            self.assertIsNone(result["best_inference"])

    def test_gradient_reset_and_fp64_clipping_are_retained(self):
        clips=[]
        original=training.stable_clip_grad_norm_
        def checked(parameters,max_norm):
            report=original(parameters,max_norm)
            clips.append(report)
            return report
        with patch.object(training,"stable_clip_grad_norm_",side_effect=checked), \
             patch.object(training,"evaluate_model_selection",side_effect=self.metric_sequence([20,19,18],[0,0,0])):
            result=self.run_small()
        self.assertEqual(result["global_updates"],2)
        self.assertEqual(len(clips),2)
        self.assertTrue(all(row["norm_after"]<=5*(1+1e-6) for row in clips))
        saved=torch.load(self.root/"run/checkpoints/update_00000002.pt",weights_only=True)
        self.assertEqual(saved["gradient_clipping"]["norm_dtype"],"float64")
        self.assertEqual(saved["gradient_clipping"]["scaling_dtype"],"float64")
        rows=[json.loads(line) for line in (self.root/"run/updates.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["gradient_reset_confirmed"] for row in rows))

    def test_deterioration_does_not_replace_healthy_best(self):
        cfg=self.cfg(max_updates=10)
        with patch.object(training,"evaluate_model_selection",side_effect=self.metric_sequence([20,19,25,26],[10,10,10,10])):
            result=self.run_small(config=cfg,smoke_test=False)
        self.assertEqual(result["status"],"stopped",result.get("reason"))
        self.assertEqual(result["best_checkpoint"]["global_updates"],1)
        pointer=result["best_checkpoint"]
        self.assertEqual(hashlib.sha256(Path(pointer["path"]).read_bytes()).hexdigest(),pointer["sha256"])
        self.assertEqual(result["best_inference"]["global_updates"],1)

    def test_untrained_initial_model_is_never_exported_as_final_detector(self):
        with patch.object(training,"evaluate_model_selection",side_effect=self.metric_sequence([20,21,21],[0,0,0])):
            result=self.run_small()
        self.assertEqual(result["best_checkpoint"]["global_updates"],0)
        self.assertIsNone(result["best_inference"])
        self.assertFalse((self.root/"run/best_inference.pt").exists())


if __name__=="__main__":unittest.main()
