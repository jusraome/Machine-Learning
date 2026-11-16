"""Tiny synthetic normal-only training regressions; never a production run."""
import copy
import csv
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
    from src import train_normal_ae as training
    from src import normal_only_protocol as protocol
    from src.train_ae import NonFiniteGradientError


class Pipeline:
    def __init__(self):
        self.vocab={"itos":["<UNK>","a","b"],"stoi":{"<UNK>":0,"a":1,"b":2}}
        self.config={"window_size":5}
        self.groups={"train":{"TRAIN"},"val":{f"V{i}" for i in range(6)},"test":{"TEST"}}
        self.rows={"train":[dict(character_id="TRAIN",x="ababa" if i%3==0 else "bbbbb",
                    y="b" if i%3==0 else "a",y_id=2 if i%3==0 else 1) for i in range(17)],
                   "val":[dict(character_id=s,x="ababa",y="a" if i%2 else "b",y_id=1 if i%2 else 2)
                          for s in sorted(self.groups["val"]) for i in range(12)]}
        self.read_splits=[]
        self.encoded=[]

    def contract(self):return {"dataset":"normal-training-fixture","window_size":5,"revision":"unit"}

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


@unittest.skipUnless(AVAILABLE,"normal-only training needs Torch and NumPy")
class NormalOnlyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def setUp(self):
        temporary_root=Path(__file__).resolve().parents[1]/"tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix="normal-only-trainer-",dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.pipeline=Pipeline()
        self.config=dict(epochs=1,min_epochs=1,batch_size=4,embedding_dim=3,hidden_size=4,
            model_selection_examples=8,diagnostic_examples=4,per_speaker_cap=8,read_batch_size=5,
            evaluate_every_updates=1,shuffle_buffer_batches=2,max_updates=2,max_seconds=30)

    def run_small(self,name="run",**kwargs):
        cfg=kwargs.pop("config",self.config)
        return training.run_normal_only(self.pipeline,self.root/name,config=cfg,device="cpu",
                                        execute=True,smoke_test=kwargs.pop("smoke_test",True),**kwargs)

    def metric_sequence(self,ces,swaps=None):
        original=training.evaluate_model_selection
        count=0
        def metrics(*args,**kwargs):
            nonlocal count
            result=original(*args,**kwargs)
            index=min(count,len(ces)-1)
            result["reconstruction_nats_per_sequence"]=float(ces[index])
            result["reconstruction_nats_per_character"]=float(ces[index])/5
            if swaps is not None:result["swap_damage_nats_per_sequence"]=float(swaps[min(count,len(swaps)-1)])
            count+=1
            return result
        return metrics

    def test_default_plan_does_not_read_data_create_adam_or_results(self):
        with patch.object(self.pipeline,"_iter_rows_batches",side_effect=AssertionError("dry read")), \
             patch.object(torch.optim,"Adam",side_effect=AssertionError("dry Adam")):
            plan=training.run_normal_only(self.pipeline,self.root/"dry",config=self.config)
        self.assertFalse(plan["execute"])
        self.assertFalse((self.root/"dry").exists())
        self.assertEqual(plan["test_access"],"none")
        self.assertIn("from_scratch",plan["operation"])

    def test_cannot_initialize_historical_models_other_architectures_or_grids(self):
        for change in ({"model":"vae"},{"latent_dim":64},{"latent_activation":"tanh"},
                       {"source_checkpoint":"best.pt"},{"resume":"last.pt"},{"all":True},
                       {"epochs":31},{"min_epochs":31},{"max_seconds":21601},{"grad_clip":4.0}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                training.validate_config({**self.config,**change})

    def test_existing_output_is_not_reused_or_changed(self):
        destination=self.root/"old"
        destination.mkdir()
        sentinel=destination/"keep.pt"
        sentinel.write_bytes(b"historical")
        with self.assertRaises(FileExistsError):self.run_small(name="old")
        self.assertEqual(sentinel.read_bytes(),b"historical")

    def test_one_from_scratch_smoke_keeps_historical_file_and_never_restores_adam(self):
        historical=self.root/"historical_best.pt"
        historical.write_bytes(b"not a training source")
        before=hashlib.sha256(historical.read_bytes()).hexdigest()
        with patch.object(torch.optim.Adam,"load_state_dict",side_effect=AssertionError("restored old Adam")), \
             patch.object(training,"evaluate_model_selection",side_effect=self.metric_sequence([20,19,18],[0,.1,.2])):
            result=self.run_small()
        self.assertEqual(result["status"],"completed",result.get("reason"))
        self.assertEqual(result["global_updates"],2)
        self.assertEqual(result["training_config"]["normal_class"],1)
        self.assertTrue(result["training_config"]["from_scratch"])
        self.assertIsNotNone(result["best_inference"])
        initial=torch.load(self.root/"run/checkpoints/update_00000000.pt",weights_only=True)
        final=torch.load(result["best_inference"]["path"],weights_only=True)
        self.assertEqual(initial["optimizer_state_dict"]["state"],{})
        self.assertIsNone(initial["source_checkpoint"])
        self.assertEqual(final["checkpoint_role"],"normal_only_ae_best_inference")
        self.assertEqual(final["accepted_updates"],2)
        self.assertNotIn("optimizer_state_dict",final)
        self.assertEqual(hashlib.sha256(historical.read_bytes()).hexdigest(),before)

    def test_training_sees_only_normals_and_selection_never_encodes_other_roles(self):
        partition=protocol.role_partition(self.pipeline,self.config.get("seed",42))
        with patch.object(training,"evaluate_model_selection",side_effect=self.metric_sequence([20,19,18],[0,0,0])):
            result=self.run_small()
        self.assertEqual(result["status"],"completed",result.get("reason"))
        self.assertEqual(set(self.pipeline.read_splits),{"train","val"})
        selection=set(partition["roles"]["model_selection"])
        self.assertTrue(all(label==1 for _,_,label in self.pipeline.encoded))
        self.assertTrue(all(speaker in selection for split,speaker,_ in self.pipeline.encoded if split=="val"))
        updates=[json.loads(line) for line in (self.root/"run/updates.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["all_labels_normal"] and row["normal_class"]==1 for row in updates))
        sampling=json.loads((self.root/"run/model_selection_sampling.json").read_text())
        self.assertEqual(set(sampling["roles"]),{"model_selection"})

    def test_complete_normal_epoch_includes_partial_batch_without_dropping_examples(self):
        cfg={**self.config,"max_updates":10}
        with patch.object(training,"evaluate_model_selection",side_effect=self.metric_sequence([20,19,18,17],[0,0,0,0])):
            result=self.run_small(config=cfg,smoke_test=False)
        self.assertEqual(result["status"],"completed",result.get("reason"))
        self.assertEqual(result["epochs_completed"],1)
        self.assertEqual(result["training_examples_seen"],11)
        updates=[json.loads(line) for line in (self.root/"run/updates.jsonl").read_text().splitlines()]
        self.assertEqual([r["examples"] for r in updates],[4,4,3])

    def test_invalid_loss_or_gradient_stops_before_any_adam_step(self):
        original_loss=training.reconstruction_loss
        def invalid_loss(logits,targets,*args,**kwargs):
            return logits.sum()*float("nan") if torch.is_grad_enabled() else original_loss(logits,targets,*args,**kwargs)
        for name,context in (("loss",patch.object(training,"reconstruction_loss",side_effect=invalid_loss)),
                             ("gradient",patch.object(training,"stable_clip_grad_norm_",side_effect=NonFiniteGradientError("bad gradient")))):
            with self.subTest(name=name),context,patch.object(torch.optim.Adam,"step",side_effect=AssertionError("unsafe Adam")):
                result=self.run_small(name=name)
            self.assertEqual(result["global_updates"],0)
            self.assertEqual(result["status"],"stopped",result.get("reason"))
            self.assertTrue((self.root/name/"checkpoints/update_00000000.pt").is_file())
            self.assertIsNone(result["best_inference"])

    def test_gradient_reset_and_safe_fp64_clipping_are_checked_every_update(self):
        clips=[]
        forwards=[]
        original_clip=training.stable_clip_grad_norm_
        original_forward=training.Autoencoder.forward
        def forward(model,x):
            if model.training:
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                forwards.append(len(x))
            return original_forward(model,x)
        def clip(parameters,max_norm):
            report=original_clip(parameters,max_norm)
            clips.append(report)
            return report
        with patch.object(training.Autoencoder,"forward",new=forward), \
             patch.object(training,"stable_clip_grad_norm_",side_effect=clip), \
             patch.object(training,"evaluate_model_selection",side_effect=self.metric_sequence([20,19,18],[0,0,0])):
            result=self.run_small()
        self.assertEqual(result["global_updates"],2)
        self.assertEqual(forwards,[4,4])
        self.assertEqual(len(clips),2)
        self.assertTrue(all(c["norm_after"]<=5*(1+1e-6) for c in clips))
        checkpoint=torch.load(self.root/"run/checkpoints/update_00000002.pt",weights_only=True)
        self.assertEqual(checkpoint["gradient_clipping"]["norm_dtype"],"float64")
        self.assertEqual(checkpoint["gradient_clipping"]["scaling_dtype"],"float64")

    def test_best_is_immutable_and_deterioration_cannot_replace_it(self):
        cfg={**self.config,"max_updates":10}
        with patch.object(training,"evaluate_model_selection",side_effect=self.metric_sequence([20,19,25,26],[10,10,10,10])):
            result=self.run_small(config=cfg,smoke_test=False)
        self.assertEqual(result["status"],"stopped",result.get("reason"))
        self.assertEqual(result["global_updates"],3)
        self.assertEqual(result["best_checkpoint"]["global_updates"],1)
        best=Path(result["best_checkpoint"]["path"])
        self.assertEqual(hashlib.sha256(best.read_bytes()).hexdigest(),result["best_checkpoint"]["sha256"])
        self.assertEqual(result["best_inference"]["global_updates"],1)
        self.assertTrue((self.root/"run/checkpoints/update_00000003.pt").is_file())
        with self.assertRaises(FileExistsError):training._checkpoint(best,{"replace":True})
        self.assertEqual(hashlib.sha256(best.read_bytes()).hexdigest(),result["best_checkpoint"]["sha256"])

    def test_initial_random_best_is_preserved_without_export_as_a_trained_detector(self):
        with patch.object(training,"evaluate_model_selection",side_effect=self.metric_sequence([20,21,21],[0,0,0])):
            result=self.run_small()
        self.assertEqual(result["best_checkpoint"]["global_updates"],0)
        self.assertIsNone(result["best_inference"])
        self.assertIn("no_trained_best",result["best_inference_status"])
        self.assertFalse((self.root/"run/best_inference.pt").exists())

    def test_plateau_and_learned_utility_rules_are_distinct_from_random_response(self):
        cfg=training.validate_config({**self.config,"early_stopping_patience":2})
        state,decision=training.training_decision(20,0,None,cfg)
        state,decision=training.training_decision(20,-.1,state,cfg)
        self.assertFalse(decision["early_stopping"])
        self.assertFalse(decision["swap_guard_applicable"])
        state,decision=training.training_decision(20,.001,state,cfg)
        self.assertTrue(decision["early_stopping"])
        self.assertFalse(decision["stop"])
        state,decision=training.training_decision(19,10,state,cfg)
        self.assertFalse(decision["early_stopping"])
        state,decision=training.training_decision(18,4,state,cfg)
        self.assertTrue(decision["swap_guard_applicable"])
        self.assertFalse(decision["healthy"])
        self.assertFalse(decision["stop"])
        state,decision=training.training_decision(17,3,state,cfg)
        self.assertTrue(decision["stop"])
        self.assertIn("sustained_loss_of_learned_latent_utility",decision["reasons"])

    def test_plateau_warmup_does_not_stop_before_three_completed_epochs(self):
        cfg={**self.config,"epochs":4,"min_epochs":3,"max_updates":20,"early_stopping_patience":1}
        with patch.object(training,"evaluate_model_selection",side_effect=self.metric_sequence([20,19,19],[0,0,0])):
            result=self.run_small(config=cfg,smoke_test=False)
        self.assertEqual(result["reason"],"early_stopping_normal_selection")
        self.assertGreaterEqual(result["epochs_completed"],3)
        early=[row for row in result["evaluations"] if row["decision"]["early_stopping"]]
        self.assertTrue(early)
        self.assertTrue(all(row["decision"]["early_stopping_minimum_epochs_satisfied"] for row in early))
        before=[r for r in result["evaluations"] if not r["decision"]["early_stopping_minimum_epochs_satisfied"]]
        self.assertTrue(before)
        self.assertTrue(all(not r["decision"]["early_stopping"] for r in before))

    def test_deadline_failure_leaves_an_honest_summary_and_no_updates(self):
        with patch.object(training,"_deadline",side_effect=TimeoutError("injected deadline")), \
             patch.object(torch.optim.Adam,"step",side_effect=AssertionError("expired Adam")):
            result=self.run_small()
        self.assertEqual(result["global_updates"],0)
        self.assertEqual(result["status"],"failed")
        self.assertIn("deadline",result["reason"])
        self.assertTrue((self.root/"run/summary.json").is_file())


if __name__=="__main__":unittest.main()
