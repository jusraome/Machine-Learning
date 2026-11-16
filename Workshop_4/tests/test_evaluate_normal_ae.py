"""Synthetic frozen normal-only evaluation; no optimizer or training steps."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE=all(importlib.util.find_spec(name) for name in ("torch","numpy"))
if AVAILABLE:
    import torch
    from src.autoencoder import Autoencoder
    from src import evaluate_normal_ae as evaluate
    from src.normal_only_protocol import role_partition,fit_normal_class,build_train_input_guard


class Pipeline:
    def __init__(self):
        self.vocab=dict(itos=["<UNK>","a","b"],stoi={"<UNK>":0,"a":1,"b":2},size=3,unk_index=0)
        self.config=dict(window_size=5,batch_size=4)
        self.groups={"train":{"TRAIN"},"val":{f"V{i}" for i in range(6)},"test":{"TEST"}}
        self.rows={"train":[dict(character_id="TRAIN",x="ababa",y="a" if i<12 else "b",y_id=1 if i<12 else 2) for i in range(16)],
            "val":[dict(character_id=s,x="aaaba" if i%2 else "baZab",y="a" if i%2 else "b",y_id=1 if i%2 else 2)
                   for s in sorted(self.groups["val"]) for i in range(16)],
            "test":[dict(character_id="TEST",x="abbba" if i%2 else "bbaba",y="a" if i%2 else "b",y_id=1 if i%2 else 2) for i in range(16)]}
        self.read_splits=[]
        self.encodings=[]
        self.allow_test=False

    def contract(self):return dict(dataset="synthetic_normal_detector",revision="unit",window_size=5)
    def _iter_rows_batches(self,split,batch_size=None):
        if split=="test" and not self.allow_test:raise AssertionError("Test must stay sealed")
        self.read_splits.append(split)
        for start in range(0,len(self.rows[split]),batch_size or 4):yield self.rows[split][start:start+(batch_size or 4)]
    def _validate_rows(self,rows,split):
        if any(row["character_id"] not in self.groups[split] for row in rows):raise ValueError("Wrong speaker")
        return [row["x"] for row in rows],[self.vocab["stoi"].get(row["y"],0) for row in rows]
    def encode_tensor_rows(self,rows,split):
        self._validate_rows(rows,split)
        self.encodings.append((split,[row["character_id"] for row in rows]))
        return {"x":torch.tensor([[self.vocab["stoi"].get(c,0) for c in row["x"]] for row in rows],dtype=torch.long),
                "y":torch.tensor([row["y_id"] for row in rows],dtype=torch.long)}
    def iter_tensor_batches(self,split,batch_size=None):
        for rows in self._iter_rows_batches(split,batch_size):yield self.encode_tensor_rows(rows,split)
    def validate_checkpoint(self,checkpoint):
        if checkpoint["vocab"]!=self.vocab or checkpoint["data_contract"]!=self.contract():raise ValueError("Wrong contract")
        return True


@unittest.skipUnless(AVAILABLE,"Frozen normal-only evaluation requires torch/numpy")
class NormalOnlyEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def setUp(self):
        temporary=Path(__file__).resolve().parents[1]/"tmp"
        temporary.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix="normal-evaluation-",dir=temporary)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.pipeline=Pipeline()
        self.config=dict(max_calibration=12,max_evaluation=12,max_test=12,per_speaker_cap=6,
                         batch_size=4,read_batch_size=8,max_seconds=60,alpha=.2)
        self.partition=role_partition(self.pipeline,42)
        self.normal=fit_normal_class(self.pipeline,batch_size=8)
        self.guard=build_train_input_guard(self.pipeline,self.normal["normal_class"],read_batch_size=8)
        self.checkpoint=self.make_checkpoint("normal.pt")
        self.baseline=self.make_checkpoint("best8.pt",baseline=True)
        self.pipeline.read_splits.clear()
        self.pipeline.encodings.clear()

    def make_checkpoint(self,name,baseline=False):
        torch.manual_seed(42 if not baseline else 43)
        model=Autoencoder(vocab_size=3,sequence_length=5,embedding_dim=32,hidden_size=128,latent_dim=32)
        payload=dict(model_type="ae",model_hparams=model.hparams,state_dict=model.state_dict(),
            vocab=self.pipeline.vocab,data_contract=self.pipeline.contract(),epoch=8 if baseline else 2,
            global_updates=8,checkpoint_role="best_inference_only" if baseline else "normal_only_ae_best_inference",
            training_config=dict(from_scratch=True,normal_class=None if baseline else self.normal["normal_class"],smoke_test=False),
            normal_definition=self.normal,validation_partition=self.partition,
            train_input_guard=self.guard["metadata"],
            normal_only_protocol=dict(from_scratch=True,normal_class=self.normal["normal_class"],role_digest=self.partition["role_digest"],
                                      experiment_scope="normal_only_training"),source_checkpoint=None,
            optimizer_state_dict={"must_never_restore":True})
        path=self.root/name
        torch.save(payload,path)
        return path
    def run_small(self,name="evaluate",**kwargs):
        return evaluate.run_normal_evaluation(self.pipeline,self.checkpoint,self.root/name,
            config=self.config,baseline_checkpoint=kwargs.pop("baseline_checkpoint",self.baseline),execute=True,**kwargs)

    def test_ap_pr_auc_and_prevalence_are_distinct_with_ties(self):
        result=evaluate.complete_detection_metrics([1,1,1,1],[1,1,1,0],1)
        self.assertEqual(result["roc_auc"],.5)
        self.assertEqual(result["average_precision"],.75)
        self.assertEqual(result["pr_auc_trapezoidal"],.875)
        self.assertEqual(result["constant_score_average_precision"],.75)
        self.assertEqual(result["confusion"],dict(tp=0,fp=0,tn=1,fn=3))

    def test_perfect_ranking_does_not_flip_score_direction(self):
        good=evaluate.complete_detection_metrics([.1,.2,.8,.9],[0,0,1,1],.5)
        bad=evaluate.complete_detection_metrics([.9,.8,.2,.1],[0,0,1,1],.5)
        self.assertEqual(good["roc_auc"],1)
        self.assertEqual(good["average_precision"],1)
        self.assertEqual(good["pr_auc_trapezoidal"],1)
        self.assertEqual(bad["roc_auc"],0)
        self.assertEqual(bad["recall"],0)
        self.assertTrue(bad["score_direction_fixed"])

    def test_single_class_or_empty_metrics_remain_explicitly_undefined(self):
        result=evaluate.complete_detection_metrics([1,2],[0,0])
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["pr_auc_trapezoidal"])
        self.assertIsNone(result["average_precision"])
        self.assertIsNone(evaluate.complete_detection_metrics([],[])["anomaly_prevalence"])

    def test_dry_plan_reads_neither_checkpoint_nor_data_and_creates_nothing(self):
        with patch.object(torch,"load",side_effect=AssertionError("dry checkpoint")), \
             patch.object(self.pipeline,"_iter_rows_batches",side_effect=AssertionError("dry data")), \
             patch.object(torch.optim,"Adam",side_effect=AssertionError("optimizer")):
            plan=evaluate.run_normal_evaluation(self.pipeline,self.root/"missing.pt",self.root/"dry")
        self.assertFalse(plan["execute"])
        self.assertFalse((self.root/"dry").exists())

    def test_test_requires_a_frozen_artifact_and_existing_outputs_are_refused(self):
        with self.assertRaises(ValueError):
            evaluate.run_normal_evaluation(self.pipeline,self.checkpoint,self.root/"test",stage="test")
        existing=self.root/"old"
        existing.mkdir()
        (existing/"keep.txt").write_text("old")
        with self.assertRaises(FileExistsError):evaluate.run_normal_evaluation(self.pipeline,self.checkpoint,existing)
        self.assertEqual((existing/"keep.txt").read_text(),"old")

    def test_paired_evaluation_freezes_before_mixed_forward_and_preserves_inputs_rng(self):
        source_hash={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.checkpoint,self.baseline)}
        torch_state=torch.get_rng_state().clone()
        python_state=random.getstate()
        original=evaluate._score_population
        observations=[]
        def guarded(pipeline,model,entries,split,normal,cfg,deadline):
            if any(entry["label"]!=normal for entry in entries):
                path=self.root/"evaluate/frozen_protocol.json"
                self.assertTrue(path.is_file(),"freeze must precede mixed-score forward")
                frozen,_=evaluate.read_frozen_protocol(path)
                self.assertEqual(frozen["primary_score"],"brier_mse")
                observations.append(path.read_bytes())
            return original(pipeline,model,entries,split,normal,cfg,deadline)
        with patch.object(evaluate,"_score_population",side_effect=guarded), \
             patch.object(torch.optim,"Adam",side_effect=AssertionError("evaluation restored Adam")):
            result=self.run_small()
        self.assertEqual(result["status"],"completed",result.get("reason"))
        self.assertTrue(result["paired_best8"] and result["test_ready"] and observations)
        self.assertNotIn("test",self.pipeline.read_splits)
        self.assertTrue(result["model_unchanged"] and result["checkpoint_sources_unchanged"])
        self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        self.assertEqual(python_state,random.getstate())
        for path,digest in source_hash.items():self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
        with (self.root/"evaluate/scores.csv").open(newline="",encoding="utf-8") as stream:
            rows=list(csv.DictReader(stream))
        for population in ("calibration","evaluation"):
            left=[(r["row_index"],r["row_sha256"]) for r in rows if r["model"]=="normal_only" and r["population"]==population]
            right=[(r["row_index"],r["row_sha256"]) for r in rows if r["model"]=="best8" and r["population"]==population]
            self.assertEqual(left,right)
        encoded_speakers={s for _,speakers in self.pipeline.encodings for s in speakers}
        self.assertFalse(encoded_speakers & set(self.partition["roles"]["model_selection"]))
        json.dumps(result,allow_nan=False)

    def test_test_stage_never_refits_or_reads_calibration_and_uses_frozen_thresholds(self):
        first=self.run_small()
        self.assertEqual(first["status"],"completed",first.get("reason"))
        artifact=self.root/"evaluate/frozen_protocol.json"
        before=artifact.read_bytes()
        self.pipeline.allow_test=True
        self.pipeline.read_splits.clear()
        with patch.object(evaluate,"conformal_threshold",side_effect=AssertionError("test refitted threshold")), \
             patch.object(torch.optim,"Adam",side_effect=AssertionError("test optimizer")):
            final=self.run_small("test",stage="test",frozen_protocol=artifact)
        self.assertEqual(final["status"],"completed",final.get("reason"))
        self.assertEqual(set(self.pipeline.read_splits),{"train","test"})
        self.assertEqual(artifact.read_bytes(),before)
        for name in ("normal_only","best8"):
            for score in ("brier_mse","nll_nats_per_sequence"):
                self.assertEqual(final["metrics"][name][score]["threshold"],first["metrics"][name][score]["threshold"])

    def test_test_without_config_uses_sealed_settings_instead_of_defaults(self):
        self.assertEqual(self.run_small()["status"],"completed")
        self.pipeline.allow_test=True
        final=evaluate.run_normal_evaluation(self.pipeline,self.checkpoint,self.root/"sealed_config_test",
            baseline_checkpoint=self.baseline,stage="test",frozen_protocol=self.root/"evaluate/frozen_protocol.json",
            execute=True)
        self.assertEqual(final["status"],"completed",final.get("reason"))
        self.assertEqual(final["effective_config"]["alpha"],.2)
        self.assertEqual(final["effective_config"]["max_seconds"],60)

    def test_semantically_invalid_threshold_is_rejected_before_test_even_with_a_new_sidecar(self):
        self.assertEqual(self.run_small()["status"],"completed")
        path=self.root/"evaluate/frozen_protocol.json"
        payload=json.loads(path.read_text())
        payload["models"]["normal_only"]["thresholds"]["brier_mse"]["threshold"]="invalid"
        path.write_text(json.dumps(payload),encoding="utf-8")
        path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest(),encoding="ascii")
        self.pipeline.read_splits.clear()
        result=self.run_small("bad_threshold",stage="test",frozen_protocol=path)
        self.assertEqual(result["status"],"failed")
        self.assertNotIn("test",self.pipeline.read_splits)

    def test_tampered_protocol_or_changed_model_is_rejected_before_test(self):
        self.assertEqual(self.run_small()["status"],"completed")
        artifact=self.root/"evaluate/frozen_protocol.json"
        artifact.write_text(artifact.read_text()+" ",encoding="utf-8")
        self.pipeline.read_splits.clear()
        final=self.run_small("tampered",stage="test",frozen_protocol=artifact)
        self.assertEqual(final["status"],"failed")
        self.assertIn("SHA-256",final["reason"])
        self.assertNotIn("test",self.pipeline.read_splits)

    def test_changed_checkpoint_cannot_be_substituted_after_freezing(self):
        self.assertEqual(self.run_small()["status"],"completed")
        artifact=self.root/"evaluate/frozen_protocol.json"
        payload=torch.load(self.checkpoint,weights_only=True)
        name=next(iter(payload["state_dict"]))
        payload["state_dict"][name]=payload["state_dict"][name]+.001
        torch.save(payload,self.checkpoint)
        self.pipeline.read_splits.clear()
        result=self.run_small("changed_model",stage="test",frozen_protocol=artifact)
        self.assertEqual(result["status"],"failed")
        self.assertIn("selected model changed",result["reason"])
        self.assertNotIn("test",self.pipeline.read_splits)

    def test_normal_train_input_duplicates_are_quarantined_before_calibration_and_evaluation(self):
        for role in ("calibration","evaluation"):
            speaker=self.partition["roles"][role][0]
            self.pipeline.rows["val"].append(dict(character_id=speaker,x="ababa",y="a",y_id=1))
        result=self.run_small("quarantine")
        self.assertEqual(result["status"],"completed",result.get("reason"))
        frozen,_=evaluate.read_frozen_protocol(self.root/"quarantine/frozen_protocol.json")
        normal_hash=hashlib.sha256("ababa".encode()).hexdigest()
        for role in ("calibration","evaluation"):
            self.assertNotIn(normal_hash,{row["input_sha256"] for row in frozen["selected_rows"][role]})

    def test_paired_baseline_quarantines_other_class_train_inputs_from_both_models_before_forward(self):
        overlap="bbabb"
        # Only a non-normal training row changes: normal-only model guard is
        # unchanged, while best8's all-class exposure must still be quarantined.
        self.pipeline.rows["train"][12]["x"]=overlap
        evaluation_speaker=self.partition["roles"]["evaluation"][0]
        for row in self.pipeline.rows["val"]:
            if row["character_id"]==evaluation_speaker and row["y_id"]==2:row["x"]=overlap
        original=evaluate._score_population
        def guarded(pipeline,model,entries,*args):
            self.assertTrue(all(entry["row"]["x"]!=overlap for entry in entries))
            return original(pipeline,model,entries,*args)
        with patch.object(evaluate,"_score_population",side_effect=guarded):
            result=self.run_small("paired_quarantine")
        self.assertEqual(result["status"],"completed",result.get("reason"))
        quarantine=result["paired_input_quarantine"]
        self.assertTrue(quarantine["applied"])
        self.assertEqual(quarantine["matching_normal_train_rows"],0)
        self.assertEqual(quarantine["matching_other_class_train_rows"],1)
        self.assertEqual(quarantine["matched_unique_inputs"],1)
        self.assertGreater(quarantine["roles"]["evaluation"]["excluded_rows"],0)
        frozen,_=evaluate.read_frozen_protocol(self.root/"paired_quarantine/frozen_protocol.json")
        dirty_hash=hashlib.sha256(overlap.encode()).hexdigest()
        self.assertNotIn(dirty_hash,{row["input_sha256"] for row in frozen["selected_rows"]["evaluation"]})
        self.assertEqual(frozen["paired_input_quarantine"],quarantine)

    def test_paired_test_quarantines_all_class_train_inputs_without_reading_val(self):
        self.pipeline.rows["train"][12]["x"]="bbaba"
        first=self.run_small()
        self.assertEqual(first["status"],"completed",first.get("reason"))
        self.pipeline.allow_test=True
        self.pipeline.read_splits.clear()
        result=self.run_small("paired_test_quarantine",stage="test",frozen_protocol=self.root/"evaluate/frozen_protocol.json")
        self.assertEqual(result["status"],"completed",result.get("reason"))
        self.assertEqual(set(self.pipeline.read_splits),{"train","test"})
        self.assertGreater(result["paired_input_quarantine"]["roles"]["test"]["excluded_rows"],0)
        self.assertEqual(result["paired_input_quarantine"]["matched_unique_inputs"],1)
        with (self.root/"paired_test_quarantine/scores.csv").open(newline="",encoding="utf-8") as stream:
            scored=list(csv.DictReader(stream))
        expected={(row["row_index"],row["row_sha256"]) for row in scored if row["model"]=="normal_only"}
        actual={(row["row_index"],row["row_sha256"]) for row in scored if row["model"]=="best8"}
        self.assertEqual(actual,expected)

    def test_without_best8_other_class_training_inputs_remain_valid_novelty_examples(self):
        self.pipeline.rows["train"][12]["x"]="baZab"
        result=self.run_small("normal_only_view",baseline_checkpoint=None)
        self.assertEqual(result["status"],"completed",result.get("reason"))
        self.assertFalse(result["paired_input_quarantine"]["applied"])
        self.assertGreater(result["metrics"]["normal_only"]["brier_mse"]["positives"],0)

    def test_modified_normal_train_inputs_cannot_open_sealed_test(self):
        self.assertEqual(self.run_small()["status"],"completed")
        self.pipeline.rows["train"][0]["x"]="bbbbb"
        self.pipeline.read_splits.clear()
        result=self.run_small("changed_train",stage="test",frozen_protocol=self.root/"evaluate/frozen_protocol.json")
        self.assertEqual(result["status"],"failed")
        self.assertIn("input hashes differ",result["reason"])
        self.assertNotIn("test",self.pipeline.read_splits)

    def test_insufficient_calibration_is_explicit_and_cannot_open_test(self):
        self.config["alpha"]=.01
        first=self.run_small()
        self.assertEqual(first["status"],"completed",first.get("reason"))
        self.assertFalse(first["test_ready"])
        self.assertEqual(first["calibration_status"],"insufficient_normal_calibration")
        self.pipeline.read_splits.clear()
        result=self.run_small("no_threshold",stage="test",frozen_protocol=self.root/"evaluate/frozen_protocol.json")
        self.assertEqual(result["status"],"failed")
        self.assertNotIn("test",self.pipeline.read_splits)

    def test_untrained_smoke_or_continued_candidates_are_not_final_models(self):
        for i,edit in enumerate((lambda p:p.update(global_updates=0),
                                lambda p:p["training_config"].update(smoke_test=True),
                                lambda p:p["training_config"].update(from_scratch=False),
                                lambda p:p.update(source_checkpoint="historical_best8.pt"))):
            payload=torch.load(self.checkpoint,weights_only=True)
            edit(payload)
            path=self.root/f"invalid_{i}.pt"
            torch.save(payload,path)
            result=evaluate.run_normal_evaluation(self.pipeline,path,self.root/f"invalid_{i}",config=self.config,execute=True)
            self.assertEqual(result["status"],"failed")
            self.assertFalse((self.root/f"invalid_{i}/frozen_protocol.json").exists())

    def test_wrong_role_partition_cannot_reuse_training_selection_for_calibration(self):
        payload=torch.load(self.checkpoint,weights_only=True)
        payload["validation_partition"]["role_digest"]="bad"
        torch.save(payload,self.checkpoint)
        result=self.run_small("wrong_partition")
        self.assertEqual(result["status"],"failed")
        self.assertFalse((self.root/"wrong_partition/frozen_protocol.json").exists())

    def test_invalid_settings_cannot_enable_training_or_change_budgets_unboundedly(self):
        for settings in ({"epochs":30},{"max_calibration":8192,"max_evaluation":16384},
                         {"max_test":16385},{"alpha":float("nan")},{"primary_score":"negative_nll"}):
            with self.assertRaises(ValueError):evaluate.validate_config(settings)


if __name__=="__main__":unittest.main()
