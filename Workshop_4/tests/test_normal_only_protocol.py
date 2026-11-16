"""Boundaries and complete normal-only streaming; no model training."""
import copy
import importlib.util
import random
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch","numpy"))
if AVAILABLE:
    import torch
    from src import normal_only_protocol as protocol


class Pipeline:
    def __init__(self):
        self.vocab={"itos":["<UNK>","a","b"],"stoi":{"<UNK>":0,"a":1,"b":2}}
        self.config={"window_size":5}
        self.groups={"train":{"TRAIN"},"val":{f"V{i}" for i in range(6)},"test":{"TEST"}}
        self.rows={"train":[dict(character_id="TRAIN",x="ababa" if i%3==0 else "bbbbb",
                   y="b" if i%3==0 else "a",y_id=2 if i%3==0 else 1) for i in range(17)],
                   "val":[dict(character_id=s,x="ababa",y="a" if i%2 else "b",y_id=1 if i%2 else 2)
                          for s in sorted(self.groups["val"]) for i in range(12)],
                   "test":[dict(character_id="TEST",x="ababa",y="a" if i%2 else "b",y_id=1 if i%2 else 2)
                           for i in range(12)]}
        self.seen=[]
        self.allow_test=False
        self.closed=[]

    def contract(self):
        return {"dataset":"synthetic-normal-only","window_size":5,"revision":"unit"}

    def _iter_rows_batches(self,split,batch_size=4):
        if split=="test" and not self.allow_test:
            raise AssertionError("Test reader invoked without opt-in")
        self.seen.append(split)
        try:
            for start in range(0,len(self.rows[split]),batch_size):
                yield self.rows[split][start:start+batch_size]
        finally:
            self.closed.append(split)

    def _validate_rows(self,rows,split):
        if any(row["character_id"] not in self.groups[split] for row in rows):
            raise ValueError("speaker outside split")
        labels=[self.vocab["stoi"].get(row["y"],0) for row in rows]
        if any(row["y_id"] != label or len(row["x"])!=5 for row,label in zip(rows,labels)):
            raise ValueError("invalid row")
        return [row["x"] for row in rows],labels

    def encode_tensor_rows(self,rows,split):
        texts,labels=self._validate_rows(rows,split)
        return {"x":torch.tensor([[self.vocab["stoi"].get(c,0) for c in text] for text in texts],dtype=torch.long),
                "y":torch.tensor(labels,dtype=torch.long)}


@unittest.skipUnless(AVAILABLE,"normal-only protocol requires Torch and NumPy")
class NormalOnlyProtocolTests(unittest.TestCase):
    def setUp(self):
        self.pipeline=Pipeline()
        self.partition=protocol.role_partition(self.pipeline,seed=42)

    def test_normal_definition_is_train_y_only_not_reconstruction_characters(self):
        normal=protocol.fit_normal_class(self.pipeline)
        self.assertEqual(normal["normal_class"],1)
        self.assertEqual(normal["train_label_counts"],[0,11,6])
        self.assertEqual(self.pipeline.seen,["train"])
        batches=list(protocol.iter_normal_train_batches(self.pipeline,1,batch_size=4,shuffle_buffer_batches=2))
        self.assertTrue(all((b["y"]==1).all().item() for b in batches))
        # Normal examples contain only 'b' in x, although y='a'. Filtering x
        # instead of y would incorrectly remove every one of these windows.
        self.assertTrue(all((b["x"]==2).all().item() for b in batches))

    def test_three_roles_are_deterministic_disjoint_and_exhaustive(self):
        self.assertEqual(self.partition,protocol.role_partition(self.pipeline,seed=42))
        groups=[set(self.partition["roles"][name]) for name in protocol.ROLE_NAMES]
        self.assertTrue(all(groups))
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        self.assertEqual(set.union(*groups),self.pipeline.groups["val"])
        self.assertTrue(protocol.validate_partition(self.pipeline,self.partition))
        self.assertEqual(self.pipeline.seen,[])

    def test_overlap_or_inadequate_speakers_cannot_silently_share_roles(self):
        self.pipeline.groups["val"]={"V0","V1"}
        with self.assertRaises(ValueError):protocol.role_partition(self.pipeline)
        self.pipeline.groups["val"]={"V0","V1","TRAIN"}
        with self.assertRaises(ValueError):protocol.role_partition(self.pipeline)

    def test_partition_tampering_and_contract_changes_are_rejected(self):
        broken=copy.deepcopy(self.partition)
        broken["roles"]["calibration"][0]=broken["roles"]["model_selection"][0]
        with self.assertRaises(ValueError):protocol.validate_partition(self.pipeline,broken)
        broken=copy.deepcopy(self.partition)
        broken["data_contract"]["revision"]="different"
        with self.assertRaises(ValueError):protocol.validate_partition(self.pipeline,broken)
        broken=copy.deepcopy(self.partition)
        broken["role_digest"]="0"*64
        with self.assertRaises(ValueError):protocol.validate_partition(self.pipeline,broken)

    def test_selection_and_calibration_are_normal_evaluation_is_mixed(self):
        result=protocol.prepare_role_samples(self.pipeline,self.partition,1,
            roles=protocol.ROLE_NAMES,max_per_role=100,per_speaker_cap=100,read_batch_size=7)
        for name in ("model_selection","calibration"):
            self.assertTrue(result["samples"][name])
            self.assertEqual({e["label"] for e in result["samples"][name]},{1})
            self.assertEqual(result["sampling"]["roles"][name]["anomaly_rows"],0)
        self.assertEqual({e["label"] for e in result["samples"]["evaluation"]},{1,2})
        self.assertEqual(result["sampling"]["roles"]["evaluation"]["anomaly_prevalence"],.5)
        self.assertEqual(self.pipeline.seen,["val"])

    def test_training_selection_never_retains_or_encodes_calibration_evaluation(self):
        encoded_speakers=[]
        original=self.pipeline.encode_tensor_rows
        def encode(rows,split):
            encoded_speakers.extend(row["character_id"] for row in rows)
            return original(rows,split)
        self.pipeline.encode_tensor_rows=encode
        result=protocol.prepare_role_samples(self.pipeline,self.partition,1,max_per_role=8)
        self.assertEqual(set(result["samples"]),{"model_selection"})
        self.assertEqual(encoded_speakers,[])
        batch=protocol.role_tensor(self.pipeline,result["samples"]["model_selection"])
        self.assertTrue((batch["y"]==1).all().item())
        self.assertLessEqual(set(encoded_speakers),set(self.partition["roles"]["model_selection"]))

    def test_sampling_is_deterministic_bounded_and_does_not_mutate_rows_or_rng(self):
        before=copy.deepcopy(self.pipeline.rows)
        python_state=random.getstate()
        torch_state=torch.get_rng_state().clone()
        kwargs=dict(roles=protocol.ROLE_NAMES,max_per_role=10,per_speaker_cap=4,read_batch_size=5)
        first=protocol.prepare_role_samples(self.pipeline,self.partition,1,**kwargs)
        second=protocol.prepare_role_samples(self.pipeline,self.partition,1,**kwargs)
        self.assertEqual(first,second)
        self.assertEqual(before,self.pipeline.rows)
        self.assertEqual(random.getstate(),python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(),torch_state))
        for role,entries in first["samples"].items():
            self.assertLessEqual(len(entries),10)
            self.assertTrue(all(len(e["row_sha256"])==64 and len(e["input_sha256"])==64 for e in entries))
            self.assertEqual(first["sampling"]["roles"][role]["selected_rows"],len(entries))

    def test_no_row_or_speaker_overlap_between_selected_validation_roles(self):
        result=protocol.prepare_role_samples(self.pipeline,self.partition,1,roles=protocol.ROLE_NAMES,max_per_role=100)
        row_sets=[{e["row_index"] for e in result["samples"][r]} for r in protocol.ROLE_NAMES]
        speakers=[{e["row"]["character_id"] for e in result["samples"][r]} for r in protocol.ROLE_NAMES]
        for left,right in ((0,1),(0,2),(1,2)):
            self.assertFalse(row_sets[left]&row_sets[right])
            self.assertFalse(speakers[left]&speakers[right])
        # Same text may occur under different speakers; separation does not
        # falsely claim semantic or exact-text deduplication.
        hashes=[{e["input_sha256"] for e in result["samples"][r]} for r in protocol.ROLE_NAMES]
        self.assertTrue(hashes[0]&hashes[1])
        self.assertTrue(any("deduplication" in s for s in result["sampling"]["limitations"]))

    def test_test_requires_explicit_opt_in_and_cannot_enter_calibration(self):
        with self.assertRaises(ValueError):
            protocol.prepare_role_samples(self.pipeline,self.partition,1,roles=("test",))
        self.assertEqual(self.pipeline.seen,[])
        self.pipeline.allow_test=True
        result=protocol.prepare_role_samples(self.pipeline,self.partition,1,roles=("test",),allow_test=True,max_per_role=12)
        self.assertEqual(self.pipeline.seen,["test"])
        self.assertEqual({e["label"] for e in result["samples"]["test"]},{1,2})
        self.assertFalse(result["sampling"]["roles"]["test"]["normal_only"])

    def test_complete_filtered_train_epoch_has_no_loss_duplication_or_dropped_tail(self):
        batches=list(protocol.iter_normal_train_batches(self.pipeline,1,batch_size=4,
            shuffle_buffer_batches=2,read_batch_size=3,seed=42,epoch=1))
        indices=[i for b in batches for i in b["row_indices"]]
        expected=[i for i,row in enumerate(self.pipeline.rows["train"]) if row["y_id"]==1]
        self.assertEqual(sorted(indices),expected)
        self.assertEqual(len(set(indices)),len(indices))
        self.assertEqual([len(b["x"]) for b in batches],[4,4,3])
        self.assertEqual(self.pipeline.seen,["train"])
        self.assertIn("train",self.pipeline.closed)

    def test_train_shuffle_is_local_deterministic_and_epoch_specific(self):
        python_state=random.getstate()
        torch_state=torch.get_rng_state().clone()
        def order(epoch):
            return [i for b in protocol.iter_normal_train_batches(self.pipeline,1,batch_size=4,
                shuffle_buffer_batches=3,read_batch_size=2,epoch=epoch) for i in b["row_indices"]]
        self.assertEqual(order(1),order(1))
        self.assertNotEqual(order(1),order(2))
        self.assertEqual(random.getstate(),python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(),torch_state))

    def test_expired_deadline_stops_and_closes_reader(self):
        with self.assertRaises(TimeoutError):
            list(protocol.iter_normal_train_batches(self.pipeline,1,deadline=0))
        self.assertIn("train",self.pipeline.closed)
        with self.assertRaises(TimeoutError):
            protocol.prepare_role_samples(self.pipeline,self.partition,1,deadline=0)
        self.assertIn("val",self.pipeline.closed)

    def test_empty_normal_role_is_explicit_and_limits_reject_unbounded_requests(self):
        for row in self.pipeline.rows["val"]:row["y"],row["y_id"]="b",2
        result=protocol.prepare_role_samples(self.pipeline,self.partition,1)
        self.assertEqual(result["samples"]["model_selection"],[])
        with self.assertRaises(ValueError):protocol.role_tensor(self.pipeline,[])
        for kwargs in (dict(max_per_role=True),dict(max_per_role=65537),
                       dict(roles=protocol.ROLE_NAMES,max_per_role=30000),dict(normal_class=True)):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                opts=dict(normal_class=1,**{k:v for k,v in kwargs.items() if k!="normal_class"})
                if "normal_class" in kwargs:opts["normal_class"]=kwargs["normal_class"]
                protocol.prepare_role_samples(self.pipeline,self.partition,**opts)

    def test_input_guard_hashes_only_actual_normal_training_inputs(self):
        guard=protocol.build_train_input_guard(self.pipeline,1,read_batch_size=3)
        import hashlib
        normal_hash=hashlib.sha256(b"bbbbb").hexdigest()
        anomaly_hash=hashlib.sha256(b"ababa").hexdigest()
        self.assertEqual(guard["input_hashes"],{normal_hash})
        self.assertNotIn(anomaly_hash,guard["input_hashes"])
        self.assertEqual(guard["metadata"]["normal_train_rows"],11)
        self.assertEqual(guard["metadata"]["scanned_train_rows"],17)
        self.assertEqual(guard["metadata"]["unique_inputs"],1)
        self.assertEqual(self.pipeline.seen,["train"])
        repeated=protocol.build_train_input_guard(self.pipeline,1,read_batch_size=8)
        self.assertEqual(guard,repeated)

    def test_exact_training_input_overlap_is_quarantined_before_reservoir_by_role(self):
        guard=protocol.build_train_input_guard(self.pipeline,1)
        for role in protocol.ROLE_NAMES:
            speaker=self.partition["roles"][role][0]
            normals=[r for r in self.pipeline.rows["val"] if r["character_id"]==speaker and r["y_id"]==1]
            normals[0]["x"]="bbbbb"
            if role=="evaluation":
                next(r for r in self.pipeline.rows["val"] if r["character_id"]==speaker and r["y_id"]==2)["x"]="bbbbb"
        before=copy.deepcopy(self.pipeline.rows)
        result=protocol.prepare_role_samples(self.pipeline,self.partition,1,roles=protocol.ROLE_NAMES,
            max_per_role=100,per_speaker_cap=100,exclude_input_hashes=guard["input_hashes"])
        self.assertTrue(result["sampling"]["training_input_guard_applied"])
        self.assertEqual(result["sampling"]["excluded_training_input_overlap_rows_total"],4)
        for role,expected in (("model_selection",1),("calibration",1),("evaluation",2)):
            metadata=result["sampling"]["roles"][role]
            self.assertEqual(metadata["excluded_training_input_overlap_rows"],expected)
            self.assertEqual(metadata["excluded_non_normal_rows"],12 if role!="evaluation" else 0)
            self.assertFalse({r["input_sha256"] for r in result["samples"][role]} & guard["input_hashes"])
        self.assertEqual(before,self.pipeline.rows)
        # Anomalous TRAIN x was not used to fit model weights, so it was not
        # incorrectly quarantined merely because it occurs in the raw split.
        self.assertTrue(all(e["row"]["x"]=="ababa" for entries in result["samples"].values() for e in entries))

    def test_test_quarantine_reuses_train_guard_without_touching_validation(self):
        guard=protocol.build_train_input_guard(self.pipeline,1)
        self.pipeline.allow_test=True
        self.pipeline.rows["test"][0]["x"]="bbbbb"
        result=protocol.prepare_role_samples(self.pipeline,self.partition,1,roles=("test",),allow_test=True,
            max_per_role=100,exclude_input_hashes=guard["input_hashes"])
        self.assertEqual(self.pipeline.seen,["train","test"])
        self.assertEqual(result["sampling"]["roles"]["test"]["excluded_training_input_overlap_rows"],1)
        self.assertEqual(result["sampling"]["roles"]["test"]["selected_rows"],11)
        self.assertNotIn("calibration",result["samples"])

    def test_input_guard_memory_bound_and_expiry_fail_without_truncating(self):
        first_normal=next(row for row in self.pipeline.rows["train"] if row["y_id"]==1)
        first_normal["x"]="aabaa"
        with self.assertRaises(ValueError):protocol.build_train_input_guard(self.pipeline,1,max_unique_inputs=1)
        with self.assertRaises(TimeoutError):protocol.build_train_input_guard(self.pipeline,1,deadline=0)
        self.assertTrue(all(split=="train" for split in self.pipeline.seen))


if __name__=="__main__":unittest.main()
