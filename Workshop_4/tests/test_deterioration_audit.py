"""Synthetic copied-log evidence; no model loading or training is performed."""
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from src import deterioration_audit as module


class DeteriorationAuditTests(unittest.TestCase):
    def setUp(self):
        directory = Path(__file__).resolve().parents[1]/"tmp"
        directory.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="phase4-log-tests-",dir=directory)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self,name,text):
        path = self.root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text,encoding="utf-8")
        return path

    def analyze(self,**kwargs):
        return module.analyze_deterioration_logs(reported_history=None,**kwargs)

    def fixture(self,folder="historical"):
        history = self.write(folder+"/history.csv",
            "epoch,train_reconstruction,val_reconstruction,train_accepted_batches\n"+
            "8,164.7,168.67,2\n9,163,169,2\n10,160,171,2\n11,159.07,172.69,2\n"
            "12,709.8,1212.92,2\n13,1043.41,336.01,2\n14,882.82,1131.97,2\n15,1490.27,1333.13,2\n")
        gradients = self.write(folder+"/gradient_norms.csv",
            "session,epoch,batch_index,norm_before,norm_after,scale,clipped\n"
            "a,12,1,10,5,0.5,True\na,12,2,2,2,1,False\nb,12,1,1e20,5,5e-20,True\n")
        events = self.write(folder+"/nonfinite_events.jsonl","".join(json.dumps(row)+"\n" for row in (
            {"session":"a","epoch":11,"batch_index":2,"reason":"skipped_fraction_exceeded","action":"abort"},
            {"session":"a","epoch":12,"batch_index":1,"reason":"large_finite_gradient_clipped","action":"accepted_after_verified_clipping"},
            {"session":"a","epoch":12,"batch_index":2,"reason":"nonfinite_gradient_elements","action":"skip_batch"},
            {"session":"b","epoch":12,"batch_index":99,"reason":"nonfinite_model_or_adam","action":"abort"})))
        metadata = self.write(folder+"/metadata.json",json.dumps({"model_type":"ae","runtime":{"gradient_log_every":1}}))
        return dict(history=history,gradient_norms=gradients,events=events,metadata=metadata)

    def test_confirmed_checkpoint_comparison_is_reported_not_recomputed(self):
        result = module.confirmed_checkpoint_comparison()
        self.assertIn("user_confirmed",result["provenance"])
        self.assertIn("not full validation",result["validation_scope"])
        self.assertAlmostEqual(result["reconstruction_increase_nats_per_sequence"],89.729666)
        self.assertAlmostEqual(result["swap_damage_retained_fraction"],3.640303/359.958162)
        self.assertGreater(result["rows"][0]["gain_over_marginal_nats_per_sequence"],90)
        self.assertLess(result["rows"][1]["gain_over_marginal_nats_per_sequence"],3)
        result["rows"][0]["epoch"] = 100
        self.assertEqual(module.CONFIRMED_CHECKPOINTS[0]["epoch"],8)

    def test_epoch_window_and_sessions_are_not_mixed(self):
        sources = self.fixture()
        before = {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in sources.items()}
        report = self.analyze(**sources)
        history = report["history"]
        self.assertEqual(history["missing_epochs"],[])
        self.assertEqual(history["large_adjacent_increases"][0]["from_epoch"],11)
        self.assertEqual(history["large_adjacent_increases"][0]["to_epoch"],12)
        self.assertAlmostEqual(history["rows"][4]["val_relative_to_epoch8"],1212.92/168.67-1)
        a,b = report["gradient_norms"]
        self.assertEqual((a["session"],b["session"]),("a","b"))
        self.assertEqual(a["before_mean"],6)
        self.assertEqual(a["clipped_ratio_logged_rows"],.5)
        self.assertEqual(b["clipped_ratio_logged_rows"],1)
        self.assertEqual(b["fp32_square_reduction_risk_rows"],1)
        self.assertEqual(a["norm_before_quantiles"]["0.5"],6)
        self.assertEqual(report["events"]["periods"],{"before_epoch12":1,"epoch12_onward":3})
        self.assertEqual(report["events"]["groups"][1]["categories"]["finite_gradient_clipping"],1)
        self.assertEqual(report["cross_log_alignment"][0]["overlapping_logged_event_batch_indices"],2)
        self.assertEqual(report["cross_log_alignment"][1]["overlapping_logged_event_batch_indices"],0)
        self.assertTrue(report["sources_unchanged"])
        self.assertFalse(report["root_cause_established"])
        for key,path in sources.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before[key])

    def test_missing_is_distinct_from_logged_zero(self):
        missing = self.analyze(events=self.root/"missing.jsonl")
        self.assertEqual(missing["sources"]["events"]["status"],"missing")
        self.assertIsNone(missing["events"])
        self.assertIsNone(missing["gradient_norms"])
        empty = self.analyze(events=self.write("empty.jsonl",""),
            gradient_norms=self.write("empty.csv","session,epoch,batch_index,norm_before,norm_after,clipped\n"))
        self.assertEqual(empty["sources"]["events"]["status"],"present_empty")
        self.assertEqual(empty["events"]["total_logged_events"],0)
        self.assertEqual(empty["sources"]["gradient_norms"]["status"],"present_empty")
        self.assertEqual(empty["gradient_norms"],[])

    def test_smoke_rows_are_not_claimed_as_epoch8_to15_evidence(self):
        history = self.write("smoke.csv","epoch,val_total\n1,10\n2,9\n")
        gradients = self.write("smoke_grad.csv","session,epoch,batch_index,norm_before,norm_after,clipped\nx,1,1,10,5,True\n")
        report = self.analyze(history=history,gradient_norms=gradients)
        self.assertEqual(report["history"]["rows"],[])
        self.assertEqual(report["history"]["missing_epochs"],list(range(8,16)))
        self.assertEqual(report["sources"]["gradient_norms"]["status"],"no_rows_in_epoch_window")

    def test_earlier_user_report_fallback_keeps_provenance_and_missing_epochs(self):
        reported = self.write("reported.json",json.dumps({"rows":[{"epoch":8,"train_loss":164.7,"val_loss":168.67},
            {"epoch":11,"train_loss":159.07,"val_loss":172.69},{"epoch":12,"train_loss":709.8,"val_loss":1212.92}]}))
        report = module.analyze_deterioration_logs(reported_history=reported)
        self.assertIn("earlier_user_reported",report["history"]["provenance"])
        self.assertEqual(report["sources"]["history"]["status"],"not_provided")
        self.assertEqual(report["history"]["missing_epochs"],[9,10,13,14,15])
        self.assertFalse(report["history"]["transitions"][0]["adjacent_epochs"])
        self.assertEqual(report["history"]["rows"][0]["val_nats_per_sequence"],168.67)
        self.assertEqual(report["confirmed_checkpoint_comparison"]["rows"][0]["reconstruction_nats_per_sequence"],160.400189)

    def test_duplicate_history_does_not_select_an_ambiguous_epoch8_reference(self):
        history = self.write("duplicate.csv","epoch,val_reconstruction\n8,100\n8,110\n9,120\n")
        report = self.analyze(history=history)["history"]
        self.assertEqual(report["duplicate_epochs"],[8])
        self.assertFalse(report["epoch8_reference_available"])
        self.assertIsNone(report["rows"][2]["val_relative_to_epoch8"])

    def test_nonfinite_csv_and_json_are_rejected_without_output(self):
        for token in ("nan","inf","1e999"):
            path = self.write("bad.csv",f"epoch,val_reconstruction\n12,{token}\n")
            with self.assertRaises(ValueError):
                self.analyze(history=path)
        for token in ("NaN","Infinity","1e999"):
            path = self.write("bad.jsonl",'{"epoch":12,"metric":'+token+'}\n')
            with self.assertRaises(ValueError):
                self.analyze(events=path)
        path = self.write("badnorm.csv","epoch,batch_index,norm_before,norm_after,clipped\n12,1,nan,5,True\n")
        with self.assertRaises(ValueError):
            self.analyze(gradient_norms=path)

    def test_reservoir_is_bounded_deterministic_and_does_not_advance_global_rng(self):
        path = self.write("many.csv","session,epoch,batch_index,norm_before,norm_after,scale,clipped\n"+
            "".join(f"a,12,{i},{i},5,0.1,True\n" for i in range(1,5001)))
        state = random.getstate()
        first = self.analyze(gradient_norms=path)["gradient_norms"][0]
        self.assertEqual(random.getstate(),state)
        second = self.analyze(gradient_norms=path)["gradient_norms"][0]
        self.assertEqual(first["norm_before_quantiles"],second["norm_before_quantiles"])
        self.assertEqual(first["quantile_sample_size"],2048)
        self.assertFalse(first["quantiles_exact"])
        self.assertEqual(first["distinct_logged_batches"],5000)

    def test_repeated_batch_indices_and_unknown_sessions_are_explicit(self):
        gradients = self.write("g.csv","epoch,batch_index,norm_before,norm_after,clipped\n12,1,10,5,True\n12,1,20,5,True\n")
        events = self.write("e.jsonl",'{"epoch":12,"batch_index":1,"reason":"nonfinite_gradient"}\n')
        report = self.analyze(gradient_norms=gradients,events=events)
        self.assertEqual(report["gradient_norms"][0]["duplicate_batch_rows"],1)
        self.assertFalse(report["cross_log_alignment"][0]["session_alignment_recorded"])
        self.assertIsNone(report["cross_log_alignment"][0]["overlapping_logged_event_batch_indices"])

    def test_source_change_during_second_stream_is_detected(self):
        path = self.write("g.csv","epoch,batch_index,norm_before,norm_after,clipped\n12,1,10,5,True\n")
        original = module._gradient_details
        def changed(*args):
            report = original(*args)
            path.write_text(path.read_text()+"12,2,20,5,True\n")
            return report
        with patch.object(module,"_gradient_details",side_effect=changed):
            with self.assertRaisesRegex(RuntimeError,"Source changed"):
                self.analyze(gradient_norms=path)

    def test_dry_plan_has_no_analysis_and_new_outputs_are_exclusive(self):
        destination = self.root/"dry"
        with patch.object(module,"analyze_deterioration_logs",side_effect=AssertionError("analysis in dry plan")):
            plan = module.write_deterioration_audit(destination)
        self.assertEqual(plan["optimizer_steps"],0)
        self.assertFalse(destination.exists())
        report = module.write_deterioration_audit(destination,execute=True,reported_history=None)
        self.assertTrue((destination/"report.json").is_file())
        self.assertFalse(report["root_cause_established"])
        with self.assertRaises(FileExistsError):
            module.write_deterioration_audit(destination,execute=True)

    def test_metadata_is_validated_and_included_as_a_separate_source(self):
        metadata = self.write("meta.json",'{"runtime":{"gradient_log_every":12}}')
        report = self.analyze(metadata=metadata)
        self.assertEqual(report["metadata"]["runtime"]["gradient_log_every"],12)
        self.assertIsNone(report["history"])
        metadata.write_text('[1,2]')
        with self.assertRaises(ValueError):
            self.analyze(metadata=metadata)

    def test_stable_log_bundle_contains_exact_logs_no_checkpoints(self):
        files = self.fixture()
        source = files["history"].parent
        self.write("historical/checkpoints/best.pt","must not read or copy")
        before = {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in files.items()}
        result = module.stage_log_bundle(source,self.root/"copies")
        self.assertEqual(result["status"],"validated_stable_log_copies")
        self.assertEqual(set(result["copied_files"]),{"history","gradient_norms","events","metadata"})
        with zipfile.ZipFile(result["zip_path"]) as archive:
            self.assertEqual(set(archive.namelist()),{path.name for path in files.values()})
        for key,path in files.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before[key])
            self.assertEqual(Path(result["copied_files"][key]).read_bytes(),path.read_bytes())
        with self.assertRaises(FileExistsError):
            module.stage_log_bundle(source,self.root/"copies")

    def test_log_bundle_rejects_historical_destinations_missing_logs_and_existing_zip(self):
        files = self.fixture()
        source = files["history"].parent
        with self.assertRaises(ValueError):
            module.stage_log_bundle(source,source/"copies")
        with self.assertRaises(FileNotFoundError):
            module.stage_log_bundle(self.root/"not_here",self.root/"new")
        zip_path = self.write("existing.zip","previous bundle")
        with self.assertRaises(FileExistsError):
            module.stage_log_bundle(source,self.root/"new",zip_path)
        self.assertEqual(zip_path.read_text(),"previous bundle")

    def test_log_bundle_rejects_sources_changing_during_copy_before_zip_creation(self):
        files = self.fixture()
        metadata = files["metadata"]
        original = module.audit_logs._fingerprint
        metadata_checks = 0
        def changing(path):
            nonlocal metadata_checks
            if Path(path) == metadata:
                metadata_checks += 1
                if metadata_checks == 2:
                    metadata.write_text('{"changed_during_copy":true}')
            return original(path)
        destination = self.root/"partial_copies"
        with patch.object(module.audit_logs,"_fingerprint",side_effect=changing):
            with self.assertRaisesRegex(RuntimeError,"changed during copy"):
                module.stage_log_bundle(metadata.parent,destination)
        self.assertFalse((destination/"historical_log_bundle.zip").exists())

    def test_finite_inputs_with_nonfinite_derived_ratios_do_not_create_outputs(self):
        history = self.write("huge_ratios.csv","epoch,val_reconstruction\n8,1e-308\n12,1e308\n")
        destination = self.root/"invalid_result"
        with self.assertRaises(ValueError):
            module.write_deterioration_audit(destination,execute=True,history=history,reported_history=None)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
