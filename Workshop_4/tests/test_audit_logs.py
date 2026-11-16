"""Small copied-log audit fixtures; no training or production metrics."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import audit_logs as module


class LogAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="workshop4-log-audit-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_known_history_and_separate_session_norm_statistics(self):
        history = self.root / "history.csv"
        history.write_text("epoch,train_total,val_total,train_clipped_batches,train_skipped_batches\n1,10,12,2,0\n2,8,9,1,0\n", encoding="utf-8")
        norms = self.root / "gradient_norms.csv"
        norms.write_text("session,epoch,batch_index,norm_before,norm_after,clipped\na,2,1,10,5,True\na,2,2,2,2,False\nb,2,1,20,5,True\n", encoding="utf-8")
        events = self.root / "events.jsonl"
        events.write_text(json.dumps({"session": "b", "epoch": 2, "reason": "large_finite_gradient_clipped", "action": "accepted_after_verified_clipping"}) + "\n", encoding="utf-8")
        inputs = [history, norms, events]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
        report = module.write_audit(self.root / "new.json", history=history, gradient_norms=norms, events=events)
        self.assertEqual(report["history"]["best_epoch_by_val_total"], 2)
        self.assertEqual(report["history"]["last_to_first_val_ratio"], 0.75)
        self.assertIsNone(report["history"]["rows"][0]["train_accepted_batches"])
        a, b = report["gradient_norms"]
        self.assertEqual(a["norm_before_mean"], 6.0)
        self.assertEqual(a["norm_after_max"], 5.0)
        self.assertEqual(a["clipped_ratio_logged_rows"], 0.5)
        self.assertEqual(b["clipped_ratio_logged_rows"], 1.0)
        self.assertIn("logged_rows_only", a["coverage"])
        self.assertEqual(report["events"][0]["reasons"]["large_finite_gradient_clipped"], 1)
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_nonfinite_json_constants_and_csv_metrics_are_rejected(self):
        metadata = self.root / "metadata.json"
        for invalid in ("NaN", "Infinity", "-Infinity", "1e999"):
            with self.subTest(invalid=invalid):
                metadata.write_text('{"metric":' + invalid + '}', encoding="utf-8")
                with self.assertRaises(ValueError):
                    module.audit_logs(metadata=metadata)
        history = self.root / "history.csv"
        history.write_text("epoch,val_total\n1,nan\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            module.audit_logs(history=history)

    def test_output_collision_refuses_to_overwrite_any_file(self):
        history = self.root / "history.csv"
        text = "epoch,val_total\n1,2\n"
        history.write_text(text, encoding="utf-8")
        with self.assertRaises(FileExistsError):
            module.write_audit(history, history=history)
        existing = self.root / "existing.json"
        existing.write_text("existing", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            module.write_audit(existing, history=history)
        self.assertEqual(history.read_text(encoding="utf-8"), text)
        self.assertEqual(existing.read_text(encoding="utf-8"), "existing")

    def test_source_change_during_audit_is_detected(self):
        history = self.root / "history.csv"
        history.write_text("epoch,val_total\n1,2\n", encoding="utf-8")
        original_reader = module._read_history
        def changed(path):
            result = original_reader(path)
            path.write_text("epoch,val_total\n1,3\n", encoding="utf-8")
            return result
        with patch.object(module, "_read_history", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                module.audit_logs(history=history)


if __name__ == "__main__":
    unittest.main()
