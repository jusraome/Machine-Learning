"""Fast standard-library checks of Colab project transport and CLI wiring."""
import tempfile
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import Mock, patch
import zipfile

from src.colab_utils import (build_training_command, export_project_zip, extract_project_zip,
                             plot_history, stage_processed_parquet)


class ColabUtilsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_zip(self, extra=None):
        archive_path = self.root / "project.zip"
        required = ("Workshop_4/src/colab_utils.py", "Workshop_4/src/train_ae.py",
                    "Workshop_4/config.json", "Workshop_4/config/experiments.yaml",
                    "Workshop_1/data/vocab.json", "Workshop_1/data/splits.json")
        with zipfile.ZipFile(archive_path, "w") as archive:
            for name in required:
                archive.writestr("parent/" + name, "content")
            archive.writestr("parent/Workshop_2/src/model.py", "must not extract")
            archive.writestr("parent/Workshop_4/checkpoints/best.pt", "must not extract")
            if extra:
                archive.writestr(*extra)
        return archive_path

    def test_zip_selects_only_workshop4_and_immutable_metadata(self):
        output = self.root / "extracted"
        zip_path = self.make_zip()
        workshop = extract_project_zip(zip_path, output)
        self.assertEqual(workshop, output / "Workshop_4")
        self.assertTrue((workshop / "src/train_ae.py").is_file())
        self.assertFalse((output / "Workshop_2").exists())
        self.assertFalse((workshop / "checkpoints").exists())
        self.assertEqual(extract_project_zip(zip_path, output), workshop)
        (workshop / "src/train_ae.py").write_text("different", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            extract_project_zip(zip_path, output)

    def test_zip_rejects_traversal_before_any_extraction(self):
        zip_path = self.make_zip(("../escape.py", "bad"))
        output = self.root / "extracted"
        with self.assertRaises(ValueError):
            extract_project_zip(zip_path, output)
        self.assertFalse(output.exists())

    def test_export_roundtrip_excludes_data_and_environment(self):
        source = self.root / "source"
        workshop = extract_project_zip(self.make_zip(), source)
        for relative in ("runs/history.csv", ".venv/ignored.py", "exports/old.json",
                         "data/train.parquet", "checkpoints/ae_best.pt"):
            artifact = workshop / relative
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("must not export", encoding="utf-8")
        (source / "Workshop_1/data/train.parquet").write_text("must not export", encoding="utf-8")
        archive_path = export_project_zip(workshop, self.root / "export.zip")
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            self.assertEqual(len(names), 6)
            self.assertTrue(all(name.startswith("Workshop_4/") or name in (
                "Workshop_1/data/vocab.json", "Workshop_1/data/splits.json") for name in names))
        restored = extract_project_zip(archive_path, self.root / "restored")
        self.assertEqual((restored / "src/train_ae.py").read_bytes(),
                         (workshop / "src/train_ae.py").read_bytes())

    def test_stage_train_val_without_test_or_source_changes(self):
        source, destination = self.root / "drive", self.root / "local"
        source.mkdir()
        for name in ("train.parquet", "val.parquet", "test.parquet"):
            (source / name).write_bytes(name.encode())
        self.assertEqual(stage_processed_parquet(source), source)
        self.assertEqual(stage_processed_parquet(source, destination), destination)
        self.assertFalse((destination / "test.parquet").exists())
        self.assertEqual((source / "train.parquet").read_bytes(), b"train.parquet")
        self.assertEqual(stage_processed_parquet(source, destination), destination)
        (destination / "train.parquet").write_bytes(b"changed")
        with self.assertRaises(FileExistsError):
            stage_processed_parquet(source, destination)

    def test_command_is_argv_and_overrides_do_not_edit_sources(self):
        command = build_training_command(self.root, self.root / "Drive with spaces",
                                         self.root / "data", experiments=["ae_z32", "vae_z32_b1"],
                                         resume=True, dry_run=True, latent_dim=64, beta=0.5)
        self.assertEqual(command.count("--experiment"), 2)
        self.assertIn("--dry-run", command)
        self.assertEqual(command[command.index("--resume") + 1], "auto")
        self.assertEqual(command[command.index("--latent-dim") + 1], "64")
        self.assertFalse((self.root / "config").exists())
        with self.assertRaises(ValueError):
            build_training_command(self.root, self.root, self.root, run_all=True, experiments=["ae_z32"])
        with self.assertRaises(ValueError):
            build_training_command(self.root, self.root, self.root, run_all=True, epochs=2)

    def test_plot_uses_actual_history_values_and_persistent_output_path(self):
        run = self.root / "ae_z32"
        run.mkdir()
        (run / "history.csv").write_text(
            "epoch,train_reconstruction,train_kl,train_total,val_reconstruction,val_kl,val_total\n"
            "1,10,0,10,12,0,12\n2,9,0,9,11,0,11\n", encoding="utf-8")
        figure, axes = Mock(), [Mock(), Mock(), Mock()]
        pyplot = ModuleType("matplotlib.pyplot")
        pyplot.subplots = Mock(return_value=(figure, axes))
        matplotlib = ModuleType("matplotlib")
        matplotlib.pyplot = pyplot
        with patch.dict("sys.modules", {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}):
            figures = plot_history(self.root, experiments=["ae_z32"])
        self.assertEqual(figures, [figure])
        axes[0].plot.assert_any_call([1, 2], [10.0, 9.0], label="train")
        axes[0].plot.assert_any_call([1, 2], [12.0, 11.0], label="val")
        axes[1].plot.assert_any_call([1, 2], [0.0, 0.0], label="val")
        figure.savefig.assert_called_once_with(run / "training_curves.png", dpi=150)


if __name__ == "__main__":
    unittest.main()
