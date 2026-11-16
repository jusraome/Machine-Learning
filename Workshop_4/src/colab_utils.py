"""Colab transport and plotting helpers; training stays in train_ae.py.

Importing this module requires only the Python standard library. No network
requests, dependency installation, training or Drive mounting occur implicitly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import zipfile


_SKIP_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache",
               "checkpoints", "runs", "reports", "output", "tmp", "exports"}
_ALLOWED_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".md", ".txt", ".ipynb", ".pdf"}
_METADATA_PATHS = {"Workshop_1/data/vocab.json", "Workshop_1/data/splits.json"}
_REQUIRED_PATHS = {PurePosixPath(name) for name in (
    "Workshop_4/src/colab_utils.py", "Workshop_4/src/train_ae.py",
    "Workshop_4/config.json", "Workshop_4/config/experiments.yaml",
    *_METADATA_PATHS)}


def _project_relative_path(path):
    """Common export/extraction allow-list; no data/model artifacts."""
    parts = path.parts
    if any(part in _SKIP_PARTS for part in parts):
        return None
    if "Workshop_4" in parts:
        relative = PurePosixPath(*parts[parts.index("Workshop_4"):])
        if relative.suffix.lower() in _ALLOWED_SUFFIXES or relative.name == ".gitignore":
            return relative
    elif "Workshop_1" in parts:
        relative = PurePosixPath(*parts[parts.index("Workshop_1"):])
        if relative.as_posix() in _METADATA_PATHS:
            return relative
    return None


def export_project_zip(workshop_dir, destination):
    """Export code/docs and unchanged W1 metadata, without Git or dataset files."""
    workshop_dir = Path(workshop_dir).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    selected = {}
    for directory, subdirectories, files in os.walk(workshop_dir, followlinks=False):
        subdirectories[:] = sorted(name for name in subdirectories
                                   if name not in _SKIP_PARTS and not (Path(directory) / name).is_symlink())
        for name in sorted(files):
            source = Path(directory) / name
            relative = PurePosixPath("Workshop_4", *source.relative_to(workshop_dir).parts)
            if _project_relative_path(relative) is not None:
                if source.is_symlink():
                    raise ValueError(f"Project symlinks are not supported: {source}")
                selected[relative] = source
    for name in _METADATA_PATHS:
        relative = PurePosixPath(name)
        source = workshop_dir.parent.joinpath(*relative.parts)
        if source.is_symlink():
            raise ValueError(f"Metadata symlinks are not supported: {source}")
        if source.is_file():
            selected[relative] = source
    missing = _REQUIRED_PATHS.difference(selected)
    if missing:
        raise FileNotFoundError("Export is missing: " + ", ".join(sorted(map(str, missing))))
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix="workshop4-export-", suffix=".zip.tmp", dir=destination.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative, source in sorted(selected.items()):
                archive.write(source, arcname=relative.as_posix())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def extract_project_zip(zip_path, destination):
    """Extract only Workshop_4 code/config/docs and W1 split/vocab metadata.

    Accepts either a repository ZIP or a ZIP with a single parent directory.
    Refuses traversal, links, colliding names and changing existing files.
    Models, logs, Parquet, environments and other workshops are never extracted.
    Returns the absolute Workshop_4 directory under destination.
    """
    destination = Path(destination).expanduser().resolve()
    selected = {}
    with zipfile.ZipFile(zip_path) as archive:
        for item in archive.infolist():
            name = item.filename.replace("\\", "/")
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
                raise ValueError(f"Unsafe ZIP member: {item.filename}")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"ZIP symlinks are not supported: {item.filename}")
            if item.is_dir():
                continue
            relative = _project_relative_path(path)
            if relative is None:
                continue
            if relative in selected:
                raise ValueError(f"Multiple ZIP files map to {relative}")
            selected[relative] = item
        missing = _REQUIRED_PATHS.difference(selected)
        if missing:
            raise FileNotFoundError("ZIP is missing: " + ", ".join(sorted(map(str, missing))))
        if sum(item.file_size for item in selected.values()) > 256 * 1024 * 1024:
            raise ValueError("Code/metadata ZIP exceeds the 256 MiB extraction limit")
        for relative, item in selected.items():
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"Destination escapes project directory: {relative}")
            if target.exists() and target.read_bytes() != archive.read(item):
                raise FileExistsError(f"Different project file already exists: {target}; use a new destination")
        for relative, item in selected.items():
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with archive.open(item) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    return destination / "Workshop_4"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_processed_parquet(source_dir, destination=None):
    """Use Drive directly, or copy only train/val to local storage with hashes.

    test.parquet and raw data are neither read nor staged for hyperparameter
    selection. Source files are never modified. Existing differing local files
    are rejected, rather than silently mixing datasets between sessions.
    """
    source_dir = Path(source_dir).expanduser().resolve()
    for name in ("train.parquet", "val.parquet"):
        if not (source_dir / name).is_file():
            raise FileNotFoundError(source_dir / name)
    if destination is None:
        return source_dir
    destination = Path(destination).expanduser().resolve()
    if source_dir == destination:
        return source_dir
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("train.parquet", "val.parquet"):
        source, target = source_dir / name, destination / name
        expected = _sha256(source)
        if target.exists():
            if _sha256(target) != expected:
                raise FileExistsError(f"Local Parquet differs from Drive: {target}")
            continue
        temporary = target.with_suffix(".parquet.part")
        try:
            shutil.copyfile(source, temporary)
            if _sha256(temporary) != expected:
                raise OSError(f"Parquet copy checksum mismatch: {name}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def runtime_info():
    """Report dependencies and accelerator without installing anything."""
    result = {"python": sys.version.split()[0]}
    for name in ("torch", "pyarrow", "matplotlib"):
        try:
            module = importlib.import_module(name)
        except ImportError as error:
            raise RuntimeError(f"Missing dependency {name}; install Workshop_4/requirements.txt") from error
        result[name] = module.__version__
        if name == "torch":
            result["cuda_available"] = module.cuda.is_available()
            result["device"] = module.cuda.get_device_name(0) if module.cuda.is_available() else "CPU"
    return result


def build_training_command(workshop_dir, output_dir, processed_dir, *,
                           experiments=None, run_all=False, device="auto",
                           epochs=30, batch_size=512, learning_rate=0.001,
                           seed=42, latent_dim=None, beta=None,
                           resume=False, dry_run=False, python_executable=None,
                           recommended=False, tensor_loader=False, prefetch_batches=0,
                           pin_memory=False, profile_batches=0, gradient_log_every=1):
    """Build an argv list without shell quoting or editing YAML/source files."""
    if sum((bool(run_all), bool(experiments), bool(recommended))) != 1:
        raise ValueError("Choose exactly one of run_all, recommended or named experiments")
    from .runtime import validate_runtime
    runtime = validate_runtime(dict(tensor_loader=tensor_loader, prefetch_batches=prefetch_batches,
                                    pin_memory=pin_memory, profile_batches=profile_batches,
                                    gradient_log_every=gradient_log_every))
    if type(epochs) is not int or not 30 <= epochs <= 50:
        raise ValueError("Full experiments require 30-50 epochs")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    workshop_dir = Path(workshop_dir).expanduser().resolve()
    command = [str(python_executable or sys.executable), "-m", "src.train_ae",
               "--experiments", str(workshop_dir / "config" / "experiments.yaml"),
               "--data-config", str(workshop_dir / "config.json"),
               "--output-dir", str(Path(output_dir).expanduser().resolve()),
               "--processed-dir", str(Path(processed_dir).expanduser().resolve()),
               "--device", device, "--epochs", str(epochs),
               "--batch-size", str(batch_size), "--learning-rate", str(learning_rate),
               "--seed", str(seed)]
    if run_all:
        command.append("--all")
    elif recommended:
        command.append("--recommended")
    else:
        for name in experiments:
            command.extend(["--experiment", str(name)])
    if latent_dim is not None:
        command.extend(["--latent-dim", str(latent_dim)])
    if beta is not None:
        command.extend(["--beta", str(beta)])
    if resume:
        command.extend(["--resume", "auto" if resume is True else str(resume)])
    if dry_run:
        command.append("--dry-run")
    if runtime["tensor_loader"]:
        command.append("--tensor-loader")
    if runtime["pin_memory"]:
        command.append("--pin-memory")
    for key in ("prefetch_batches", "profile_batches", "gradient_log_every"):
        if runtime[key] != {"prefetch_batches": 0, "profile_batches": 0, "gradient_log_every": 1}[key]:
            command.extend(["--" + key.replace("_", "-"), str(runtime[key])])
    return command


def plot_history(output_dir, experiments=None):
    """Plot actual train/validation histories and persist figures beside logs.

    No metric is synthesized when a run or column is absent. Matplotlib is
    imported only here; callers may display the returned figure objects.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir).expanduser().resolve()
    paths = ([output_dir / name / "history.csv" for name in experiments]
             if experiments else sorted(output_dir.glob("*/history.csv")))
    figures = []
    for path in paths:
        if not path.is_file():
            print(f"No history yet: {path}")
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            print(f"Empty history: {path}")
            continue
        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        for axis, metric in zip(axes, ("reconstruction", "kl", "total")):
            for split in ("train", "val"):
                column = f"{split}_{metric}"
                axis.plot([int(row["epoch"]) for row in rows],
                          [float(row[column]) for row in rows], label=split)
            axis.set(xlabel="Epoch", ylabel="Nats per sequence", title=metric)
            axis.legend()
            axis.grid(alpha=0.25)
        figure.suptitle(path.parent.name)
        figure.tight_layout()
        figure.savefig(path.parent / "training_curves.png", dpi=150)
        figures.append(figure)
    return figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-zip", type=Path, required=True,
                        help="Write a Colab code ZIP; excludes Parquet, checkpoints and other workshops")
    parser.add_argument("--workshop-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    print(export_project_zip(args.workshop_dir, args.export_zip))


if __name__ == "__main__":
    main()
