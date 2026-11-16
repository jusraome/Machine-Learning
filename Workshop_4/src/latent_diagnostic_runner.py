"""Phase 3 only: read-only checkpoint evaluation and capped fresh AE probes.

No long trainer, experiment matrix, resume, test split or historical writes.
Run from Workshop_4: python -m src.latent_diagnostic_runner --help.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

from .autoencoder import Autoencoder, reconstruction_loss
from .data import ROOT, DataPipeline, load_config
from .latent_diagnostics import fit_character_baseline, evaluate_latent_dependence
from .stability_experiments import _bounded_inputs, _state_hash, _evaluate
from .train_ae import stable_clip_grad_norm_, NonFiniteGradientError, _invalid_training_tensors


DEFAULTS = dict(seed=42, val_examples=128, batch_size=128, max_updates=24,
                learning_rate=0.001, embedding_dim=32, hidden_size=128,
                latent_dim=32, num_layers=1, dropout=0.0,
                max_seconds=600, max_seconds_per_variant=180)
SNAPSHOTS = (0, 1, 3, 6, 12, 24)


def validate_settings(overrides=None):
    settings = {**DEFAULTS, **(overrides or {})}
    if set(settings) != set(DEFAULTS):
        raise ValueError("Unknown diagnostic setting")
    for key, lo, hi in (("seed", 0, 2**31-1), ("val_examples", 2, 128),
                        ("batch_size", 2, 128), ("max_updates", 1, 24),
                        ("embedding_dim", 1, 32), ("hidden_size", 1, 128),
                        ("latent_dim", 1, 32), ("num_layers", 1, 1),
                        ("max_seconds", 1, 600), ("max_seconds_per_variant", 1, 180)):
        if type(settings[key]) is not int or not lo <= settings[key] <= hi:
            raise ValueError(f"{key} must be an integer in [{lo}, {hi}]")
    lr = settings["learning_rate"]
    if isinstance(lr, bool) or not isinstance(lr, (float, int)) or not math.isfinite(lr) or not 0 < lr <= 0.001:
        raise ValueError("learning_rate must be finite and in (0, .001]")
    if settings["dropout"] != 0.0:
        raise ValueError("The paired diagnostic uses dropout=0")
    return settings


def _hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _save(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Diagnostic wall-time budget exhausted")


def _snapshot(model, val_x, baseline, deadline):
    _deadline(deadline)
    device = next(model.parameters()).device
    before = _state_hash(model)
    report = evaluate_latent_dependence(model, val_x.to(device), baseline)
    # Retain the exact historical metric on its original first 32 val examples.
    report["legacy_first_batch"] = _evaluate(model, val_x, min(32, len(val_x)), deadline)
    if before != _state_hash(model):
        raise RuntimeError("Evaluation changed model parameters")
    _deadline(deadline)
    return report


def run_diagnostics(pipeline, output_dir, *, checkpoints=None, short_probe=False,
                    settings=None, device="cpu", execute=False):
    cfg = validate_settings(settings)
    checkpoints = checkpoints or {}
    if len(checkpoints) > 4 or any(not k.replace("_", "").isalnum() for k in checkpoints):
        raise ValueError("At most four checkpoints with alphanumeric labels are allowed")
    if type(short_probe) is not bool:
        raise ValueError("short_probe must be bool")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Choose a new output directory; historical outputs are immutable")
    if any(destination == Path(p).resolve() or destination in Path(p).resolve().parents for p in checkpoints.values()):
        raise ValueError("Output directory cannot contain a source checkpoint")
    plan = {"phase": 3, "settings": cfg, "checkpoints": {k: str(v) for k,v in checkpoints.items()},
            "short_probe": short_probe, "architectures": ["identity", "tanh"] if short_probe else [],
            "snapshots": sorted({s for s in SNAPSHOTS if s <= cfg["max_updates"]} | {cfg["max_updates"]}) if short_probe else [],
            "max_total_updates": 2*cfg["max_updates"] if short_probe else 0,
            "baseline": "all train x windows, add-alpha=0.5; never val/test fitting",
            "device": str(device), "execute": execute,
            "limitations": ["fixed validation prefix; possibly correlated overlapping windows and speakers",
                            "short probes use a fixed training prefix, not 30-epoch stability evidence",
                            "intervention utility is not an estimate of mutual information",
                            "zero z may be outside the learned latent distribution",
                            "reported T4 pilots cannot be recreated without their saved model weights"]}
    if not execute:
        return plan
    selected_device = torch.device(device)
    if selected_device.type not in ("cpu", "cuda") or (selected_device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Choose an available CPU/CUDA device")
    destination.mkdir(parents=True, exist_ok=False)
    _save(destination / "plan.json", plan)
    deadline = time.monotonic() + cfg["max_seconds"]
    result = {"status": "running", "checkpoints": [], "probes": [], "plan": plan}
    # manual_seed seeds every CUDA generator, so preserve every generator too.
    devices = list(range(torch.cuda.device_count())) if selected_device.type == "cuda" else []
    try:
        # Fitting is streaming over ALL train. The CLI additionally supervises wall time.
        baseline = fit_character_baseline(pipeline, alpha=0.5)
        _save(destination / "baseline.json", baseline)
        _deadline(deadline)
        val_x = _bounded_inputs(pipeline, "val", cfg["val_examples"], True, deadline)
        result["val_tensor_sha256"] = hashlib.sha256(val_x.numpy().tobytes()).hexdigest()
        result["data_contract"] = pipeline.contract()
        _save(destination / "validation_baseline.json", evaluate_latent_dependence_baseline(baseline, val_x))
        with torch.random.fork_rng(devices=devices):
            for label, raw_path in checkpoints.items():
                _deadline(deadline)
                path = Path(raw_path)
                before = _hash_file(path)
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                pipeline.validate_checkpoint(checkpoint)
                if checkpoint.get("model_type") != "ae":
                    raise ValueError("This Phase 3 diagnostic accepts AE checkpoints only")
                model = Autoencoder(**checkpoint["model_hparams"]).to(selected_device)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                if _invalid_training_tensors(model):
                    raise ValueError("Checkpoint contains nonfinite parameters")
                report = _snapshot(model, val_x, baseline, deadline)
                if _hash_file(path) != before:
                    raise RuntimeError("Source checkpoint changed during evaluation")
                entry = {"label": label, "path": str(path.resolve()), "sha256": before,
                         "epoch": checkpoint.get("epoch"), "optimizer_steps": 0,
                         "model_hparams": model.hparams, "diagnostics": report}
                _save(destination / f"checkpoint_{label}.json", entry)
                result["checkpoints"].append(entry)
                del model, checkpoint
            if short_probe:
                count = cfg["batch_size"] * cfg["max_updates"]
                train_x = _bounded_inputs(pipeline, "train", count, True, deadline)
                order = torch.randperm(count, generator=torch.Generator().manual_seed(cfg["seed"]))
                train_x = train_x[order]
                result["train_tensor_sha256"] = hashlib.sha256(train_x.numpy().tobytes()).hexdigest()
                for activation in ("identity", "tanh"):
                    _deadline(deadline)
                    folder = destination / f"short_{activation}"
                    folder.mkdir()
                    torch.manual_seed(cfg["seed"])
                    if selected_device.type == "cuda":
                        torch.cuda.manual_seed_all(cfg["seed"])
                    hparams = {k: cfg[k] for k in ("embedding_dim", "hidden_size", "latent_dim", "num_layers", "dropout")}
                    model = Autoencoder(vocab_size=len(pipeline.vocab["itos"]),
                                        sequence_length=pipeline.config["window_size"],
                                        latent_activation=activation, **hparams).to(selected_device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
                    entry = {"activation": activation, "updates": 0, "examples": 0, "status": "running",
                             "initial_state_sha256": _state_hash(model), "snapshots": [], "training": []}
                    result["probes"].append(entry)
                    variant_deadline = min(deadline, time.monotonic()+cfg["max_seconds_per_variant"])
                    initial_loss = None
                    try:
                        for update in range(cfg["max_updates"]+1):
                            _deadline(variant_deadline)
                            if update in SNAPSHOTS or update == cfg["max_updates"]:
                                report = _snapshot(model, val_x, baseline, variant_deadline)
                                loss = report["legacy_first_batch"]["val_reconstruction"]
                                initial_loss = loss if initial_loss is None else initial_loss
                                if loss > initial_loss*3:
                                    raise RuntimeError("Extreme validation deterioration")
                                if max(report["legacy_first_batch"]["raw_z_max_abs"],
                                       report["legacy_first_batch"]["z_max_abs"]) > 100:
                                    raise RuntimeError("Latent magnitude guard")
                                _save(folder / f"step_{update:03d}.json", report)
                                entry["snapshots"].append({"update": update, "file": f"step_{update:03d}.json"})
                            if update == cfg["max_updates"]:
                                break
                            model.train()
                            x = train_x[update*cfg["batch_size"]:(update+1)*cfg["batch_size"]].to(selected_device)
                            optimizer.zero_grad(set_to_none=True)
                            output = model(x)
                            raw_z = model.encode_raw(x).detach()
                            loss = reconstruction_loss(output.logits, x)
                            if (not torch.isfinite(loss).item() or not torch.isfinite(raw_z).all().item()
                                    or not torch.isfinite(output.z).all().item()
                                    or max(raw_z.abs().max().item(), output.z.abs().max().item()) > 100):
                                raise RuntimeError("Nonfinite loss/latent or latent magnitude guard")
                            if loss.item() > initial_loss*6:
                                raise RuntimeError("Extreme training deterioration")
                            loss.backward()
                            clipping = stable_clip_grad_norm_(model.parameters(), 5.0)
                            if clipping["norm_before"] > 1e8:
                                raise RuntimeError("Raw gradient norm guard")
                            _deadline(variant_deadline)
                            optimizer.step()
                            entry["updates"] += 1
                            entry["examples"] += len(x)
                            if _invalid_training_tensors(model, optimizer):
                                raise RuntimeError("Nonfinite model/Adam after step")
                            entry["training"].append({"update": update+1, "loss_nats_per_sequence": loss.item(), **clipping})
                        entry["status"] = "completed"
                    except (TimeoutError, RuntimeError, FloatingPointError) as error:
                        entry.update(status="stopped", reason=str(error))
                    finally:
                        optimizer.zero_grad(set_to_none=True)
                        entry["final_state_sha256"] = _state_hash(model)
                        _save(folder / "result.json", entry)
                    del model, optimizer
        result["status"] = "completed" if all(p["status"] == "completed" for p in result["probes"]) else "partial"
    except Exception as error:
        # Invalid checkpoint serialization/type must also leave an honest final
        # status. Interrupts and process termination are deliberately not caught.
        result.update(status="failed", reason=str(error))
    finally:
        _save(destination / "summary.json", result)
    return result


def evaluate_latent_dependence_baseline(baseline, x):
    from .latent_diagnostics import evaluate_baseline
    return evaluate_baseline(baseline, x)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[], help="LABEL=PATH, AE evaluation only")
    parser.add_argument("--short-probe", action="store_true", help="Only two fresh models, at most 24 updates each")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    settings = validate_settings(json.loads(args.settings.read_text(encoding="utf-8")) if args.settings else None)
    checkpoints = {}
    for item in args.checkpoint:
        if "=" not in item:
            parser.error("checkpoint requires LABEL=PATH")
        label, path = item.split("=", 1)
        if label in checkpoints:
            parser.error("duplicate checkpoint label")
        checkpoints[label] = path
    if args.execute and not args._worker:
        command = [sys.executable, "-m", "src.latent_diagnostic_runner", *(sys.argv[1:] if argv is None else argv), "--_worker"]
        try:
            return subprocess.run(command, timeout=settings["max_seconds"]+20, check=False).returncode
        except subprocess.TimeoutExpired:
            print("Diagnostic process exceeded wall-time cap; existing files are partial, not completed.", file=sys.stderr)
            return 124
    config = load_config(args.data_config)
    if args.processed_dir:
        config["processed_dir"] = args.processed_dir.resolve()
    result = run_diagnostics(DataPipeline(config), args.output_dir, checkpoints=checkpoints,
                             short_probe=args.short_probe, settings=settings, device=args.device, execute=args.execute)
    print(json.dumps(result if not args.execute else {"status": result["status"], "output_dir": str(args.output_dir)}, indent=2))
    return 0 if result.get("status", "completed") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
