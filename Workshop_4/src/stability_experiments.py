"""Explicit, bounded AE stability pilots; never resume a historical optimizer.

All variants start from the same fresh seeded weights. A historical best.pt may
be evaluated separately. There is no test-set access or automatic training.
The CLI supervises executed suites in a child process with a wall-clock timeout.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import torch

from .autoencoder import Autoencoder, reconstruction_loss
from .benchmark_ae import _preserve_rng_and_flags
from .data import ROOT, DataPipeline, load_config
from .train_ae import (NonFiniteGradientError, _invalid_training_tensors, _sha256,
                       data_manifest, resolve_device, seed_everything, stable_clip_grad_norm_)


HARD_CAPS = dict(updates=128, examples=65536, seconds_per_trial=180, seconds_total=900,
                 total_updates=1024, total_examples=262144, validation_examples=512)
FIELDS = ["trial", "phase", "updates", "examples", "train_reconstruction", "val_reconstruction",
          "grad_norm_pre_mean", "grad_norm_pre_max", "grad_norm_post_max", "clipped_fraction",
          "z_max_abs", "raw_z_max_abs", "z_rms", "decoder_saturation_fraction", "h0_saturation_fraction",
          "latent_sensitivity", "sensitivity_meaningful", "examples_per_second", "training_examples_per_second",
          "total_seconds", "peak_gpu_allocated_bytes", "status", "reason"]
CONFIG_KEYS = {"seed", "embedding_dim", "hidden_size", "latent_dim", "num_layers", "dropout",
               "batch_size", "learning_rate", "latent_activation", "tensor_loader", "grad_clip"}
LIMIT_KEYS = {"max_seconds_per_trial", "max_seconds_total", "max_total_updates", "max_total_examples",
              "validation_examples", "validation_batch_size", "evaluate_every_updates", "max_grad_norm",
              "max_latent_abs", "max_reconstruction_ratio", "deterioration_patience"}


def _number(value, name, low, high, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if integer and type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be {'an integer ' if integer else ''}in [{low},{high}]")


def validate_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {"version", "defaults", "variants", "budgets", "limits"} or plan["version"] != 1:
        raise ValueError("Stability plan requires version=1, defaults, variants, budgets and limits")
    plan = copy.deepcopy(plan)
    cfg, limits = plan["defaults"], plan["limits"]
    if not isinstance(cfg, dict) or set(cfg) != CONFIG_KEYS or not isinstance(limits, dict) or set(limits) != LIMIT_KEYS:
        raise ValueError("Unexpected or missing defaults/limits keys")
    for key in ("embedding_dim", "hidden_size", "latent_dim"):
        _number(cfg[key], key, 1, 256, True)
    _number(cfg["num_layers"], "num_layers", 1, 2, True)
    _number(cfg["seed"], "seed", 0, 2**32 - 1, True)
    _number(cfg["batch_size"], "batch_size", 1, 512, True)
    _number(cfg["learning_rate"], "learning_rate", 1e-6, 0.01)
    if cfg["grad_clip"] != 5.0 or cfg["dropout"] != 0.0 or cfg["latent_activation"] != "identity" or type(cfg["tensor_loader"]) is not bool:
        raise ValueError("Reference requires grad_clip=5, dropout=0, identity latent and an explicit loader choice")
    variants = plan["variants"]
    if not isinstance(variants, list) or not 1 <= len(variants) <= 5:
        raise ValueError("Provide 1-5 isolated variants")
    names, references = set(), 0
    for variant in variants:
        if not isinstance(variant, dict) or set(variant) - {"name", "changes", "optional"} or not isinstance(variant.get("changes"), dict):
            raise ValueError("Each variant requires name and changes")
        name, changes = variant.get("name"), variant["changes"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in names:
            raise ValueError("Variant names must be unique simple directory names")
        names.add(name)
        if type(variant.get("optional", False)) is not bool:
            raise ValueError("optional must be bool")
        if not changes:
            references += 1
            if variant.get("optional", False):
                raise ValueError("The reference cannot be optional")
        elif len(changes) != 1 or set(changes) - {"batch_size", "learning_rate", "latent_activation"}:
            raise ValueError("Each variant must change exactly one allowed factor")
        else:
            key, value = next(iter(changes.items()))
            if value == cfg[key]:
                raise ValueError("A variant must actually change its factor")
            if key == "batch_size":
                _number(value, key, 1, 512, True)
            elif key == "learning_rate":
                _number(value, key, 1e-6, 0.01)
            elif value != "tanh" or not variant.get("optional", False):
                raise ValueError("The tanh architecture variant must be explicitly optional")
    if references != 1:
        raise ValueError("Exactly one unchanged reference is required")
    if set(plan["budgets"]) != {"updates", "examples"}:
        raise ValueError("Provide separate updates and examples budgets")
    for budget in plan["budgets"].values():
        if set(budget) != {"max_updates", "max_examples"}:
            raise ValueError("Each budget requires max_updates and max_examples")
        _number(budget["max_updates"], "max_updates", 1, HARD_CAPS["updates"], True)
        _number(budget["max_examples"], "max_examples", 1, HARD_CAPS["examples"], True)
    for key, cap in (("max_seconds_per_trial", 180), ("max_seconds_total", 900),
                     ("max_grad_norm", 1e16), ("max_latent_abs", 1e6), ("max_reconstruction_ratio", 100)):
        _number(limits[key], key, 0.001 if "seconds" in key else 1, cap)
    for key, cap in (("max_total_updates", 1024), ("max_total_examples", 262144),
                     ("validation_examples", 512), ("validation_batch_size", 128),
                     ("evaluate_every_updates", 128), ("deterioration_patience", 5)):
        _number(limits[key], key, 1, cap, True)
    if limits["max_reconstruction_ratio"] <= 1 or limits["max_grad_norm"] < 5:
        raise ValueError("Deterioration ratio must exceed 1 and raw gradient guard must be at least 5")
    return plan


def load_plan(path):
    # JSON is the supplied YAML 1.2 subset, avoiding dependency installation.
    return validate_plan(json.loads(Path(path).read_text(encoding="utf-8")))


def expand_trials(plan, variant_names=None, budget="both", include_architecture=False):
    plan = validate_plan(plan)
    if budget not in ("updates", "examples", "both"):
        raise ValueError("budget must be updates, examples or both")
    available = {v["name"]: v for v in plan["variants"]}
    names = list(variant_names) if variant_names is not None else [v["name"] for v in plan["variants"] if not v.get("optional", False)]
    if not names or len(names) != len(set(names)) or set(names) - available.keys():
        raise ValueError("Select unique known variants")
    trials = []
    for name in names:
        variant = available[name]
        if "latent_activation" in variant["changes"] and not include_architecture:
            raise ValueError("Architecture variant requires --include-architecture and explicit selection")
        cfg = {**plan["defaults"], **variant["changes"]}
        for mode in (("updates", "examples") if budget == "both" else (budget,)):
            bounds = plan["budgets"][mode]
            if mode == "updates":
                updates, examples = bounds["max_updates"], bounds["max_updates"] * cfg["batch_size"]
                feasible = examples <= bounds["max_examples"]
            else:
                examples = bounds["max_examples"]
                updates = math.ceil(examples / cfg["batch_size"])
                feasible = updates <= bounds["max_updates"] and examples % cfg["batch_size"] == 0
            trials.append(dict(name=f"{name}__{mode}", variant=name, budget=mode, config=cfg,
                               budget_limits=bounds, target_updates=updates, target_examples=examples,
                               budget_feasible=feasible))
    return trials


class _StopTrial(RuntimeError):
    pass


def _deadline_check(deadline):
    if time.monotonic() >= deadline:
        raise _StopTrial("time_deadline_reached")


def _state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _bounded_inputs(pipeline, split, count, tensor_loader, deadline):
    chunks, remaining = [], count
    reader = (pipeline.iter_tensor_batches if tensor_loader else pipeline.iter_batches)(split, batch_size=min(count, 512))
    try:
        while remaining:
            _deadline_check(deadline)
            batch = next(reader)
            x = batch["x"] if isinstance(batch["x"], torch.Tensor) else torch.tensor(batch["x"], dtype=torch.long)
            chunks.append(x[:remaining])
            remaining -= len(chunks[-1])
    except StopIteration as error:
        raise ValueError(f"Not enough {split} rows for the common fixed subset ({count})") from error
    finally:
        reader.close()
    return torch.cat(chunks)


@torch.no_grad()
def _evaluate(model, val_x, batch_size, deadline=None):
    was_training, captures = model.training, {}
    hook = model.decoder.register_forward_hook(lambda module, inputs, output: captures.update(hidden=output[0], h0=inputs[1]))
    raw_hook = model.to_latent.register_forward_hook(lambda module, inputs, output: captures.update(raw=output))
    total, count, diagnostics = 0.0, 0, None
    try:
        model.eval()
        device = next(model.parameters()).device
        for start in range(0, len(val_x), batch_size):
            if deadline is not None:
                _deadline_check(deadline)
            x = val_x[start:start + batch_size].to(device)
            output = model(x)
            loss = reconstruction_loss(output.logits, x)
            if not torch.isfinite(loss).item() or not torch.isfinite(output.z).all().item():
                raise _StopTrial("nonfinite_validation")
            total += loss.item() * len(x)
            count += len(x)
            if diagnostics is None:
                hidden, h0, raw = captures["hidden"], captures["h0"], captures["raw"]
                if not torch.isfinite(hidden).all().item() or not torch.isfinite(raw).all().item():
                    raise _StopTrial("nonfinite_decoder_or_raw_latent")
                z = output.z.detach()
                perturbed = model.decode(z.flip(0))
                sensitivity = ((perturbed.double() - output.logits.double()).norm() /
                               output.logits.double().norm().clamp_min(1e-12)).item()
                diagnostics = dict(z_max_abs=z.abs().max().item(), z_rms=z.double().square().mean().sqrt().item(),
                                   raw_z_max_abs=raw.abs().max().item(),
                                   decoder_saturation_fraction=(hidden.abs() >= 0.99).float().mean().item(),
                                   h0_saturation_fraction=(h0.abs() >= 0.99).float().mean().item(),
                                   latent_sensitivity=sensitivity if len(z) > 1 else None,
                                   sensitivity_meaningful=len(z) > 1)
                if not all(value is None or math.isfinite(value) for value in diagnostics.values()):
                    raise _StopTrial("nonfinite_diagnostics")
    finally:
        hook.remove()
        raw_hook.remove()
        model.train(was_training)
    return {"val_reconstruction": total / count, **diagnostics}


def _write_json(path, payload):
    # Exclusive outputs: a pilot never overwrites a historical artifact.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _comparisons(plan, trials):
    """Descriptive comparisons; never select a production model automatically."""
    reference = next(v["name"] for v in plan["variants"] if not v["changes"])
    comparisons = []
    for trial in trials:
        control = next((t for t in trials if t["variant"] == reference and t["budget"] == trial["budget"]), None)
        matched = bool(control and trial["status"] == control["status"] == "completed" and
                       trial["initial_state_sha256"] == control["initial_state_sha256"] and
                       trial["updates" if trial["budget"] == "updates" else "examples"] ==
                       control["updates" if trial["budget"] == "updates" else "examples"])
        control_loss = control.get("val_reconstruction") if control else None
        loss = trial.get("val_reconstruction")
        comparisons.append({"trial": trial["name"], "control": control["name"] if control else None,
                            "matched_budget": matched, "budget": trial["budget"],
                            "updates": trial["updates"], "examples": trial["examples"],
                            "val_reconstruction": loss,
                            "relative_val_loss_change": loss / control_loss - 1 if matched and control_loss and loss is not None else None,
                            "interpretation": "descriptive pilot only; no long-run stability or automatic winner"})
    return comparisons


def _trial(spec, pipeline, train_x, val_x, folder, writer, metrics_stream, limits, device, suite_deadline):
    started = time.monotonic()
    deadline = min(suite_deadline, started + limits["max_seconds_per_trial"])
    cfg = spec["config"]
    seed_everything(cfg["seed"])
    hparams = {key: cfg[key] for key in ("embedding_dim", "hidden_size", "latent_dim", "num_layers", "dropout", "latent_activation")}
    model = Autoencoder(vocab_size=len(pipeline.vocab["itos"]), sequence_length=pipeline.config["window_size"], **hparams).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    result = {**spec, "initial_state_sha256": _state_hash(model), "updates": 0, "examples": 0,
              "status": "running", "reason": "", "training_seconds": 0.0,
              "grad_norm_pre_mean": 0.0, "grad_norm_pre_max": 0.0, "grad_norm_post_max": 0.0,
              "clipped_fraction": 0.0, "train_reconstruction": None}
    total_loss, clipped, deterioration = 0.0, 0, 0
    last_diag, initial_val, trained_at_eval = {}, None, -1
    captures = {}
    hook = model.to_latent.register_forward_hook(lambda module, inputs, output: captures.update(raw=output.detach()))

    def emit(phase):
        elapsed = time.monotonic() - started
        diagnostics = last_diag if phase != "train" else result.get("last_training_latent", {})
        row = {"trial": spec["name"], "phase": phase, **{key: result.get(key, "") for key in FIELDS if key in result}, **diagnostics,
               "examples_per_second": result["examples"] / elapsed if elapsed > 0 else 0.0,
               "training_examples_per_second": result["examples"] / result["training_seconds"] if result["training_seconds"] > 0 else 0.0,
               "total_seconds": elapsed,
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        if not result["updates"]:
            for key in ("grad_norm_pre_mean", "grad_norm_pre_max", "grad_norm_post_max", "clipped_fraction"):
                row[key] = None  # Not measured yet; never present absence as a zero gradient.
        writer.writerow({key: row.get(key, "") for key in FIELDS})
        metrics_stream.flush()
        return row

    def evaluate():
        nonlocal last_diag, initial_val, deterioration, trained_at_eval
        last_diag = _evaluate(model, val_x, limits["validation_batch_size"], deadline)
        trained_at_eval = result["updates"]
        if initial_val is None:
            initial_val = last_diag["val_reconstruction"]
        deterioration = (deterioration + 1 if last_diag["val_reconstruction"] >
                         initial_val * limits["max_reconstruction_ratio"] else 0)
        emit("validation")
        if deterioration >= limits["deterioration_patience"]:
            raise _StopTrial("extreme_validation_deterioration")
        if last_diag["z_max_abs"] > limits["max_latent_abs"] or last_diag["raw_z_max_abs"] > limits["max_latent_abs"]:
            raise _StopTrial("latent_magnitude_guard")

    try:
        _deadline_check(deadline)
        if not spec["budget_feasible"]:
            raise _StopTrial("incomparable_budget_configuration")
        evaluate()
        for start in range(0, spec["target_examples"], cfg["batch_size"]):
            _deadline_check(deadline)
            step_started = time.monotonic()
            x = train_x[start:start + cfg["batch_size"]].to(device)
            if len(x) != cfg["batch_size"]:
                raise _StopTrial("insufficient_common_examples")
            optimizer.zero_grad(set_to_none=True)
            model.train()
            output = model(x)
            loss = reconstruction_loss(output.logits, x)
            if not torch.isfinite(loss).item() or not torch.isfinite(output.z).all().item():
                raise _StopTrial("nonfinite_training_loss_or_latent")
            raw_max, z_max = captures["raw"].abs().max().item(), output.z.detach().abs().max().item()
            if not math.isfinite(raw_max) or max(raw_max, z_max) > limits["max_latent_abs"]:
                raise _StopTrial("latent_magnitude_guard")
            if loss.item() > initial_val * limits["max_reconstruction_ratio"] * 2:
                raise _StopTrial("extreme_training_deterioration")
            loss.backward()
            clipping = stable_clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            if clipping["norm_before"] > limits["max_grad_norm"]:
                result["last_rejected_gradient"] = clipping
                raise _StopTrial("raw_gradient_norm_guard")
            _deadline_check(deadline)
            optimizer.step()
            result["updates"] += 1
            result["examples"] += len(x)
            if _invalid_training_tensors(model, optimizer):
                raise _StopTrial("nonfinite_model_or_adam")
            result["training_seconds"] += time.monotonic() - step_started
            total_loss += loss.detach().item() * len(x)
            clipped += int(clipping["clipped"])
            result["train_reconstruction"] = total_loss / result["examples"]
            result["grad_norm_pre_mean"] += (clipping["norm_before"] - result["grad_norm_pre_mean"]) / result["updates"]
            result["grad_norm_pre_max"] = max(result["grad_norm_pre_max"], clipping["norm_before"])
            result["grad_norm_post_max"] = max(result["grad_norm_post_max"], clipping["norm_after"])
            result["clipped_fraction"] = clipped / result["updates"]
            result["last_training_latent"] = {"z_max_abs": z_max, "raw_z_max_abs": raw_max,
                                              "z_rms": output.z.detach().double().square().mean().sqrt().item()}
            emit("train")
            if result["updates"] % limits["evaluate_every_updates"] == 0:
                evaluate()
        if trained_at_eval != result["updates"]:
            evaluate()
        _deadline_check(deadline)
        result["status"] = "completed"
        result["reason"] = "matched_budget_completed"
    except NonFiniteGradientError as error:
        result["status"], result["reason"] = "stopped", f"nonfinite_gradient: {error}"
    except _StopTrial as error:
        result["status"], result["reason"] = "stopped", str(error)
    except RuntimeError as error:
        result["status"], result["reason"] = "failed", f"runtime_error: {error}"
    finally:
        optimizer.zero_grad(set_to_none=True)
        hook.remove()
        result.update({key: value for key, value in emit("final").items() if key not in ("phase", "trial")})
        result["final_state_sha256"] = _state_hash(model)
        _write_json(folder / "result.json", result)
    return result


def run_suite(plan, pipeline, output_dir, variants=None, budget="both", device="cpu", include_architecture=False,
              reference_checkpoint=None, execute=False):
    plan = validate_plan(plan)
    trials = expand_trials(plan, variants, budget, include_architecture)
    overview = {"pilot_only": True, "execute": execute, "trials": trials, "plan": plan,
                "limitations": ["Short success does not establish 30-epoch stability", "AE pilot, no VAE stability claim",
                                "Fixed prefix of train/val, train shuffled once with common seed; not a representative full split"]}
    if not execute:
        return overview
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    for spec in trials:
        (output_dir / spec["name"]).mkdir()
        _write_json(output_dir / spec["name"] / "configuration.json", spec)
    limits = plan["limits"]
    total_started = time.monotonic()
    deadline = total_started + limits["max_seconds_total"]
    result = {**overview, "trials": [], "status": "running", "reason": "", "reference_evaluation": None,
              "environment": {"torch": str(torch.__version__), "device": str(device), "gpu": None}}
    used_updates = used_examples = validation_work = 0
    manifest = None
    try:
        _deadline_check(deadline)
        chosen = resolve_device(device)
        result["environment"].update(device=str(chosen), gpu=torch.cuda.get_device_name(chosen) if chosen.type == "cuda" else None)
        manifest = data_manifest(pipeline)
        result["data_manifest"] = manifest
        _write_json(output_dir / "plan.json", overview)
        # Identical pool/permutation even when variants or budget modes are run
        # in separate CLI calls with this same plan.
        pool_count = max(bounds["max_examples"] for bounds in plan["budgets"].values())
        with _preserve_rng_and_flags():
            train_x = _bounded_inputs(pipeline, "train", pool_count, plan["defaults"]["tensor_loader"], deadline)
            val_x = _bounded_inputs(pipeline, "val", limits["validation_examples"], plan["defaults"]["tensor_loader"], deadline)
            generator = torch.Generator().manual_seed(plan["defaults"]["seed"])
            train_x = train_x[torch.randperm(len(train_x), generator=generator)]
            result["common_train_ids_sha256"] = hashlib.sha256(train_x.numpy().tobytes()).hexdigest()
            result["common_val_ids_sha256"] = hashlib.sha256(val_x.numpy().tobytes()).hexdigest()
            if reference_checkpoint is not None:
                source = Path(reference_checkpoint)
                before = _sha256(source)
                checkpoint = torch.load(source, map_location="cpu", weights_only=True)
                pipeline.validate_checkpoint(checkpoint)
                if checkpoint.get("model_type") != "ae":
                    raise ValueError("Historical reference must be an AE checkpoint")
                reference = Autoencoder(**checkpoint["model_hparams"]).to(chosen)
                reference.load_state_dict(checkpoint["state_dict"])
                if _invalid_training_tensors(reference):
                    raise _StopTrial("nonfinite_reference_checkpoint")
                result["reference_evaluation"] = {"path": str(source.resolve()), "sha256": before,
                    "epoch": checkpoint.get("epoch"), "evaluation_only": True, "optimizer_restored": False,
                    **_evaluate(reference, val_x, limits["validation_batch_size"], deadline)}
                if _sha256(source) != before:
                    raise RuntimeError("Reference checkpoint changed while being read")
                del reference
            with (output_dir / "metrics.csv").open("x", newline="", encoding="utf-8") as metrics:
                writer = csv.DictWriter(metrics, fieldnames=FIELDS)
                writer.writeheader()
                for spec in trials:
                    _deadline_check(deadline)
                    expected_validation = (2 + math.ceil(spec["target_updates"] / limits["evaluate_every_updates"])) * len(val_x)
                    if (used_updates + spec["target_updates"] > limits["max_total_updates"] or
                        used_examples + spec["target_examples"] > limits["max_total_examples"] or
                        expected_validation > 8192 or validation_work + expected_validation > 65536):
                        result["status"], result["reason"] = "stopped", "suite_work_budget"
                        break
                    trial = _trial(spec, pipeline, train_x, val_x, output_dir / spec["name"], writer, metrics,
                                   limits, chosen, deadline)
                    result["trials"].append(trial)
                    used_updates += trial["updates"]
                    used_examples += trial["examples"]
                    validation_work += expected_validation
                    if trial["status"] == "failed" or "nonfinite" in trial["reason"].lower() or "nan" in trial["reason"].lower():
                        result["status"], result["reason"] = "stopped", "numerical_or_runtime_failure"
                        break
            if result["status"] == "running":
                result["status"] = "completed" if all(t["status"] == "completed" for t in result["trials"]) else "stopped"
    except _StopTrial as error:
        result["status"], result["reason"] = "stopped", str(error)
    except (RuntimeError, ValueError) as error:
        result["status"], result["reason"] = "failed", f"{type(error).__name__}: {error}"
    finally:
        result.update(total_seconds=time.monotonic() - total_started, total_updates=used_updates, total_examples=used_examples)
        result["comparisons"] = _comparisons(plan, result["trials"])
        _write_json(output_dir / "summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/stability_experiments.yaml")
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument("--variant", action="append")
    select.add_argument("--all", action="store_true", help="Select nonoptional controls only")
    parser.add_argument("--budget", choices=("updates", "examples", "both"), default="both")
    parser.add_argument("--include-architecture", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--data-config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--max-seconds-per-trial", type=float)
    parser.add_argument("--max-seconds-total", type=float)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    plan = load_plan(args.config)
    for key in ("max_seconds_per_trial", "max_seconds_total"):
        value = getattr(args, key)
        if value is not None:
            plan["limits"][key] = value
    plan = validate_plan(plan)
    expand_trials(plan, args.variant, args.budget, args.include_architecture)
    if args.execute and not args.worker:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        command = [sys.executable, "-m", "src.stability_experiments", *(sys.argv[1:] if argv is None else argv), "--worker"]
        try:
            subprocess.run(command, check=True, timeout=plan["limits"]["max_seconds_total"] + 20,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(args.output_dir / "supervisor_timeout.json", {"status": "stopped", "reason": "hard_wall_clock_timeout",
                "seconds_with_startup_allowance": plan["limits"]["max_seconds_total"] + 20})
            raise SystemExit("Stability pilot killed at hard wall-clock limit; partial CSV is not a completed trial")
        return
    cfg = load_config(args.data_config)
    if args.processed_dir is not None:
        cfg["processed_dir"] = args.processed_dir.resolve()
    result = run_suite(plan, DataPipeline(cfg), args.output_dir, args.variant, args.budget, args.device,
                       args.include_architecture, args.reference_checkpoint, args.execute)
    print(json.dumps(result if not args.execute else {"status": result["status"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
