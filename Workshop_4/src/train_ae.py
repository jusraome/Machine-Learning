"""Reproducible streaming AE/VAE training; never reads the test partition.

Run from Workshop_4 with ``python -m src.train_ae --help``. Paths supplied by
CLI are relative to the current directory; defaults are relative to this module.
Epoch-boundary checkpoints are authoritative, including history and RNG state.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
import warnings
from datetime import datetime, timezone

import torch

from .autoencoder import Autoencoder, VariationalAutoencoder, reconstruction_loss, vae_loss
from .data import ROOT, DataPipeline, load_config
from .profiling import StageProfiler
from .runtime import DEFAULT_RUNTIME, iter_prefetched, validate_runtime


DEFAULTS = dict(epochs=30, batch_size=512, learning_rate=1e-3, seed=42,
                beta=1.0, latent_dim=32, embedding_dim=32, hidden_size=128,
                num_layers=1, dropout=0.0, shuffle_buffer_batches=16,
                grad_clip=5.0, normal_class=None)
HISTORY_FIELDS = ["epoch", "train_reconstruction", "train_kl", "train_total",
                  "val_reconstruction", "val_kl", "val_total", "train_examples",
                  "val_examples", "seconds", "learning_rate",
                  "train_attempted_batches", "train_accepted_batches", "train_skipped_batches",
                  "train_attempted_examples", "train_skipped_examples", "train_skipped_fraction",
                  "train_grad_norm_pre_mean", "train_grad_norm_pre_max", "train_grad_norm_post_max",
                  "train_clipped_batches", "train_large_grad_batches"]
STABILITY_DEFAULTS = dict(max_skipped_batches_per_epoch=1, max_skipped_batches_total=2,
                          max_consecutive_skips=1, max_skipped_fraction=0.001)
GRADIENT_CLIPPING = {"algorithm": "global_l2_fp64_v1", "norm_dtype": "float64",
                     "scaling_dtype": "float64", "postcheck_rtol": 1e-6}
GRADIENT_LOG_FIELDS = ["session", "epoch", "batch_index", "norm_before", "norm_after",
                       "scale", "clipped", "fp32_overflow_risk"]


class NonFiniteGradientError(RuntimeError):
    """Only this numerical clipping error is eligible for bounded rejection."""


def _global_grad_norm64(gradients):
    # Conversion happens BEFORE squaring/reduction. Even a norm larger than
    # max(FP32) is representable in FP64 for any realistic FP32 parameter count.
    return torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(gradient.detach().to(torch.float64)) for gradient in gradients
    ])).item()


@torch.no_grad()
def stable_clip_grad_norm_(parameters, max_norm=5.0):
    """Global L2 clipping with strict finiteness and an FP64 postcondition.

    Intended for this trainer's dense FP32/FP64 gradients (no AMP). Every input
    gradient is checked before any mutation. Multiplication ALSO uses FP64:
    casting a tiny scaling coefficient to FP32 first can lose precision or
    underflow even when the correctly scaled gradient is representable.
    Returns the actual global norms before/after casting back to original dtype.
    """
    if isinstance(max_norm, bool) or not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return dict(norm_before=0.0, norm_after=0.0, scale=1.0, clipped=False, fp32_overflow_risk=False)
    if any(gradient.layout != torch.strided or gradient.dtype not in (torch.float32, torch.float64)
           for gradient in gradients):
        raise TypeError("Stable clipping requires dense float32/float64 gradients; AMP is not enabled")
    if len({gradient.device for gradient in gradients}) != 1:
        raise ValueError("All gradients must be on the same device")
    if not torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all().item():
        raise NonFiniteGradientError("Individual gradients contain NaN or Inf; nothing was clipped")
    before = _global_grad_norm64(gradients)
    if not math.isfinite(before):
        raise NonFiniteGradientError("Global gradient norm is non-finite even in float64")
    scale = min(1.0, max_norm / before) if before else 1.0
    if scale < 1.0:
        for gradient in gradients:
            gradient.copy_(gradient.detach().to(torch.float64).mul_(scale))
    # Check the tensors that Adam will actually consume, after the cast back.
    if not torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all().item():
        raise RuntimeError("Stable clipping produced a nonfinite gradient; optimizer step forbidden")
    after = _global_grad_norm64(gradients)
    if not math.isfinite(after) or after > max_norm * (1 + GRADIENT_CLIPPING["postcheck_rtol"]):
        raise RuntimeError("Stable clipping postcondition failed; optimizer step forbidden")
    risk = any(gradient.dtype == torch.float32 for gradient in gradients) and before > math.sqrt(torch.finfo(torch.float32).max)
    return dict(norm_before=before, norm_after=after, scale=scale, clipped=scale < 1.0,
                fp32_overflow_risk=risk)


def _stability_policy(overrides=None):
    policy = {**STABILITY_DEFAULTS, **(overrides or {})}
    if set(policy) != set(STABILITY_DEFAULTS):
        raise ValueError("Unknown stability policy keys")
    for key in STABILITY_DEFAULTS.keys() - {"max_skipped_fraction"}:
        if type(policy[key]) is not int or policy[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    fraction = policy["max_skipped_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (float, int)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("max_skipped_fraction must be finite and in [0,1]")
    return policy


def _tensor_stats(tensor):
    """Failure-only summaries: JSON contains no nonstandard NaN/Infinity values."""
    tensor = tensor.detach()
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite].double()
    norm = torch.linalg.vector_norm(finite_values).item()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "nan": torch.isnan(tensor).sum().item(),
            "inf": torch.isinf(tensor).sum().item(),
            "max_abs_finite": finite_values.abs().max().item() if finite_values.numel() else None,
            "finite_part_norm_float64": norm if math.isfinite(norm) else "overflow",
            "first_nonfinite_indices": (~finite).nonzero()[:8].cpu().tolist()}


def _gradient_report(model):
    parameters = {name: _tensor_stats(param.grad) for name, param in model.named_parameters()
                  if param.grad is not None}
    bad = [name for name, stats in parameters.items() if stats["nan"] or stats["inf"]]
    norms = [stats["finite_part_norm_float64"] for stats in parameters.values()]
    norm64 = math.hypot(*norms) if all(isinstance(norm, (float, int)) for norm in norms) else float("inf")
    return {"bad_gradient_parameters": bad, "gradients": parameters,
            "finite_part_total_norm_float64": norm64 if math.isfinite(norm64) else "overflow"}


def _invalid_training_tensors(model, optimizer=None):
    tensors = [(f"model.{name}", param.detach()) for name, param in model.named_parameters()]
    if optimizer is not None:
        names = {id(param): name for name, param in model.named_parameters()}
        tensors += [(f"adam.{names.get(id(param), '?')}.{key}", value)
                    for param, state in optimizer.state.items() for key, value in state.items()
                    if isinstance(value, torch.Tensor)]
    # One host synchronization on the healthy path; detailed stats only on failure.
    if not tensors or torch.stack([torch.isfinite(value).all().to(next(model.parameters()).device)
                                   for _, value in tensors]).all().item():
        return {}
    return {name: _tensor_stats(value) for name, value in tensors if not torch.isfinite(value).all().item()}


class _StabilityMonitor:
    """Append-only incidents and bounded rejection budget, separate from hyperparameters."""

    def __init__(self, run_dir, policy=None, checkpoint_state=None, detect_anomaly=False):
        self.run_dir = Path(run_dir)
        self.policy = _stability_policy(policy)
        self.detect_anomaly = detect_anomaly
        self.state = {"total_skipped_batches": 0, "consecutive_skips": 0}
        self.state.update(checkpoint_state or {})
        state_path = self.run_dir / "stability_state.json"
        if state_path.exists():
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
            if persisted["policy"] != self.policy:
                raise ValueError("Stability policy changed in an existing run; use a new output directory")
            if persisted["state"]["total_skipped_batches"] >= self.state["total_skipped_batches"]:
                self.state = persisted["state"]
        self.session = f"{datetime.now(timezone.utc).isoformat()}:{os.getpid()}"

    def persist(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.run_dir / "stability_state.json", {"policy": self.policy, "state": self.state})

    def event(self, reason, epoch, batch_index=None, x=None, **details):
        event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "session": self.session,
                 "run": self.run_dir.name, "torch_version": str(torch.__version__),
                 "cuda_version": torch.version.cuda,
                 "reason": reason, "epoch": epoch, "batch_index": batch_index,
                 "policy": self.policy, "state": dict(self.state), **details}
        if x is not None:
            tokens = x.detach().cpu().tolist()
            event["batch"] = {"shape": list(x.shape), "token_min": x.min().item(),
                              "token_max": x.max().item(), "device": str(x.device),
                              "sha256_token_ids": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "nonfinite_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"stability_incident": event}, allow_nan=False), flush=True)

    def reject(self, model, optimizer, epoch, batch_index, x, skipped_in_epoch, error, report):
        optimizer.zero_grad(set_to_none=True)
        self.state["total_skipped_batches"] += 1
        self.state["consecutive_skips"] += 1
        violations = [name for name, count in (
            ("max_skipped_batches_per_epoch", skipped_in_epoch),
            ("max_skipped_batches_total", self.state["total_skipped_batches"]),
            ("max_consecutive_skips", self.state["consecutive_skips"])) if count > self.policy[name]]
        invalid_state = _invalid_training_tensors(model, optimizer)
        if invalid_state:
            violations.append("nonfinite_model_or_adam_state")
        reason = "nonfinite_gradient_elements" if report["bad_gradient_parameters"] else "nonfinite_gradient_norm"
        self.event(reason, epoch, batch_index, x, error=str(error), action="abort" if violations else "skip_batch",
                   split="train", violations=violations, invalid_training_tensors=invalid_state, **report)
        self.persist()  # Rejected/aborted epochs must not reset the lifetime budget on restart.
        if violations:
            raise FloatingPointError(f"Numerical incident budget exceeded: {violations}; see nonfinite_events.jsonl") from error
        warnings.warn(f"Rejected epoch {epoch}, batch {batch_index}: {reason}; metrics exclude these examples",
                      RuntimeWarning, stacklevel=2)


def validate_experiment(config: dict, smoke_test: bool = False) -> dict:
    unknown = set(config) - set(DEFAULTS) - {"name", "model"}
    if unknown:
        raise ValueError(f"Unknown experiment keys: {sorted(unknown)}")
    cfg = {**DEFAULTS, **config}
    if not isinstance(cfg.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cfg["name"]):
        raise ValueError("name must be a simple, nonempty directory name")
    if cfg.get("model") not in ("ae", "vae"):
        raise ValueError("model must be ae or vae")
    for key in ("epochs", "batch_size", "latent_dim", "embedding_dim", "hidden_size",
                "num_layers", "shuffle_buffer_batches"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if smoke_test:
        cfg["epochs"] = min(cfg["epochs"], 2)
    elif not 30 <= cfg["epochs"] <= 50:
        raise ValueError("Full experiments require 30-50 epochs; use --smoke-test for tiny checks")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for key in ("learning_rate", "grad_clip", "beta", "dropout"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite numeric")
    if cfg["learning_rate"] <= 0 or cfg["grad_clip"] <= 0 or cfg["beta"] < 0:
        raise ValueError("learning_rate/grad_clip must be positive and beta nonnegative")
    if not 0 <= cfg["dropout"] < 1:
        raise ValueError("dropout must be in [0,1)")
    if cfg["normal_class"] is not None and (type(cfg["normal_class"]) is not int or cfg["normal_class"] < 0):
        raise ValueError("normal_class must be a nonnegative token ID or null")
    if cfg["model"] == "vae" and not smoke_test and cfg["beta"] not in (0.5, 1.0, 4.0):
        raise ValueError("Workshop VAE experiments require beta 0.5, 1.0 or 4.0")
    return cfg


def _experiment_document(path: str | Path) -> dict:
    """JSON-compatible YAML needs only stdlib; full YAML optionally uses PyYAML."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Use JSON-compatible YAML as supplied, or install PyYAML for full YAML syntax") from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict) or set(payload) - {"defaults", "experiments", "recommended"}:
        raise ValueError("Experiment file requires defaults and experiments")
    return payload


def load_experiments(path: str | Path) -> list[dict]:
    payload = _experiment_document(path)
    defaults = payload.get("defaults", {})
    experiments = payload.get("experiments")
    if not isinstance(defaults, dict) or not isinstance(experiments, list) or not experiments:
        raise ValueError("defaults must be a mapping; experiments must be a nonempty list")
    if any(not isinstance(item, dict) for item in experiments):
        raise ValueError("Each experiment must be a mapping")
    merged = [{**defaults, **item} for item in experiments]
    names = [item.get("name") for item in merged]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError("Experiment names must be unique strings")
    return merged


def recommended_experiments(path: str | Path) -> list[str]:
    document = _experiment_document(path)
    names = document.get("recommended")
    available = {item["name"] for item in load_experiments(path)}
    if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
        raise ValueError("The experiment document must define a recommended list")
    if len(set(names)) != len(names) or set(names) - available:
        raise ValueError("Recommended experiments must be unique and present in experiments")
    return names


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select a GPU runtime or --device cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _rng_state() -> dict:
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact RNG recovery requires the original device count")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_manifest(pipeline: DataPipeline) -> dict:
    """Read only train/val; hash bytes to detect replacement during resume."""
    import pyarrow.parquet as pq

    files = {}
    for split in ("train", "val"):
        path = Path(pipeline.config["processed_dir"]) / f"{split}.parquet"
        with pq.ParquetFile(path) as parquet:
            if not {"character_id", "x", "y"}.issubset(parquet.schema_arrow.names):
                raise ValueError(f"Missing columns in {path}")
            count = parquet.metadata.num_rows
        expected = pipeline.splits.get("counts", {}).get(split, {}).get("n_rows")
        if count <= 0 or expected is not None and count != expected:
            raise ValueError(f"Unexpected {split} row count: {count} != {expected}")
        files[split] = {"sha256": _sha256(path), "rows": count, "size_bytes": path.stat().st_size}
    if pipeline.vocab.get("pad_index") is not None or pipeline.vocab.get("pad_token") is not None:
        raise ValueError("This fixed-length character pipeline expects no PAD token")
    return {"data_contract": pipeline.contract(), "files": files,
            "tokens": {"unk_index": 0, "pad_index": None, "mask": "all_positions"}}


def _batches(pipeline: DataPipeline, split: str, cfg: dict, epoch: int, tensor_loader=False):
    # Bounded-window shuffling: no complete in-memory copy of millions of rows.
    size = cfg["batch_size"]
    chunk_size = size * cfg["shuffle_buffer_batches"] if split == "train" else size
    generator = torch.Generator().manual_seed((cfg["seed"] + epoch) % 2**32)
    reader = (pipeline.iter_tensor_batches if tensor_loader else pipeline.iter_batches)(split, batch_size=chunk_size)
    try:
        for batch in reader:
            x = batch["x"] if isinstance(batch["x"], torch.Tensor) else torch.tensor(batch["x"], dtype=torch.long)
            if cfg["normal_class"] is not None:
                labels = batch["y"] if isinstance(batch["y"], torch.Tensor) else torch.tensor(batch["y"], dtype=torch.long)
                x = x[labels == cfg["normal_class"]]
            if split == "train":
                x = x[torch.randperm(len(x), generator=generator)]
            for start in range(0, len(x), size):
                yield x[start:start + size]
    finally:
        reader.close()


def _profiled_batches(reader, profiler, limit):
    index = 0
    while True:
        profiler.enabled = index < limit
        try:
            with profiler.stage("read_prepare", gpu=False):
                value = next(reader)
        except StopIteration:
            return
        yield value
        index += 1
        if index == limit:
            profiler.summary()  # Only one profiling synchronization after the last sampled batch.
            profiler.enabled = False


def _pinned_batches(reader):
    try:
        for x in reader:
            yield x.pin_memory()
    finally:
        reader.close()


def _epoch(model, pipeline, cfg, epoch, device, optimizer=None, max_batches=None, monitor=None):
    training = optimizer is not None
    if monitor is None:
        raise ValueError("An explicit stability monitor is required for epoch diagnostics")
    model.train(training)
    totals = dict(reconstruction=0.0, kl=0.0, total=0.0)
    examples = attempted_batches = skipped_batches = attempted_examples = skipped_examples = 0
    grad_count = clipped_batches = large_grad_batches = 0
    grad_pre_mean = grad_pre_max = grad_post_max = 0.0
    gradient_log = None
    runtime = validate_runtime(getattr(monitor, "runtime", None))
    profiler = StageProfiler(device, enabled=runtime["profile_batches"] > 0)
    # Keep the legacy call signature when no opt-in is requested (including test fixtures).
    reader = (_batches(pipeline, "train" if training else "val", cfg, epoch, tensor_loader=True)
              if runtime["tensor_loader"] else _batches(pipeline, "train" if training else "val", cfg, epoch))
    if runtime["pin_memory"] and device.type == "cuda":
        reader = _pinned_batches(reader)
    reader = iter_prefetched(reader, runtime["prefetch_batches"])
    timed_reader = _profiled_batches(reader, profiler, runtime["profile_batches"])
    try:
        if training:
            monitor.run_dir.mkdir(parents=True, exist_ok=True)
            log_path = monitor.run_dir / "gradient_norms.csv"
            needs_header = not log_path.exists() or log_path.stat().st_size == 0
            gradient_log = log_path.open("a", newline="", encoding="utf-8")
            norm_writer = csv.DictWriter(gradient_log, fieldnames=GRADIENT_LOG_FIELDS)
            if needs_header:
                norm_writer.writeheader()
        with torch.set_grad_enabled(training):
            for batch_index, x in enumerate(timed_reader):
                with profiler.stage("h2d"):
                    x = x.to(device, non_blocking=runtime["pin_memory"] and device.type == "cuda")
                attempted_batches += 1
                attempted_examples += len(x)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                # Validation estimates E_q[reconstruction] with one fixed-seed
                # MC sample per example, rather than calling the loss at mu.
                anomaly_context = (torch.autograd.detect_anomaly(check_nan=True)
                                   if training and monitor.detect_anomaly else nullcontext())
                with anomaly_context:
                    with profiler.stage("forward_loss"):
                        output = model(x, sample=True) if cfg["model"] == "vae" else model(x)
                        if cfg["model"] == "vae":
                            terms = vae_loss(output.logits, x, output.mu, output.logvar, cfg["beta"])
                            recon, kl, total = terms.reconstruction, terms.kl, terms.total
                        else:
                            recon = reconstruction_loss(output.logits, x)
                            kl = recon.new_zeros(())
                            total = recon
                        if not torch.isfinite(total).item():
                            monitor.event("nonfinite_loss", epoch, batch_index + 1, x,
                                          split="train" if training else "val", action="abort",
                                          logits=_tensor_stats(output.logits))
                            raise FloatingPointError("Nonfinite loss; no batch skipping is permitted for this failure")
                    if training:
                        try:
                            with profiler.stage("backward"):
                                total.backward()
                        except RuntimeError as error:
                            monitor.event("backward_exception", epoch, batch_index + 1, x,
                                          error=str(error), action="abort", **_gradient_report(model))
                            raise  # Do not reinterpret CUDA errors or anomaly traces as recoverable batches.
                if training:
                    try:
                        with profiler.stage("clipping"):
                            clipping = stable_clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    except NonFiniteGradientError as error:
                        report = _gradient_report(model)
                        skipped_batches += 1
                        skipped_examples += len(x)
                        monitor.reject(model, optimizer, epoch, batch_index + 1, x,
                                       skipped_batches, error, report)
                        if max_batches is not None and attempted_batches >= max_batches:
                            break
                        continue  # No optimizer step and no metrics for this rejected batch.
                    except RuntimeError as error:
                        monitor.event("clipping_exception", epoch, batch_index + 1, x,
                                      error=str(error), action="abort", **_gradient_report(model))
                        raise
                    grad_count += 1
                    grad_pre_mean += (clipping["norm_before"] - grad_pre_mean) / grad_count
                    grad_pre_max = max(grad_pre_max, clipping["norm_before"])
                    grad_post_max = max(grad_post_max, clipping["norm_after"])
                    clipped_batches += int(clipping["clipped"])
                    large_grad_batches += int(clipping["fp32_overflow_risk"])
                    with profiler.stage("diagnostics_logging", gpu=False):
                        if clipping["fp32_overflow_risk"] or batch_index % runtime["gradient_log_every"] == 0:
                            norm_writer.writerow({"session": monitor.session, "epoch": epoch,
                                                  "batch_index": batch_index + 1, **clipping})
                        if clipping["fp32_overflow_risk"]:
                            gradient_log.flush()
                            monitor.event("large_finite_gradient_clipped", epoch, batch_index + 1, x,
                                          action="accepted_after_verified_clipping", split="train",
                                          grad_clip=cfg["grad_clip"], gradient_clipping=GRADIENT_CLIPPING, **clipping)
                        elif grad_count % 100 == 0:
                            gradient_log.flush()
                    with profiler.stage("optimizer"):
                        optimizer.step()
                    monitor.state["consecutive_skips"] = 0
                count = len(x)
                with profiler.stage("metrics_logging", gpu=False):
                    # One host transfer rather than three separate .item() synchronizations.
                    values = torch.stack([recon.detach(), kl.detach(), total.detach()]).cpu().tolist()
                    for key, value in zip(("reconstruction", "kl", "total"), values):
                        totals[key] += value * count
                examples += count
                if max_batches is not None and batch_index + 1 >= max_batches:
                    break
    finally:
        timed_reader.close()
        reader.close()
        if training:
            optimizer.zero_grad(set_to_none=True)
            if gradient_log is not None:
                gradient_log.close()
            monitor.persist()
        if runtime["profile_batches"]:
            profile_dir = monitor.run_dir / "profiles"
            profile_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(profile_dir / f"epoch_{epoch}_{'train' if training else 'val'}_{time.time_ns()}.json",
                         {"epoch": epoch, "split": "train" if training else "val",
                          "batch_limit": runtime["profile_batches"], **profiler.summary()})
    if not examples:
        monitor.event("no_accepted_examples", epoch, action="abort", skipped_batches=skipped_batches)
        if skipped_batches:
            raise FloatingPointError("All attempted batches were rejected; epoch cannot be committed")
        raise ValueError("No examples remain; check the normal_class filter and local data")
    skipped_fraction = skipped_examples / attempted_examples if attempted_examples else 0.0
    if training and skipped_fraction > monitor.policy["max_skipped_fraction"]:
        monitor.event("skipped_fraction_exceeded", epoch, action="abort", skipped_fraction=skipped_fraction,
                      skipped_batches=skipped_batches, attempted_examples=attempted_examples)
        raise FloatingPointError("Rejected-example fraction exceeds the stability policy; epoch not committed")
    return {**{key: value / examples for key, value in totals.items()}, "examples": examples,
            "attempted_batches": attempted_batches, "accepted_batches": attempted_batches - skipped_batches,
            "skipped_batches": skipped_batches, "attempted_examples": attempted_examples,
            "skipped_examples": skipped_examples, "skipped_fraction": skipped_fraction,
            "grad_norm_pre_mean": grad_pre_mean, "grad_norm_pre_max": grad_pre_max,
            "grad_norm_post_max": grad_post_max, "clipped_batches": clipped_batches,
            "large_grad_batches": large_grad_batches}


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def _write_history(path: Path, history: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def run_experiment(config: dict, pipeline: DataPipeline, output_dir: str | Path,
                   device: str = "cpu", resume: str | Path | None = None,
                   smoke_test: bool = False, max_train_batches: int | None = None,
                   max_val_batches: int | None = None, stop_after_epoch: int | None = None,
                   stability_policy: dict | None = None, detect_anomaly: bool = False,
                   runtime: dict | None = None) -> dict:
    """Train or resume one run; smoke mode is capped at 2 epochs x 2 batches.

    stop_after_epoch simulates interruption after a committed epoch, without
    changing the target epochs in the saved configuration. Mid-epoch interruption
    restarts that epoch from last.pt; it never claims to recover partial updates.
    """
    cfg = validate_experiment(config, smoke_test)
    policy = _stability_policy(stability_policy)
    runtime = validate_runtime(runtime)
    if not smoke_test and (max_train_batches is not None or max_val_batches is not None):
        raise ValueError("Batch limits are permitted only in --smoke-test mode")
    limits = []
    for limit in (max_train_batches, max_val_batches):
        if smoke_test:
            limit = 2 if limit is None else limit
            if type(limit) is not int or not 1 <= limit <= 2:
                raise ValueError("Smoke-test batch limits must be 1 or 2")
        limits.append(limit)
    max_train_batches, max_val_batches = limits
    if stop_after_epoch is not None and (type(stop_after_epoch) is not int or not 1 <= stop_after_epoch <= cfg["epochs"]):
        raise ValueError("stop_after_epoch must be an epoch within the configured run")
    selected_device = resolve_device(device)
    seed_everything(cfg["seed"])
    manifest = data_manifest(pipeline)
    if cfg["normal_class"] is not None and cfg["normal_class"] >= len(pipeline.vocab["itos"]):
        raise ValueError("normal_class is outside the existing vocabulary")
    training_config = {**cfg, "smoke_test": smoke_test,
                       "max_train_batches": max_train_batches, "max_val_batches": max_val_batches}
    model_hparams = {key: cfg[key] for key in ("embedding_dim", "hidden_size", "latent_dim", "num_layers", "dropout")}
    model_hparams.update(vocab_size=len(pipeline.vocab["itos"]), sequence_length=pipeline.config["window_size"])
    model_type = Autoencoder if cfg["model"] == "ae" else VariationalAutoencoder
    model = model_type(**model_hparams).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    run_dir = Path(output_dir).resolve() / cfg["name"]
    checkpoint_dir = run_dir / "checkpoints"
    last_path, best_path = checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"
    history, start_epoch, best_val, best_epoch, best_state = [], 1, float("inf"), 0, None
    source_checkpoint = None
    continuation_origin = None
    if resume is not None:
        resume_path = last_path if str(resume) == "auto" else Path(resume)
        if str(resume) != "auto" and run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError("An explicit source checkpoint requires a fresh output directory; use --resume auto for a continuation")
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1 or checkpoint.get("checkpoint_role") != "last_resumable":
            raise ValueError("Resume requires a last_resumable training checkpoint (last.pt), not best.pt")
        pipeline.validate_checkpoint(checkpoint)
        if checkpoint["training_config"] != training_config or checkpoint["data_manifest"] != manifest:
            raise ValueError("Resume configuration or data manifest mismatch; use a new run name")
        if checkpoint["model_hparams"] != model_hparams or checkpoint["model_type"] != cfg["model"]:
            raise ValueError("Resume architecture mismatch")
        if checkpoint.get("stability_policy", policy) != policy:
            raise ValueError("Stability policy changed; preserve it when resuming")
        previous_runtime = checkpoint.get("runtime", DEFAULT_RUNTIME)
        transport_keys = ("tensor_loader", "prefetch_batches", "pin_memory")
        if any(previous_runtime[key] != runtime[key] for key in transport_keys):
            if str(resume) == "auto":
                raise ValueError("Transport options changed; compare in a new output folder using an explicit source checkpoint")
            warnings.warn("Transport configuration changed in a separate continuation; bitwise CUDA equivalence is unverified",
                          RuntimeWarning, stacklevel=2)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        best_val, best_epoch = checkpoint["best_val_total"], checkpoint["best_epoch"]
        best_state = checkpoint["best_state_dict"]
        _restore_rng(checkpoint["rng_state"], selected_device)
        source_checkpoint = {"path": str(resume_path.resolve()), "sha256": _sha256(resume_path),
                             "epoch": checkpoint["epoch"], "best_epoch": best_epoch}
        continuation_origin = checkpoint.get("continuation_origin") or source_checkpoint
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory already contains files; use --resume auto or a new name: {run_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monitor = _StabilityMonitor(run_dir, policy, checkpoint.get("stability_state") if resume is not None else None,
                                detect_anomaly=detect_anomaly)
    monitor.runtime = runtime
    invalid_state = _invalid_training_tensors(model, optimizer)
    if resume is not None:
        invalid_state.update({f"best.{name}": _tensor_stats(tensor) for name, tensor in best_state.items()
                              if not torch.isfinite(tensor).all().item()})
    if invalid_state:
        monitor.event("invalid_initial_model_or_adam", start_epoch, action="abort", tensors=invalid_state)
        raise FloatingPointError("Loaded model/best/Adam contains nonfinite tensors; source checkpoint left unchanged")
    monitor.persist()
    metadata = {"training_config": training_config, "model_hparams": model_hparams,
                "runtime": runtime,
                "gradient_clipping": {**GRADIENT_CLIPPING, "max_norm": cfg["grad_clip"]},
                "stability_policy": policy, "detect_anomaly": detect_anomaly,
                "resumed_from": source_checkpoint,
                "continuation_origin": continuation_origin,
                "data_manifest": manifest, "source_paths": {key: str(pipeline.config[key]) for key in
                ("vocab_path", "splits_path", "processed_dir")},
                "environment": {"torch": str(torch.__version__), "device": str(selected_device),
                                "cuda_version": torch.version.cuda,
                                "gpu": torch.cuda.get_device_name() if selected_device.type == "cuda" else None},
                "loss_units": "nats/sequence; train averages accepted examples only, val uses every example",
                "legacy_history": "blank audit columns mean not instrumented, not zero skipped batches",
                "validation_sampling": "fixed-seed single posterior sample per example for VAE",
                "selection": "minimum val_total within this run; no test data",
                "shuffle": "bounded-window shuffle, deterministic seed+epoch",
                "written_utc": datetime.now(timezone.utc).isoformat()}
    _atomic_json(run_dir / "metadata.json", metadata)
    if resume is not None:
        # Restore best and CSV from the committed last checkpoint after a crash
        # between writes, rather than trusting partially updated sidecar files.
        recovered_best = {**checkpoint, "state_dict": best_state, "epoch": best_epoch,
                          "checkpoint_role": "best_inference_only"}
        _atomic_checkpoint(best_path, recovered_best)
        if str(resume) != "auto":
            # Preserve the input artifact and seed the new continuation folder
            # before epoch 12, so a second interruption can use --resume auto.
            _atomic_checkpoint(last_path, {**checkpoint, "stability_policy": policy,
                                          "stability_state": dict(monitor.state),
                                          "runtime": runtime,
                                          "continuation_origin": continuation_origin})
    _write_history(run_dir / "history.csv", history)
    final_epoch = cfg["epochs"] if stop_after_epoch is None else stop_after_epoch
    for epoch in range(start_epoch, final_epoch + 1):
        started = time.monotonic()
        train = _epoch(model, pipeline, cfg, epoch, selected_device, optimizer, max_train_batches, monitor)
        devices = [torch.cuda.current_device()] if selected_device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg["seed"])
            if selected_device.type == "cuda":
                torch.cuda.manual_seed_all(cfg["seed"])
            validation = _epoch(model, pipeline, cfg, epoch, selected_device, max_batches=max_val_batches, monitor=monitor)
        invalid_state = _invalid_training_tensors(model, optimizer)
        if invalid_state:
            monitor.event("nonfinite_model_or_adam_after_epoch", epoch, action="abort", tensors=invalid_state)
            raise FloatingPointError("Invalid model/Adam state; epoch will not be checkpointed")
        row = {"epoch": epoch,
               **{f"train_{key}": train[key] for key in ("reconstruction", "kl", "total")},
               **{f"val_{key}": validation[key] for key in ("reconstruction", "kl", "total")},
               "train_examples": train["examples"], "val_examples": validation["examples"],
               **{f"train_{key}": train[key] for key in ("attempted_batches", "accepted_batches", "skipped_batches",
                                                        "attempted_examples", "skipped_examples", "skipped_fraction",
                                                        "grad_norm_pre_mean", "grad_norm_pre_max", "grad_norm_post_max",
                                                        "clipped_batches", "large_grad_batches")},
               "seconds": time.monotonic() - started, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        improved = validation["total"] < best_val
        if improved:
            best_val, best_epoch = validation["total"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        checkpoint = {"format_version": 1, "checkpoint_role": "last_resumable", "model_type": cfg["model"],
                      "state_dict": model.state_dict(), "model_hparams": model_hparams,
                      "vocab": pipeline.vocab, "data_contract": pipeline.contract(),
                      "data_manifest": manifest, "optimizer_state_dict": optimizer.state_dict(),
                      "training_config": training_config, "epoch": epoch, "best_val_total": best_val,
                      "best_epoch": best_epoch, "best_state_dict": best_state,
                      "history": history, "rng_state": _rng_state(),
                      "runtime": runtime,
                      "gradient_clipping": {**GRADIENT_CLIPPING, "max_norm": cfg["grad_clip"]},
                      "stability_policy": policy, "stability_state": dict(monitor.state),
                      "continuation_origin": continuation_origin}
        # Commit last first. It can reconstruct best and history on recovery.
        _atomic_checkpoint(last_path, checkpoint)
        if improved:
            _atomic_checkpoint(best_path, {**checkpoint, "state_dict": best_state,
                                          "checkpoint_role": "best_inference_only"})
        _write_history(run_dir / "history.csv", history)
        monitor.persist()
        print(json.dumps({"run": cfg["name"], "smoke_test": smoke_test, **row}), flush=True)
    return {"run_dir": str(run_dir), "last_checkpoint": str(last_path), "best_checkpoint": str(best_path),
            "best_epoch": best_epoch, "best_val_total": best_val, "history": history,
            "smoke_test": smoke_test}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=ROOT / "config" / "experiments.yaml")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--experiment", action="append", help="Run name; repeat to select multiple runs")
    selector.add_argument("--all", action="store_true", help="Run every configured experiment sequentially")
    selector.add_argument("--recommended", action="store_true", help="Run only the five recommended representation experiments")
    parser.add_argument("--data-config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--vocab-path", type=Path)
    parser.add_argument("--splits-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "workshop4")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", help="auto for each run's last.pt, or an explicit last.pt path for one run")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print plan without training or writes")
    parser.add_argument("--smoke-test", action="store_true", help="Cap each run at 2 epochs and 2 batches per split")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--stop-after-epoch", type=int, help="Stop cleanly at an epoch boundary; target epochs unchanged")
    parser.add_argument("--detect-anomaly", action="store_true",
                        help="Slow fail-fast autograd trace, including forward context; no silent recovery of backward errors")
    for name in STABILITY_DEFAULTS.keys() - {"max_skipped_fraction"}:
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--max-skipped-fraction", type=float,
                        help="Maximum rejected example fraction per epoch (default 0.001)")
    parser.add_argument("--tensor-loader", action="store_true", help="Opt-in vectorized CPU encoding, same vocabulary/order")
    parser.add_argument("--prefetch-batches", type=int, default=0, help="Opt-in bounded ordered CPU prefetch (0-8)")
    parser.add_argument("--pin-memory", action="store_true", help="Opt-in pinned CPU batches and nonblocking CUDA copy")
    parser.add_argument("--profile-batches", type=int, default=0, help="Profile at most 32 initial batches per split/epoch")
    parser.add_argument("--gradient-log-every", type=int, default=1,
                        help="Record one ordinary norm row every N batches; all large/nonfinite incidents remain logged")
    for name in ("epochs", "batch_size", "latent_dim", "embedding_dim", "hidden_size", "num_layers", "seed", "normal_class"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("learning_rate", "beta", "dropout", "grad_clip"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    configs = load_experiments(args.experiments)
    names = set(recommended_experiments(args.experiments) if args.recommended else args.experiment or [])
    if names - {item["name"] for item in configs}:
        raise ValueError(f"Unknown experiments: {sorted(names - {item['name'] for item in configs})}")
    selected = configs if args.all else [item for item in configs if item["name"] in names]
    if args.resume not in (None, "auto") and len(selected) != 1:
        raise ValueError("An explicit resume path requires exactly one experiment")
    overrides = {key: getattr(args, key) for key in DEFAULTS if hasattr(args, key) and getattr(args, key) is not None}
    selected = [validate_experiment({**item, **overrides}, args.smoke_test) for item in selected]
    policy = _stability_policy({key: getattr(args, key) for key in STABILITY_DEFAULTS if getattr(args, key) is not None})
    runtime = validate_runtime({key: getattr(args, key) for key in DEFAULT_RUNTIME})
    data_config = load_config(args.data_config)
    for key in ("processed_dir", "vocab_path", "splits_path"):
        value = getattr(args, key)
        if value is not None:
            data_config[key] = value.resolve()
    pipeline = DataPipeline(data_config)
    if args.dry_run:
        print(json.dumps({"experiments": selected, "device": str(resolve_device(args.device)),
                          "data_manifest": data_manifest(pipeline), "output_dir": str(args.output_dir.resolve()),
                          "smoke_test": args.smoke_test, "stability_policy": policy,
                          "detect_anomaly": args.detect_anomaly, "runtime": runtime}, indent=2))
        return
    for config in selected:
        effective_resume = args.resume
        # An interrupted sequential grid may include both completed and not-yet
        # started runs. Continue committed runs and start absent directories.
        run_dir = args.output_dir.resolve() / config["name"]
        if args.resume == "auto" and not (run_dir / "checkpoints" / "last.pt").exists():
            if not run_dir.exists() or not any(run_dir.iterdir()):
                effective_resume = None
        run_experiment(config, pipeline, args.output_dir, args.device, effective_resume, args.smoke_test,
                       args.max_train_batches, args.max_val_batches, args.stop_after_epoch,
                       stability_policy=policy, detect_anomaly=args.detect_anomaly, runtime=runtime)


if __name__ == "__main__":
    main()
