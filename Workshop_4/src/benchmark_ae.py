"""Bounded, isolated AE/VAE throughput benchmark; never loads training checkpoints.

Use --synthetic to isolate compute, or the existing train Parquet to include
decoding. Each batch size/profile mode starts a fresh model and Adam. This is
a microbenchmark, not an experiment, evaluation or instruction to change an
active run. Run it in a separate idle GPU session, never beside active training.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch

from .autoencoder import Autoencoder, VariationalAutoencoder, reconstruction_loss, vae_loss
from .data import ROOT, DataPipeline, load_config
from .profiling import StageProfiler
from .train_ae import resolve_device, stable_clip_grad_norm_


@contextmanager
def _preserve_rng_and_flags():
    python_state = random.getstate()
    flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = flags[:2]
        torch.use_deterministic_algorithms(flags[2], warn_only=flags[3])


def _source(pipeline, batch_size, hparams, seed, tensor_loader=False):
    if pipeline is None:
        generator = torch.Generator().manual_seed(seed + 1000)
        while True:
            yield torch.randint(hparams["vocab_size"], (batch_size, hparams["sequence_length"]),
                                dtype=torch.long, generator=generator)
    else:
        method = pipeline.iter_tensor_batches if tensor_loader else pipeline.iter_batches
        reader = method("train", batch_size=batch_size)
        try:
            for batch in reader:
                yield batch["x"] if isinstance(batch["x"], torch.Tensor) else torch.tensor(batch["x"], dtype=torch.long)
        finally:
            reader.close()


def _step(model, optimizer, host_x, device, kind, beta, grad_clip, profiler, pin_memory, log):
    with profiler.stage("h2d"):
        if pin_memory and device.type == "cuda":
            host_x = host_x.pin_memory()
        x = host_x.to(device, non_blocking=pin_memory and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    with profiler.stage("forward"):
        output = model(x)
        total = (vae_loss(output.logits, x, output.mu, output.logvar, beta).total
                 if kind == "vae" else reconstruction_loss(output.logits, x))
        if not torch.isfinite(total).item():
            raise FloatingPointError("Benchmark loss became nonfinite; no throughput result is accepted")
    with profiler.stage("backward"):
        total.backward()
    with profiler.stage("clipping"):
        clipping = stable_clip_grad_norm_(model.parameters(), grad_clip)
    with profiler.stage("optimizer"):
        optimizer.step()
    with profiler.stage("logging", gpu=False):
        # Deliberately a memory sink: this does not pretend to measure Drive I/O.
        csv.writer(log).writerow([len(x), total.detach().item(), clipping["norm_before"], clipping["norm_after"]])
    return len(x), clipping


def _trial(kind, hparams, batch_size, device, pipeline, seed, learning_rate, beta,
           grad_clip, warmup_steps, measured_steps, profile, pin_memory, tensor_loader):
    random.seed(seed)
    torch.manual_seed(seed)
    model_type = Autoencoder if kind == "ae" else VariationalAutoencoder
    model = model_type(**hparams).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    warmup = _source(pipeline, batch_size, hparams, seed, tensor_loader)
    log = io.StringIO()
    try:
        for _ in range(warmup_steps):
            _step(model, optimizer, next(warmup), device, kind, beta, grad_clip,
                  StageProfiler(device, enabled=False), pin_memory, log)
    except StopIteration as error:
        raise ValueError("Not enough train examples for benchmark warmup") from error
    finally:
        warmup.close()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    profiler = StageProfiler(device, enabled=profile)
    reader = _source(pipeline, batch_size, hparams, seed, tensor_loader)
    examples = 0
    max_post_norm = 0.0
    started = time.perf_counter()
    try:
        for _ in range(measured_steps):
            with profiler.stage("read", gpu=False):
                host_x = next(reader)
            count, clipping = _step(model, optimizer, host_x, device, kind, beta,
                                    grad_clip, profiler, pin_memory, log)
            examples += count
            max_post_norm = max(max_post_norm, clipping["norm_after"])
        summary = profiler.summary()
        if device.type == "cuda" and not profile:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    except StopIteration as error:
        raise ValueError("Not enough train examples for the requested bounded benchmark") from error
    finally:
        reader.close()
        optimizer.zero_grad(set_to_none=True)
    compute_names = ("forward", "backward", "clipping", "optimizer")
    compute_ms = sum(summary["stages"][name]["device_ms_total" if device.type == "cuda" else "wall_ms_total"]
                     for name in compute_names) if profile else None
    result = {"batch_size": batch_size, "profile_enabled": profile,
              "warmup_steps": warmup_steps, "measured_steps": measured_steps,
              "examples": examples, "elapsed_seconds": elapsed,
              "examples_per_second_end_to_end": examples / elapsed,
              "examples_per_second_compute_intervals": examples / (compute_ms / 1000.0)
              if compute_ms else None,
              "post_clip_norm_max": max_post_norm,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
              "stages": summary["stages"]}
    log.close()
    del optimizer, model
    return result


def run_benchmark(*, model="ae", batch_sizes=(128, 256, 512), device="auto",
                  pipeline=None, model_hparams=None, seed=42, learning_rate=0.001,
                  beta=1.0, grad_clip=5.0, warmup_steps=2, measured_steps=5,
                  profile_modes=(False, True), pin_memory=False, tensor_loader=False):
    """Run independent tiny trials; pipeline=None selects synthetic token data.

    Real-data mode reads train in direct batches, without the training loop's
    bounded-window shuffle. It measures Parquet decoding, not shuffle amortization.
    Results report this distinction and must not be treated as full-epoch timings.
    """
    if model not in ("ae", "vae"):
        raise ValueError("model must be ae or vae")
    if not batch_sizes or any(type(size) is not int or size <= 0 for size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    if type(warmup_steps) is not int or not 0 <= warmup_steps <= 2:
        raise ValueError("warmup_steps must be 0-2")
    if type(measured_steps) is not int or not 1 <= measured_steps <= 10:
        raise ValueError("measured_steps must be 1-10")
    if not profile_modes or any(type(mode) is not bool for mode in profile_modes):
        raise ValueError("profile_modes must contain bool values")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    for name, value in (("learning_rate", learning_rate), ("grad_clip", grad_clip), ("beta", beta)):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0 or name != "beta" and value == 0:
            raise ValueError(f"Invalid {name}")
    if tensor_loader and pipeline is None:
        raise ValueError("tensor_loader requires a real data pipeline")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    selected_device = resolve_device(device)
    hparams = dict(vocab_size=76, sequence_length=80, embedding_dim=32, hidden_size=128,
                   latent_dim=32, num_layers=1, dropout=0.0)
    hparams.update(model_hparams or {})
    if pipeline is not None:
        expected = {"vocab_size": len(pipeline.vocab["itos"]), "sequence_length": pipeline.config["window_size"]}
        for name, value in expected.items():
            if model_hparams and name in model_hparams and model_hparams[name] != value:
                raise ValueError(f"model_hparams.{name} differs from the data contract")
        hparams.update(expected)
    trials = []
    with _preserve_rng_and_flags():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        for size in batch_sizes:
            for profile in profile_modes:
                trials.append(_trial(model, hparams, size, selected_device, pipeline, seed,
                                     learning_rate, beta, grad_clip, warmup_steps, measured_steps,
                                     profile, pin_memory, tensor_loader))
    overhead = {}
    for size in batch_sizes:
        pair = {trial["profile_enabled"]: trial for trial in trials if trial["batch_size"] == size}
        if False in pair and True in pair:
            overhead[str(size)] = pair[True]["elapsed_seconds"] / pair[False]["elapsed_seconds"] - 1.0
    return {"benchmark_only": True, "model": model, "model_hparams": hparams,
            "configuration": {"seed": seed, "learning_rate": learning_rate, "beta": beta,
                              "grad_clip": grad_clip, "pin_memory": pin_memory, "tensor_loader": tensor_loader},
            "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                            "device": str(selected_device), "cuda_version": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None},
            "data_source": "synthetic CPU randint" if pipeline is None else "existing train Parquet; direct batches, no shuffle",
            "data_contract": pipeline.contract() if pipeline is not None else None,
            "logging_target": "StringIO; excludes filesystem and Drive latency",
            "timing_notes": "CUDA events measure stream intervals, not kernel occupancy; compute intervals exclude read/H2D/logging. Short trials include noise and are not full-epoch predictions.",
            "profile_overhead_fraction": overhead, "trials": trials}


def write_result(path, payload):
    """Refuse to replace a prior benchmark, log or other artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("ae", "vae"), default="ae")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--synthetic", action="store_true", help="Use CPU-generated token data instead of train Parquet")
    parser.add_argument("--data-config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--pin-memory", action="store_true", help="Benchmark pin+nonblocking H2D as a separate trial invocation")
    parser.add_argument("--tensor-loader", action="store_true", help="Measure the optional equivalent tensor Parquet decoder")
    parser.add_argument("--no-stage-profile", action="store_true", help="Only time end-to-end, without CUDA-event instrumentation")
    parser.add_argument("--output", type=Path, required=True, help="New JSON output path; existing files are refused")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.synthetic and args.processed_dir is not None:
        parser.error("--synthetic and --processed-dir are mutually exclusive")
    pipeline = None
    if not args.synthetic:
        cfg = load_config(args.data_config)
        if args.processed_dir is not None:
            cfg["processed_dir"] = args.processed_dir.resolve()
        pipeline = DataPipeline(cfg)
    # Must be configured before the first CUDA computation in this process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    result = run_benchmark(model=args.model, batch_sizes=args.batch_sizes, device=args.device,
                           pipeline=pipeline, seed=args.seed, learning_rate=args.learning_rate,
                           beta=args.beta, grad_clip=args.grad_clip, warmup_steps=args.warmup,
                           measured_steps=args.steps, pin_memory=args.pin_memory,
                           tensor_loader=args.tensor_loader,
                           profile_modes=(False,) if args.no_stage_profile else (False, True),
                           model_hparams={"latent_dim": args.latent_dim, "embedding_dim": args.embedding_dim,
                                          "hidden_size": args.hidden_size})
    write_result(args.output, result)
    print(json.dumps({"output": str(args.output.resolve()), "trials": len(result["trials"]),
                      "device": result["environment"]["device"]}))


if __name__ == "__main__":
    main()
