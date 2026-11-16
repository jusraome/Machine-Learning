"""Original-AE diagnostic windows with immutable checkpoints and early stopping.

This is deliberately not an epoch trainer. CLI defaults to a plan; --execute
runs at most 256 cumulative updates on an explicitly recorded fixed prefix.
Historical checkpoints are read-only and weights_only requires explicit choice.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

from .autoencoder import Autoencoder, reconstruction_loss
from .benchmark_ae import _preserve_rng_and_flags
from .data import ROOT, DataPipeline, load_config
from .latent_diagnostics import fit_character_baseline, evaluate_latent_dependence
from .stability_experiments import _bounded_inputs, _state_hash
from .train_ae import (stable_clip_grad_norm_, NonFiniteGradientError,
                       _invalid_training_tensors, _rng_state, _restore_rng,
                       seed_everything, GRADIENT_CLIPPING)


HARD_UPDATES = 256
DEFAULTS = dict(seed=42, batch_size=128, max_updates=64, evaluate_every=8,
                max_seconds=600, embedding_dim=32, hidden_size=128, latent_dim=32,
                num_layers=1, dropout=0.0, learning_rate=0.001, grad_clip=5.0,
                val_examples=128, ce_warning_ratio=1.2, ce_severe_ratio=2.0,
                swap_retention=0.5, min_swap_utility=1.0, patience=2,
                max_raw_grad_norm=1e8, max_latent_abs=100.0)


def validate_config(overrides=None):
    cfg = {**DEFAULTS, **(overrides or {})}
    if set(cfg) != set(DEFAULTS):
        raise ValueError("Unknown control setting; epochs, architectures and grids are unsupported")
    for key, low, high in (("seed", 0, 2**31-1), ("batch_size", 2, 128),
            ("max_updates", 1, HARD_UPDATES), ("evaluate_every", 1, 32),
            ("max_seconds", 1, 600), ("embedding_dim", 1, 32),
            ("hidden_size", 1, 128), ("val_examples", 2, 128), ("patience", 1, 5)):
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise ValueError(f"{key} must be an integer in [{low}, {high}]")
    if (type(cfg["latent_dim"]) is not int or cfg["latent_dim"] != 32 or
            type(cfg["num_layers"]) is not int or cfg["num_layers"] != 1 or
            isinstance(cfg["dropout"], bool) or cfg["dropout"] != 0):
        raise ValueError("Only the original deterministic one-layer AE z32 without dropout is supported")
    for key, low, high in (("learning_rate", 0, .001), ("grad_clip", 5, 5),
            ("ce_warning_ratio", 1.05, 1.5), ("ce_severe_ratio", 1.5, 3),
            ("swap_retention", .1, .9), ("min_swap_utility", .1, 100),
            ("max_raw_grad_norm", 1, 1e8), ("max_latent_abs", 1, 100)):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid finite diagnostic threshold: {key}")
    if cfg["learning_rate"] <= 0 or cfg["ce_severe_ratio"] <= cfg["ce_warning_ratio"]:
        raise ValueError("Invalid learning rate or CE stop thresholds")
    return cfg


def control_decision(metrics, state, config):
    """Pure descriptive-prefix stop rules. Clipping alone never triggers a stop.

    CE is compared with the best prefix CE. Swap damage is compared with its
    largest earlier meaningful value. Neither prefix metric selects a full-val
    production best. Utility must exceed a prespecified absolute floor first.
    """
    cfg = validate_config(config)
    ce, swap = float(metrics["reconstruction"]), float(metrics["swap_damage"])
    if not math.isfinite(ce) or not math.isfinite(swap) or ce < 0:
        raise FloatingPointError("Nonfinite/invalid control metric")
    old = copy.deepcopy(state or {})
    if not old:
        old = dict(initial_reconstruction=ce, initial_swap_damage=swap,
                   best_reconstruction=ce, best_swap_damage=max(0.0, swap),
                   ce_bad_count=0, swap_bad_count=0, evaluations=0)
    reference_ce = old["best_reconstruction"]
    reference_swap = old["best_swap_damage"]
    ce_bad = ce >= max(reference_ce, 1e-12) * cfg["ce_warning_ratio"]
    swap_applicable = reference_swap >= cfg["min_swap_utility"]
    swap_bad = swap_applicable and swap <= reference_swap * cfg["swap_retention"]
    old["ce_bad_count"] = old.get("ce_bad_count", 0) + 1 if ce_bad else 0
    old["swap_bad_count"] = old.get("swap_bad_count", 0) + 1 if swap_bad else 0
    old["evaluations"] = old.get("evaluations", 0) + 1
    severe = ce >= max(reference_ce, 1e-12) * cfg["ce_severe_ratio"]
    reasons = []
    if severe:
        reasons.append("severe_prefix_reconstruction_deterioration")
    if old["ce_bad_count"] >= cfg["patience"]:
        reasons.append("sustained_prefix_reconstruction_deterioration")
    if old["swap_bad_count"] >= cfg["patience"]:
        reasons.append("sustained_loss_of_meaningful_swap_utility")
    eligible = not ce_bad and not swap_bad and not reasons
    improved = ce < reference_ce
    # A low-CE snapshot with collapsed swap utility cannot replace protected best.
    if eligible:
        old["best_reconstruction"] = min(reference_ce, ce)
        old["best_swap_damage"] = max(reference_swap, swap)
    return old, dict(stop=bool(reasons), reasons=reasons,
                     warning=bool(ce_bad or swap_bad), eligible_for_best=eligible,
                     improved_reconstruction=improved,
                     swap_guard_applicable=swap_applicable,
                     reference_reconstruction=reference_ce, reference_swap_damage=reference_swap,
                     ce_bad_count=old["ce_bad_count"], swap_bad_count=old["swap_bad_count"])


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Diagnostic window wall-time budget exhausted")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value, replace=False):
    path = Path(path)
    if replace:
        # Only pointers/summary in this invocation's new output directory change.
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temp.replace(path)
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")


def _save_checkpoint(path, payload):
    # Every model artifact is immutable, including the initial and stopped state.
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to replace an immutable diagnostic checkpoint")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
    temporary.rename(path)


def _norm(tensors):
    result = math.hypot(*(torch.linalg.vector_norm(x.detach().double()).item() for x in tensors))
    if not math.isfinite(result):
        raise FloatingPointError("Nonfinite FP64 diagnostic norm")
    return result


def _gradient_blocks(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            groups.setdefault(name.split(".")[0], []).append(parameter.grad)
    return {name: _norm(values) for name, values in groups.items()}


def _adam_summary(optimizer):
    first, second, steps = [], [], []
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if value is not None:
                if not torch.isfinite(value).all().item() or (key == "exp_avg_sq" and (value < 0).any().item()):
                    raise FloatingPointError("Invalid Adam moments")
                (first if key == "exp_avg" else second).append(value)
        if "step" in state:
            step = float(state["step"])
            if not math.isfinite(step) or step < 0:
                raise FloatingPointError("Invalid Adam step")
            steps.append(step)
    return dict(exp_avg_l2=_norm(first), exp_avg_sq_l2=_norm(second),
                step_min=min(steps) if steps else 0, step_max=max(steps) if steps else 0,
                parameter_states=len(optimizer.state))


def checkpoint_optimizer_diagnostics(checkpoint):
    """Read-only moment inventory; never asserts historical weights/Adam align.

    Historical best_inference_only can carry moments from its enclosing last
    checkpoint. Their existence does not authorize an exact optimizer replay.
    """
    role = checkpoint.get("checkpoint_role")
    state = checkpoint.get("optimizer_state_dict", {}).get("state", {})
    moments, steps, bad = {"exp_avg": [], "exp_avg_sq": []}, [], []
    for parameter, values in state.items():
        for key in moments:
            value = values.get(key)
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all().item() or (key == "exp_avg_sq" and (value < 0).any().item()):
                    bad.append(f"{parameter}.{key}")
                else:
                    moments[key].append(value)
        if "step" in values:
            step = float(values["step"])
            if math.isfinite(step) and step >= 0:
                steps.append(step)
            else:
                bad.append(f"{parameter}.step")
    return dict(available=bool(state),checkpoint_role=role,parameter_states=len(state),invalid_fields=bad,
                exp_avg_l2=_norm(moments["exp_avg"]),exp_avg_sq_l2=_norm(moments["exp_avg_sq"]),
                step_min=min(steps) if steps else None,step_max=max(steps) if steps else None,
                alignment=("new_phase4_aligned" if role == "phase4_periodic_resumable" else
                           "unproven_best_inference_only" if role == "best_inference_only" else
                           "historical_alignment_not_verified"),
                limitation="moment magnitudes are observables; do not establish optimizer causality")


def _tensor_fingerprint(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().astype("<i8", copy=False).tobytes()).hexdigest()


def _environment(device):
    return dict(torch=str(torch.__version__),device_type=device.type,cuda_version=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark)


def _validate_best_reference(best, data_state):
    """A resumable periodic file must not advertise a missing protected best.

    References intentionally retain their original absolute location. Preserve
    the prior output directory when continuing; copying a periodic file alone
    does not preserve the run's protected-best artifact.
    """
    if not isinstance(best, dict) or not isinstance(best.get("path"), str):
        raise ValueError("Continuation lacks a protected-best reference")
    path = Path(best["path"])
    if not path.is_file():
        raise ValueError("Protected best checkpoint is missing; keep the original prior output folder, not only a downloaded periodic file")
    digest = _sha256(path)
    if best.get("sha256") is not None and best["sha256"] != digest:
        raise ValueError("Protected best checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_role") != "phase4_periodic_resumable":
        raise ValueError("Protected-best reference is not a Phase4 periodic checkpoint")
    if checkpoint.get("diagnostic_data_state") != data_state:
        raise ValueError("Protected-best reference has different data/order fingerprints")
    metrics = checkpoint.get("diagnostic_metrics", {})
    if (checkpoint.get("cursor_updates") != best.get("cumulative_update") or
            metrics.get("reconstruction") != best.get("reconstruction") or
            metrics.get("swap_damage") != best.get("swap_damage")):
        raise ValueError("Protected-best reference cursor/metrics mismatch")
    return {**best, "sha256": digest}


@torch.no_grad()
def _state_diagnostics(model, x):
    modes = {module: module.training for module in model.modules()}
    captured = {}
    handles = [model.encoder.register_forward_hook(lambda m,i,o: captured.update(encoder=o[0])),
               model.decoder.register_forward_hook(lambda m,i,o: captured.update(decoder=o[0], h0=i[1]))]
    try:
        model.eval()
        output = model(x)
        values = {"z": output.z, "logits": output.logits, **captured}
        if any(not torch.isfinite(v).all().item() for v in values.values()):
            raise FloatingPointError("Nonfinite latent/recurrent state/logits")
        result = {name: dict(rms=v.double().square().mean().sqrt().item(), max_abs=v.abs().max().item())
                  for name,v in values.items()}
        for name in ("encoder", "decoder", "h0"):
            result[name]["near_saturation_fraction"] = (values[name].abs() >= .99).double().mean().item()
        result["saturation_definition"] = "fraction of hidden states abs(h)>=.99; not GRU gate probabilities"
        return result
    finally:
        for handle in handles:
            handle.remove()
        for module, mode in modes.items():
            module.training = mode


@torch.no_grad()
def evaluate_full_validation(model, pipeline, *, batch_size=128, deadline=None):
    """Streaming full-val CE only; swap remains separately scoped to fixed128.

    Exposes a read-only validation primitive, not another training schedule.
    Uses every x position; weights the final partial batch by its actual size.
    """
    modes = {module: module.training for module in model.modules()}
    model.eval()
    reader = pipeline.iter_tensor_batches("val", batch_size=batch_size)
    total, count = 0.0, 0
    device = next(model.parameters()).device
    try:
        for batch in reader:
            if deadline is not None:
                _deadline(deadline)
            x = batch["x"].to(device)
            logits = model(x).logits
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError("Nonfinite full-validation logits")
            loss = reconstruction_loss(logits.double(), x, reduction="sum").item()
            if not math.isfinite(loss):
                raise FloatingPointError("Nonfinite full-validation loss")
            total += loss
            count += len(x)
        if not count:
            raise ValueError("Empty full validation split")
        return dict(scope="full_validation", examples=count, nats_per_sequence=total/count,
                    nats_per_character=total/count/pipeline.config["window_size"],
                    swap_scope="not computed here; fixed diagnostic prefix only")
    finally:
        reader.close()
        for module, mode in modes.items():
            module.training = mode


def _check_original(model_hparams, cfg):
    expected = {key: cfg[key] for key in ("embedding_dim", "hidden_size", "latent_dim", "num_layers", "dropout")}
    if model_hparams.get("latent_activation", "identity") != "identity":
        raise ValueError("Phase 4 retains the original identity AE architecture")
    if any(model_hparams.get(key) != value for key,value in expected.items()):
        raise ValueError("Source architecture differs from the declared original-AE configuration")


def run_controlled(pipeline, output_dir, *, config=None, source_checkpoint=None,
                   initialization="fresh", device="cpu", execute=False, full_validation=False):
    cfg = validate_config(config)
    if initialization not in ("fresh", "weights_only", "continue", "reset_adam"):
        raise ValueError("Choose fresh, weights_only, continue or reset_adam explicitly")
    if (initialization == "fresh") != (source_checkpoint is None):
        raise ValueError("Fresh requires no source; weights_only/continue require a source checkpoint")
    if type(execute) is not bool or type(full_validation) is not bool:
        raise ValueError("Execution/full_validation must be bool")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Output must be a new directory")
    source = Path(source_checkpoint).resolve() if source_checkpoint else None
    if source and (destination == source or destination in source.parents):
        raise ValueError("Output cannot contain the source checkpoint")
    plan = dict(phase=4, runner="original_AE_z32_controlled_window", config=cfg,
                source_checkpoint=str(source) if source else None, initialization=initialization,
                max_cumulative_updates=HARD_UPDATES, fixed_training_prefix_examples=cfg["batch_size"]*HARD_UPDATES,
                diagnostic_validation_examples=cfg["val_examples"], full_validation=full_validation,
                full_validation_schedule="initial and completed-final only, optional; stopped/invalid state is not fully evaluated",
                checkpoint_schedule="initial, each evaluation interval, final; immutable files",
                baseline_scope="all train x only, alpha=.5", device=device,
                control_rules={k:cfg[k] for k in ("ce_warning_ratio","ce_severe_ratio","swap_retention","min_swap_utility","patience")},
                selection="best eligible reconstruction on fixed diagnostic val only; not full-validation production selection",
                execution_enabled=execute,
                limitations=["fixed train prefix and fixed val prefix; no 30-epoch stability demonstration",
                             "weights_only uses fresh Adam, not historical optimizer replay",
                             "continue supports Phase4 checkpoints only; preserves fixed-prefix cursor/order",
                             "reset_adam uses Phase4 cursor/RNG/order with fresh Adam for a matched optimizer contrast",
                             "hypothesis probe, no architecture change or automatic grid"])
    if not execute:
        return plan
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda") or (selected.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Unavailable CPU/CUDA device")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "checkpoints").mkdir()
    (destination / "evaluations").mkdir()
    _write(destination / "plan.json", plan)
    started = time.monotonic()
    deadline = started + cfg["max_seconds"]
    result = dict(status="running", updates=0, cumulative_updates=0, examples=0,
                  initialization=initialization, checkpoints=[], evaluations=[], best_prefix=None,
                  full_validation=[], source_sha256=None, stopped_before_unsafe_step=False)
    optimizer = None
    source_hash = None
    stage = "setup"
    try:
        source_hash = _sha256(source) if source else None
        result["source_sha256"] = source_hash
        with _preserve_rng_and_flags():
            seed_everything(cfg["seed"])
            baseline = fit_character_baseline(pipeline, alpha=.5)
            _write(destination / "baseline.json", baseline)
            _deadline(deadline)
            val_x = _bounded_inputs(pipeline, "val", cfg["val_examples"], True, deadline)
            train_x = _bounded_inputs(pipeline, "train", cfg["batch_size"]*HARD_UPDATES, True, deadline)
            order = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(cfg["seed"]))
            train_x = train_x[order]
            data_state = dict(train_tensor_sha256=_tensor_fingerprint(train_x),
                              val_tensor_sha256=_tensor_fingerprint(val_x),
                              train_baseline_sha256=baseline["train_x_sha256"],
                              contract=pipeline.contract(), order_seed=cfg["seed"],
                              pool_examples=len(train_x), batch_size=cfg["batch_size"])
            _write(destination / "data_state.json", data_state)
            hparams = {k:cfg[k] for k in ("embedding_dim","hidden_size","latent_dim","num_layers","dropout")}
            hparams.update(vocab_size=len(pipeline.vocab["itos"]), sequence_length=pipeline.config["window_size"])
            model = Autoencoder(**hparams).to(selected)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
            control, cursor, inherited_best = {}, 0, None
            checkpoint = None
            if source:
                checkpoint = torch.load(source, map_location="cpu", weights_only=True)
                pipeline.validate_checkpoint(checkpoint)
                if checkpoint.get("model_type") != "ae":
                    raise ValueError("Phase4 accepts original AE checkpoints only")
                _check_original(checkpoint["model_hparams"], cfg)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                result["source_epoch"] = checkpoint.get("epoch")
                result["source_environment"] = checkpoint.get("environment")
                result["source_optimizer_diagnostics"] = checkpoint_optimizer_diagnostics(checkpoint)
                if initialization in ("continue", "reset_adam"):
                    if checkpoint.get("checkpoint_role") != "phase4_periodic_resumable":
                        raise ValueError("Only Phase4 periodic checkpoints have a proven cursor/order for continuation; historical best is inference-only")
                    for key in ("optimizer_state_dict","rng_state","control_state","diagnostic_data_state","control_config","cursor_updates"):
                        if key not in checkpoint:
                            raise ValueError(f"Continuation checkpoint is missing {key}")
                    if checkpoint["diagnostic_data_state"] != data_state:
                        raise ValueError("Continuation data/order fingerprint mismatch")
                    # Only the additional window budget/wall deadline may change.
                    compatible = {k:v for k,v in checkpoint["control_config"].items() if k not in ("max_updates","max_seconds")}
                    current = {k:v for k,v in cfg.items() if k not in ("max_updates","max_seconds")}
                    if compatible != current:
                        raise ValueError("Continuation configuration mismatch")
                    environment = checkpoint.get("environment", {})
                    if environment != _environment(selected):
                        raise ValueError("Continuation backend/environment differs; no exact window replay claimed")
                    if checkpoint.get("control_decision", {}).get("stop"):
                        raise ValueError("Source checkpoint already stopped by deterioration control; automatic continuation forbidden")
                    if initialization == "continue":
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    else:
                        result["optimizer_note"] = "Matched Phase4 branch: cursor/RNG/order/weights retained, Adam moments reset explicitly"
                    cursor = int(checkpoint["cursor_updates"])
                    control = copy.deepcopy(checkpoint["control_state"])
                    inherited_best = _validate_best_reference(checkpoint.get("best_prefix"), data_state)
                    _restore_rng(checkpoint["rng_state"], selected)
                else:
                    result["optimizer_note"] = "Explicit weights-only branch; Adam reset, historical replay not claimed"
            if not 0 <= cursor < HARD_UPDATES:
                raise ValueError("Continuation cursor exhausted or invalid")
            if _invalid_training_tensors(model, optimizer):
                raise FloatingPointError("Nonfinite initial model/Adam")
            _adam_summary(optimizer)
            result["initial_state_sha256"] = _state_hash(model)
            result["best_prefix"] = inherited_best
            result["cumulative_updates"] = cursor
            if full_validation:
                result["full_validation"].append(evaluate_full_validation(model,pipeline,deadline=deadline))

            def snapshot(observe=True):
                nonlocal control, stage
                stage = "evaluation"
                _deadline(deadline)
                model_hash = _state_hash(model)
                diagnostic = evaluate_latent_dependence(model,val_x.to(selected),baseline)
                states = _state_diagnostics(model,val_x[:min(32,len(val_x))].to(selected))
                states["population"] = "first at most 32 examples of fixed diagnostic val prefix"
                if max(states["z"]["max_abs"],diagnostic["latent"]["max_abs"]) > cfg["max_latent_abs"]:
                    raise FloatingPointError("Latent magnitude guard at evaluation")
                metrics = dict(reconstruction=diagnostic["modes"]["original"]["nats_per_sequence"],
                               swap_damage=diagnostic["comparisons"]["swapped_minus_original"]["delta_nats_per_sequence"]["mean"],
                               swap_tv=diagnostic["comparisons"]["swapped_minus_original"]["probability_tv_mean"])
                next_control, decision = control_decision(metrics,control,cfg)
                if observe:
                    control = next_control
                else:
                    # Re-observing the source at the start of a new window is
                    # not another independent warning interval.
                    decision.update(stop=False,reasons=[],
                                    ce_bad_count=control.get("ce_bad_count",0),
                                    swap_bad_count=control.get("swap_bad_count",0))
                if model_hash != _state_hash(model):
                    raise RuntimeError("Evaluation mutated parameters")
                relative = f"checkpoints/update_{cursor:06d}.pt"
                improved = result["best_prefix"] is None or (decision["eligible_for_best"] and metrics["reconstruction"] < result["best_prefix"]["reconstruction"])
                next_best = result["best_prefix"]
                if improved:
                    next_best = dict(path=str(destination/relative),reconstruction=metrics["reconstruction"],
                                               swap_damage=metrics["swap_damage"],cumulative_update=cursor,
                                               scope="fixed_diagnostic_val_prefix")
                report = dict(cumulative_update=cursor, window_updates=result["updates"],metrics=metrics,
                              state_diagnostics=states,decision=decision,diagnostics=diagnostic,
                              adam=_adam_summary(optimizer),scope="fixed_diagnostic_val_prefix")
                _write(destination / "evaluations" / f"update_{cursor:06d}.json",report)
                payload = dict(format_version=1, checkpoint_role="phase4_periodic_resumable",model_type="ae",
                               state_dict=model.state_dict(),model_hparams=model.hparams,vocab=pipeline.vocab,
                               data_contract=pipeline.contract(),optimizer_state_dict=optimizer.state_dict(),
                               rng_state=_rng_state(),control_state=control,control_config=cfg,
                               diagnostic_data_state=data_state,cursor_updates=cursor,best_prefix=next_best,
                               diagnostic_metrics=metrics,gradient_clipping={**GRADIENT_CLIPPING,"max_norm":5.0},
                               control_decision=decision,
                               environment=_environment(selected),
                               origin_source=dict(path=str(source),sha256=source_hash) if source else None,
                               epoch=None,epoch_scope="window updates only; no reconstructed historical epoch")
                _save_checkpoint(destination/relative,payload)
                if improved:
                    # A checkpoint cannot contain its own byte hash; the sidecar
                    # records it after commit, and continuation verifies/rebuilds
                    # that hash when the embedded self-reference omits it.
                    next_best = {**next_best, "sha256": _sha256(destination/relative)}
                result["best_prefix"] = next_best
                result["checkpoints"].append(relative)
                result["evaluations"].append(dict(cumulative_update=cursor,metrics=metrics,decision=decision))
                _write(destination / "best_prefix.json",result["best_prefix"],replace=True)
                _write(destination / "latest_checkpoint.json",dict(path=relative,cumulative_update=cursor),replace=True)
                _deadline(deadline)
                return decision

            decision = snapshot(observe=initialization not in ("continue", "reset_adam"))
            if decision["stop"]:
                result.update(status="stopped",reason=decision["reasons"])
            else:
                target = min(HARD_UPDATES,cursor+cfg["max_updates"])
                with (destination / "updates.jsonl").open("x",encoding="utf-8") as stream:
                    while cursor < target:
                        _deadline(deadline)
                        stage = "pre_update"
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        if any(p.grad is not None for p in model.parameters()):
                            raise RuntimeError("Gradient reset failed; accumulation forbidden")
                        x = train_x[cursor*cfg["batch_size"]:(cursor+1)*cfg["batch_size"]].to(selected)
                        output = model(x)
                        loss = reconstruction_loss(output.logits,x)
                        if not torch.isfinite(loss).item() or not torch.isfinite(output.z).all().item():
                            raise FloatingPointError("Nonfinite training loss/latent; Adam step forbidden")
                        if output.z.abs().max().item() > cfg["max_latent_abs"]:
                            raise FloatingPointError("Training latent magnitude guard")
                        if loss.item() > control["best_reconstruction"]*cfg["ce_severe_ratio"]*2:
                            raise FloatingPointError("Severe train reconstruction deterioration")
                        loss.backward()
                        blocks = _gradient_blocks(model)
                        adam_before = _adam_summary(optimizer)
                        clipping = stable_clip_grad_norm_(model.parameters(),5.0)
                        result["last_clipping"] = clipping
                        if clipping["norm_before"] > cfg["max_raw_grad_norm"]:
                            raise FloatingPointError("Raw gradient magnitude guard; Adam step forbidden")
                        before = {name:p.detach().clone() for name,p in model.named_parameters()}
                        _deadline(deadline)
                        optimizer.step()
                        stage = "after_update"
                        cursor += 1
                        result["updates"] += 1
                        result["cumulative_updates"] = cursor
                        result["examples"] += len(x)
                        if _invalid_training_tensors(model,optimizer):
                            raise FloatingPointError("Nonfinite model/Adam after update; latest earlier checkpoint preserved")
                        adam_after = _adam_summary(optimizer)
                        by_block = {}
                        for name,p in model.named_parameters():
                            block = name.split(".")[0]
                            item = by_block.setdefault(block,{"weights":[],"deltas":[]})
                            item["weights"].append(before[name])
                            item["deltas"].append(p.detach().double()-before[name].double())
                        block_updates = {}
                        for block,item in by_block.items():
                            weight_norm,delta_norm = _norm(item["weights"]),_norm(item["deltas"])
                            block_updates[block] = dict(weight_l2_before=weight_norm,update_l2=delta_norm,
                                                       update_to_weight_ratio=delta_norm/max(weight_norm,1e-12))
                        weight_norm = _norm(before.values())
                        delta_norm = _norm([p.detach().double()-before[n].double() for n,p in model.named_parameters()])
                        row = dict(cumulative_update=cursor,window_update=result["updates"],loss_nats_per_sequence=loss.item(),
                                   gradient_reset_confirmed=True,gradient_blocks_preclip=blocks,clipping=clipping,
                                   adam_before=adam_before,adam_after=adam_after,
                                   parameter_l2_before=weight_norm,parameter_l2_after=_norm(model.parameters()),
                                   actual_update_l2=delta_norm,update_to_weight_ratio=delta_norm/max(weight_norm,1e-12),
                                   block_updates=block_updates,z_max_abs=output.z.detach().abs().max().item())
                        stream.write(json.dumps(row,allow_nan=False)+"\n")
                        stream.flush()
                        if result["updates"] % cfg["evaluate_every"] == 0 or cursor == target:
                            optimizer.zero_grad(set_to_none=True)
                            decision = snapshot()
                            if decision["stop"]:
                                result.update(status="stopped",reason=decision["reasons"])
                                break
                if result["status"] == "running":
                    result.update(status="completed",reason="bounded_window_completed")
            if full_validation and result["status"] == "completed":
                result["full_validation"].append(evaluate_full_validation(model,pipeline,deadline=deadline))
    except Exception as error:
        result.update(status="stopped" if result["checkpoints"] else "failed",
                      reason=f"{type(error).__name__}: {error}",failure_stage=stage,
                      stopped_before_unsafe_step=stage == "pre_update")
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        result["elapsed_seconds"] = time.monotonic()-started
        unchanged = True if source is None else None
        if source_hash is not None:
            try:
                unchanged = _sha256(source) == source_hash
            except OSError:
                unchanged = False
            if not unchanged:
                result.update(status="failed",reason="Source checkpoint changed during diagnostic")
        result["source_unchanged"] = unchanged
        _write(destination / "summary.json",result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/deterioration_control.json")
    parser.add_argument("--data-config",type=Path,default=ROOT/"config.json")
    parser.add_argument("--processed-dir",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--source-checkpoint",type=Path)
    parser.add_argument("--initialization",choices=("fresh","weights_only","continue","reset_adam"),default="fresh")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--full-validation",action="store_true")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--_worker",action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    cfg = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
    if any(cfg[k] != DEFAULTS[k] for k in ("embedding_dim","hidden_size","latent_dim","num_layers","dropout")):
        raise ValueError("The CLI retains all original AE dimensions; reduced dimensions are reserved for synthetic API tests")
    if args.execute and not args._worker:
        command = [sys.executable,"-m","src.deterioration_control",*(sys.argv[1:] if argv is None else argv),"--_worker"]
        try:
            return subprocess.run(command,check=False,timeout=cfg["max_seconds"]+20).returncode
        except subprocess.TimeoutExpired:
            print("Window exceeded hard wall cap; any saved checkpoints remain immutable, result is partial",file=sys.stderr)
            return 124
    data_cfg = load_config(args.data_config)
    if args.processed_dir:
        data_cfg["processed_dir"] = args.processed_dir.resolve()
    report = run_controlled(DataPipeline(data_cfg),args.output_dir,config=cfg,
                            source_checkpoint=args.source_checkpoint,initialization=args.initialization,
                            device=args.device,execute=args.execute,full_validation=args.full_validation)
    print(json.dumps(report if not args.execute else {"status":report["status"],"output_dir":str(args.output_dir)},indent=2))
    return 0 if report.get("status","completed") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
