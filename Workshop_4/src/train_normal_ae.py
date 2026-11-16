"""One from-scratch, normal-only AE z32 experiment; never an experiment grid.

Model selection, calibration and development evaluation use distinct validation
speakers. This module reads only train and model-selection validation scores.
The CLI displays a plan unless --execute is explicitly selected.
"""
from __future__ import annotations

import argparse
import copy
import csv
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
from .latent_diagnostics import compare_logits
from .train_ae import (GRADIENT_CLIPPING, _invalid_training_tensors,
                       _rng_state, seed_everything, stable_clip_grad_norm_)


DEFAULTS = dict(epochs=30, batch_size=128, learning_rate=0.0003, seed=42,
                embedding_dim=32, hidden_size=128, latent_dim=32, num_layers=1,
                dropout=0.0, normal_class=None, grad_clip=5.0,
                shuffle_buffer_batches=16, read_batch_size=4096,
                max_updates=150000, max_seconds=14400, evaluate_every_updates=512,
                model_selection_examples=4096, per_speaker_cap=256,
                diagnostic_examples=128, early_stopping_patience=5, min_epochs=3, min_delta=0.1,
                ce_warning_ratio=1.2, ce_severe_ratio=2.0,
                deterioration_patience=2, swap_retention=0.5,
                min_swap_utility=1.0, max_raw_grad_norm=1e8, max_latent_abs=100.0)


def validate_config(overrides=None):
    cfg = {**DEFAULTS, **(overrides or {})}
    if set(cfg) != set(DEFAULTS):
        raise ValueError("Unknown setting: this runner supports one from-scratch AE z32 only")
    for name, low, high in (("epochs", 1, 30), ("batch_size", 2, 512),
            ("seed", 0, 2**31-1), ("embedding_dim", 1, 32), ("hidden_size", 1, 128),
            ("shuffle_buffer_batches", 1, 64), ("read_batch_size", 1, 8192),
            ("max_updates", 1, 150000), ("max_seconds", 1, 21600),
            ("evaluate_every_updates", 1, 4096), ("model_selection_examples", 2, 8192),
            ("per_speaker_cap", 1, 2048), ("diagnostic_examples", 2, 128),
            ("early_stopping_patience", 1, 20), ("min_epochs", 1, 30), ("deterioration_patience", 1, 5)):
        if type(cfg[name]) is not int or not low <= cfg[name] <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if (type(cfg["latent_dim"]) is not int or cfg["latent_dim"] != 32 or
            type(cfg["num_layers"]) is not int or cfg["num_layers"] != 1 or
            isinstance(cfg["dropout"], bool) or cfg["dropout"] != 0):
        raise ValueError("The original deterministic one-layer AE z32 architecture is retained")
    if cfg["normal_class"] is not None and (type(cfg["normal_class"]) is not int or cfg["normal_class"] < 0):
        raise ValueError("normal_class must be a nonnegative vocabulary ID or null")
    for name, low, high in (("learning_rate", 1e-7, .001), ("grad_clip", 5, 5),
            ("min_delta", 0, 10), ("ce_warning_ratio", 1.05, 1.5),
            ("ce_severe_ratio", 1.5, 3), ("swap_retention", .1, .9),
            ("min_swap_utility", .1, 100), ("max_raw_grad_norm", 1, 1e8),
            ("max_latent_abs", 1, 100)):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid finite training setting: {name}")
    if cfg["ce_severe_ratio"] <= cfg["ce_warning_ratio"]:
        raise ValueError("Severe reconstruction guard must exceed its warning ratio")
    if cfg["min_epochs"] > cfg["epochs"]:
        raise ValueError("min_epochs must not exceed requested epochs")
    return cfg


def training_decision(selection_ce, swap_damage, state, config):
    """Pure early-stop/health decisions; low random-code utility is not failure."""
    cfg = validate_config(config)
    ce, swap = float(selection_ce), float(swap_damage)
    if not math.isfinite(ce) or not math.isfinite(swap) or ce < 0:
        raise FloatingPointError("Invalid model-selection reconstruction/latent utility")
    current = copy.deepcopy(state or {})
    initial = not current
    if initial:
        current = dict(best_reconstruction=ce, best_swap_damage=max(0.0, swap),
                       significant_best=ce, stale_evaluations=0, ce_bad_count=0,
                       swap_bad_count=0, evaluations=0)
    ref_ce = current["best_reconstruction"]
    ref_swap = current["best_swap_damage"]
    ce_bad = ce >= max(ref_ce, 1e-12) * cfg["ce_warning_ratio"]
    swap_armed = ref_swap >= cfg["min_swap_utility"]
    swap_bad = swap_armed and swap <= ref_swap * cfg["swap_retention"]
    current["ce_bad_count"] = current["ce_bad_count"] + 1 if ce_bad else 0
    current["swap_bad_count"] = current["swap_bad_count"] + 1 if swap_bad else 0
    reasons = []
    if ce >= max(ref_ce, 1e-12) * cfg["ce_severe_ratio"]:
        reasons.append("severe_normal_selection_reconstruction_deterioration")
    if current["ce_bad_count"] >= cfg["deterioration_patience"]:
        reasons.append("sustained_normal_selection_reconstruction_deterioration")
    if current["swap_bad_count"] >= cfg["deterioration_patience"]:
        reasons.append("sustained_loss_of_learned_latent_utility")
    healthy = not ce_bad and not swap_bad and not reasons
    improved = initial or (healthy and ce < ref_ce)
    significant = initial or (healthy and ce < current["significant_best"] - cfg["min_delta"])
    if healthy:
        current["best_reconstruction"] = min(ref_ce, ce)
        current["best_swap_damage"] = max(ref_swap, swap)
    if significant:
        current["significant_best"] = ce
        current["stale_evaluations"] = 0
    elif not initial:
        current["stale_evaluations"] += 1
    current["evaluations"] += 1
    early = current["stale_evaluations"] >= cfg["early_stopping_patience"]
    return current, dict(healthy=healthy, improved=improved, significant_improvement=significant,
                         stop=bool(reasons), reasons=reasons, early_stopping=early,
                         swap_guard_applicable=swap_armed, warning=ce_bad or swap_bad,
                         ce_bad_count=current["ce_bad_count"], swap_bad_count=current["swap_bad_count"],
                         stale_evaluations=current["stale_evaluations"], reference_ce=ref_ce,
                         reference_swap_damage=ref_swap)


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Normal-only training wall-time budget exhausted")


def _json(path, value, replace=False):
    path = Path(path)
    if replace:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")


def _checkpoint(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to replace an immutable normal-only checkpoint")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
    temporary.rename(path)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


@torch.no_grad()
def evaluate_model_selection(model, x, batch_size=128, diagnostic_examples=128, deadline=None):
    """Only model-selection NORMAL validation enters this function."""
    if x.ndim != 2 or len(x) < 2:
        raise ValueError("At least two model-selection normals are needed")
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        device = next(model.parameters()).device
        total = 0.0
        for start in range(0, len(x), batch_size):
            if deadline is not None:
                _deadline(deadline)
            batch = x[start:start+batch_size].to(device)
            logits = model(batch).logits
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError("Nonfinite normal model-selection logits")
            total += reconstruction_loss(logits.double(), batch, reduction="sum").item()
        probe = x[:min(diagnostic_examples, len(x))].to(device)
        captures = {}
        handle = model.decoder.register_forward_hook(lambda m,i,o: captures.update(hidden=o[0].detach(),h0=i[1].detach()))
        try:
            z = model.encode(probe)
            original = model.decode(z)
        finally:
            handle.remove()
        if not torch.isfinite(z).all().item():
            raise FloatingPointError("Nonfinite model-selection latent")
        permutation = torch.arange(len(probe), device=device).roll(len(probe)//2)
        swapped = model.decode(z[permutation])
        zero = model.decode(torch.zeros_like(z))
        comparison = compare_logits(original, swapped, probe)
        zero_comparison = compare_logits(original, zero, probe)
        return dict(scope="normal_model_selection_validation_only", examples=len(x),
                    reconstruction_nats_per_sequence=total/len(x),
                    reconstruction_nats_per_character=total/len(x)/x.shape[1],
                    diagnostic_examples=len(probe),
                    swap_damage_nats_per_sequence=comparison["delta_nats_per_sequence"]["mean"],
                    swap_probability_tv=comparison["probability_tv_mean"],
                    swap_probability_js_nats=comparison["probability_js_nats_mean"],
                    swap_top1_change_fraction=comparison["top1_change_fraction"],
                    zero_damage_nats_per_sequence=zero_comparison["delta_nats_per_sequence"]["mean"],
                    z_max_abs=z.abs().max().item(),z_rms=z.double().square().mean().sqrt().item(),
                    z_between_example_std_rms=z.double().var(0,unbiased=False).mean().sqrt().item(),
                    decoder_saturation_fraction=(captures["hidden"].abs()>=.99).double().mean().item(),
                    initial_hidden_saturation_fraction=(captures["h0"].abs()>=.99).double().mean().item(),
                    saturation_definition="fraction abs(hidden)>=.99; not gate probabilities",
                    swap_source_target_token_agreement=(probe==probe[permutation]).double().mean().item(),
                    swap_per_position=comparison["per_position"],
                    utility_limit="descriptive fixed normal subset; small random response is not learned information loss")
    finally:
        for module, mode in modes.items():
            module.training=mode


def run_normal_only(pipeline, output_dir, *, config=None, device="cpu", execute=False, smoke_test=False):
    cfg=validate_config(config)
    if type(execute) is not bool or type(smoke_test) is not bool:
        raise ValueError("execute/smoke_test must be bool")
    from .normal_only_protocol import (role_partition, fit_normal_class, prepare_role_samples,
                                       role_tensor, iter_normal_train_batches, build_train_input_guard)
    partition=role_partition(pipeline,cfg["seed"])
    destination=Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Normal-only training requires a new output directory")
    plan=dict(operation="single_from_scratch_normal_only_AE_z32",execute=execute,
              smoke_test=smoke_test,config=cfg,validation_partition=partition,
              initialization="new original AE weights and new Adam; no historical checkpoint",
              model_selection="normal-only selection speakers; calibration/evaluation scores unavailable to training",
              test_access="none",effective_max_updates=min(2,cfg["max_updates"]) if smoke_test else cfg["max_updates"],
              early_stopping_unit="model-selection evaluations, including epoch-end evaluations",
              grid="none; AE z64 and VAE remain pending")
    if not execute:
        return plan
    selected=torch.device(device)
    if selected.type not in ("cpu","cuda") or (selected.type=="cuda" and not torch.cuda.is_available()):
        raise ValueError("Choose an available CPU/CUDA device")
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"checkpoints").mkdir()
    (destination/"evaluations").mkdir()
    _json(destination/"plan.json",plan)
    started=time.monotonic()
    deadline=started+cfg["max_seconds"]
    result=dict(status="running",reason=None,global_updates=0,training_examples_seen=0,
                epochs_completed=0,best_checkpoint=None,best_inference=None,evaluations=[],
                initialization="from_scratch",smoke_test=smoke_test,test_accessed=False)
    optimizer=None
    stage="preparation"
    try:
        with _preserve_rng_and_flags():
            seed_everything(cfg["seed"])
            normal=fit_normal_class(pipeline,cfg["normal_class"],batch_size=cfg["read_batch_size"],deadline=deadline)
            _json(destination/"normal_definition.json",normal)
            _json(destination/"validation_partition.json",partition)
            guard=build_train_input_guard(pipeline,normal["normal_class"],
                read_batch_size=cfg["read_batch_size"],deadline=deadline)
            _json(destination/"train_input_guard.json",guard["metadata"])
            sampled=prepare_role_samples(pipeline,partition,normal["normal_class"],roles=("model_selection",),
                max_per_role=cfg["model_selection_examples"],per_speaker_cap=cfg["per_speaker_cap"],
                read_batch_size=cfg["read_batch_size"],deadline=deadline,
                exclude_input_hashes=guard["input_hashes"])
            _json(destination/"model_selection_sampling.json",sampled["sampling"])
            selection=role_tensor(pipeline,sampled["samples"]["model_selection"],split="val")
            selection_x,selection_y=selection["x"],selection["y"]
            if len(selection_x)<2 or not (selection_y==normal["normal_class"]).all().item():
                raise ValueError("Model selection requires at least two exclusively normal validation examples")
            hparams={k:cfg[k] for k in ("embedding_dim","hidden_size","latent_dim","num_layers","dropout")}
            hparams.update(vocab_size=len(pipeline.vocab["itos"]),sequence_length=pipeline.config["window_size"])
            model=Autoencoder(**hparams).to(selected)
            optimizer=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
            training_config={**cfg,"model":"ae","normal_class":normal["normal_class"],
                             "from_scratch":True,"smoke_test":smoke_test}
            protocol=dict(from_scratch=True,train_filter="y_id == normal_class",normal_class=normal["normal_class"],
                          role_digest=partition["role_digest"],
                          experiment_scope="smoke_integration" if smoke_test else "normal_only_training")
            state={}
            last_evaluated=-1
            expected_normals=normal["train_label_counts"][normal["normal_class"]]
            result.update(normal_definition=normal,validation_partition=partition,
                          expected_normal_examples_per_epoch=expected_normals,training_config=training_config)
            history_stream=(destination/"history.csv").open("x",newline="",encoding="utf-8")
            fields=["epoch","global_updates","training_examples_seen","train_reconstruction_nats_per_sequence",
                    "train_cumulative_reconstruction_nats_per_sequence",
                    "selection_reconstruction_nats_per_sequence","selection_reconstruction_nats_per_character",
                    "swap_damage_nats_per_sequence","swap_probability_tv","z_max_abs",
                    "clipped_fraction","grad_norm_pre_max","grad_norm_post_max",
                    "stale_evaluations","warning","stop","elapsed_seconds"]
            writer=csv.DictWriter(history_stream,fieldnames=fields)
            writer.writeheader()
            total_loss=0.0
            epoch_loss=0.0
            epoch_examples=0
            clipped=0
            grad_pre_max=grad_post_max=0.0

            def snapshot(epoch,batch_index):
                nonlocal state,last_evaluated,stage
                stage="model_selection"
                _deadline(deadline)
                metrics=evaluate_model_selection(model,selection_x,cfg["batch_size"],cfg["diagnostic_examples"],deadline)
                if metrics["z_max_abs"]>cfg["max_latent_abs"]:
                    raise FloatingPointError("Latent magnitude guard during model selection")
                state,decision=training_decision(metrics["reconstruction_nats_per_sequence"],
                                                metrics["swap_damage_nats_per_sequence"],state,cfg)
                # Warmup only delays plateau-based stopping. Numerical and
                # actual deterioration guards remain active from the outset.
                if result["epochs_completed"] < cfg["min_epochs"]:
                    state["stale_evaluations"]=0
                    decision.update(early_stopping=False,stale_evaluations=0)
                decision["early_stopping_minimum_epochs_satisfied"]=result["epochs_completed"]>=cfg["min_epochs"]
                update=result["global_updates"]
                relative=f"checkpoints/update_{update:08d}.pt"
                payload=dict(format_version=1,checkpoint_role="normal_only_ae_periodic_state",model_type="ae",
                    state_dict=model.state_dict(),model_hparams=model.hparams,vocab=pipeline.vocab,
                    data_contract=pipeline.contract(),optimizer_state_dict=optimizer.state_dict(),rng_state=_rng_state(),
                    training_config=training_config,normal_definition=normal,validation_partition=partition,
                    train_input_guard=guard["metadata"],
                    normal_only_protocol=protocol,selection_metrics=metrics,selection_decision=decision,
                    early_stopping_state=state,epoch=epoch,global_updates=update,
                    training_examples_seen=result["training_examples_seen"],
                    train_cursor=dict(epoch=epoch,completed_batches_in_epoch=batch_index),
                    gradient_clipping={**GRADIENT_CLIPPING,"max_norm":5.0},
                    source_checkpoint=None,continuation_support="not implemented; first run starts from scratch only",
                    environment=dict(torch=str(torch.__version__),device=str(selected),cuda_version=torch.version.cuda,
                        cudnn_version=torch.backends.cudnn.version(),
                        gpu=torch.cuda.get_device_name(selected) if selected.type=="cuda" else None,
                        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                        cudnn_deterministic=torch.backends.cudnn.deterministic,
                        cudnn_benchmark=torch.backends.cudnn.benchmark))
                _checkpoint(destination/relative,payload)
                sha=_sha(destination/relative)
                if result["best_checkpoint"] is None or (decision["healthy"] and decision["improved"]):
                    pointer=dict(path=str(destination/relative),sha256=sha,epoch=epoch,global_updates=update,
                                 reconstruction_nats_per_sequence=metrics["reconstruction_nats_per_sequence"],
                                 swap_damage_nats_per_sequence=metrics["swap_damage_nats_per_sequence"],
                                 selection_scope="normal model-selection validation speakers only")
                    _json(destination/"best_checkpoint.json",pointer,replace=True)
                    result["best_checkpoint"]=pointer
                _json(destination/"latest_checkpoint.json",dict(path=str(destination/relative),sha256=sha,
                        global_updates=update),replace=True)
                report=dict(epoch=epoch,global_updates=update,selection_metrics=metrics,decision=decision,
                            checkpoint=relative)
                _json(destination/"evaluations"/f"update_{update:08d}.json",report)
                result["evaluations"].append(report)
                row=dict(epoch=epoch,global_updates=update,training_examples_seen=result["training_examples_seen"],
                    train_reconstruction_nats_per_sequence=epoch_loss/epoch_examples if epoch_examples else None,
                    train_cumulative_reconstruction_nats_per_sequence=total_loss/result["training_examples_seen"] if result["training_examples_seen"] else None,
                    selection_reconstruction_nats_per_sequence=metrics["reconstruction_nats_per_sequence"],
                    selection_reconstruction_nats_per_character=metrics["reconstruction_nats_per_character"],
                    swap_damage_nats_per_sequence=metrics["swap_damage_nats_per_sequence"],
                    swap_probability_tv=metrics["swap_probability_tv"],z_max_abs=metrics["z_max_abs"],
                    clipped_fraction=clipped/update if update else None,grad_norm_pre_max=grad_pre_max if update else None,
                    grad_norm_post_max=grad_post_max if update else None,stale_evaluations=state["stale_evaluations"],
                    warning=decision["warning"],stop=decision["stop"],elapsed_seconds=time.monotonic()-started)
                writer.writerow(row)
                history_stream.flush()
                last_evaluated=update
                return decision

            try:
                decision=snapshot(0,0)
                maximum_updates=min(2,cfg["max_updates"]) if smoke_test else cfg["max_updates"]
                with (destination/"updates.jsonl").open("x",encoding="utf-8") as updates_stream:
                    for epoch in range(1,cfg["epochs"]+1):
                        epoch_examples=0
                        epoch_loss=0.0
                        batch_index=0
                        reader=iter_normal_train_batches(pipeline,normal["normal_class"],batch_size=cfg["batch_size"],
                            epoch=epoch,seed=cfg["seed"],shuffle_buffer_batches=cfg["shuffle_buffer_batches"],
                            read_batch_size=cfg["read_batch_size"],deadline=deadline)
                        completed_epoch=True
                        try:
                            for batch_index,batch in enumerate(reader,1):
                                stage="pre_update"
                                _deadline(deadline)
                                labels=torch.as_tensor(batch["y"],dtype=torch.long)
                                if not len(labels) or not (labels==normal["normal_class"]).all().item():
                                    raise RuntimeError("Non-normal/empty example attempted to enter normal-only training")
                                x=batch["x"].to(selected)
                                model.train()
                                optimizer.zero_grad(set_to_none=True)
                                if any(p.grad is not None for p in model.parameters()):
                                    raise RuntimeError("Gradient reset failed; accumulation forbidden")
                                output=model(x)
                                loss=reconstruction_loss(output.logits,x)
                                if not torch.isfinite(loss).item() or not torch.isfinite(output.z).all().item():
                                    raise FloatingPointError("Nonfinite normal training loss/latent before Adam")
                                z_max=output.z.detach().abs().max().item()
                                if z_max>cfg["max_latent_abs"]:
                                    raise FloatingPointError("Normal training latent magnitude guard")
                                if loss.item()>max(state["best_reconstruction"],1e-12)*cfg["ce_severe_ratio"]*2:
                                    raise FloatingPointError("Extreme normal train reconstruction deterioration")
                                loss.backward()
                                clipping=stable_clip_grad_norm_(model.parameters(),5.0)
                                if clipping["norm_before"]>cfg["max_raw_grad_norm"]:
                                    result["last_rejected_clipping"]=clipping
                                    raise FloatingPointError("Raw gradient magnitude guard; Adam step forbidden")
                                _deadline(deadline)
                                optimizer.step()
                                stage="after_update"
                                result["global_updates"]+=1
                                result["training_examples_seen"]+=len(x)
                                epoch_examples+=len(x)
                                if _invalid_training_tensors(model,optimizer):
                                    raise FloatingPointError("Nonfinite normal-only model/Adam after update; prior checkpoint preserved")
                                total_loss+=loss.detach().item()*len(x)
                                epoch_loss+=loss.detach().item()*len(x)
                                clipped+=int(clipping["clipped"])
                                grad_pre_max=max(grad_pre_max,clipping["norm_before"])
                                grad_post_max=max(grad_post_max,clipping["norm_after"])
                                record=dict(epoch=epoch,batch_index=batch_index,global_updates=result["global_updates"],
                                    examples=len(x),normal_class=normal["normal_class"],all_labels_normal=True,
                                    reconstruction_nats_per_sequence=loss.item(),z_max_abs=z_max,clipping=clipping)
                                if "row_indices" in batch:
                                    indices=torch.as_tensor(batch["row_indices"],dtype=torch.int64)
                                    record["training_rows_sha256"]=hashlib.sha256(indices.numpy().tobytes()).hexdigest()
                                updates_stream.write(json.dumps(record,allow_nan=False)+"\n")
                                updates_stream.flush()
                                limit=result["global_updates"]>=maximum_updates
                                if result["global_updates"]%cfg["evaluate_every_updates"]==0 or limit:
                                    optimizer.zero_grad(set_to_none=True)
                                    decision=snapshot(epoch,batch_index)
                                    if decision["stop"]:
                                        result.update(status="stopped",reason=decision["reasons"])
                                    elif decision["early_stopping"]:
                                        result.update(status="completed",reason="early_stopping_normal_selection")
                                if result["status"]!="running" or limit:
                                    completed_epoch=epoch_examples==expected_normals
                                    if result["status"]=="running":
                                        finished_requested_epochs=completed_epoch and epoch==cfg["epochs"]
                                        result.update(status="completed" if smoke_test or finished_requested_epochs else "budget_exhausted",
                                            reason=("smoke_integration_only" if smoke_test else
                                                "requested_normal_only_epochs_completed" if finished_requested_epochs else
                                                "max_updates_reached_before_requested_epochs"))
                                    break
                        finally:
                            if hasattr(reader,"close"):
                                reader.close()
                        if not batch_index:
                            raise ValueError("No normal training examples remained")
                        if completed_epoch:
                            if epoch_examples!=expected_normals:
                                raise ValueError("Filtered epoch count differs from train-only normal definition")
                            result["epochs_completed"]=epoch
                            if last_evaluated!=result["global_updates"]:
                                optimizer.zero_grad(set_to_none=True)
                                decision=snapshot(epoch,batch_index)
                                if decision["stop"]:
                                    result.update(status="stopped",reason=decision["reasons"])
                                elif decision["early_stopping"]:
                                    result.update(status="completed",reason="early_stopping_normal_selection")
                        if result["status"]!="running":
                            break
                    if result["status"]=="running":
                        result.update(status="completed",reason="requested_normal_only_epochs_completed")
            finally:
                history_stream.close()
    except Exception as error:
        result.update(status="stopped" if result["best_checkpoint"] else "failed",
                      reason=f"{type(error).__name__}: {error}",failure_stage=stage)
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        # Only committed, finite, healthy best state is published for evaluation.
        pointer=result.get("best_checkpoint")
        if pointer is not None and pointer["global_updates"] < 1:
            result["best_inference_status"]="no_trained_best: the protected best is still the initial random model"
        if pointer is not None and pointer["global_updates"] >= 1:
            try:
                if _sha(pointer["path"])!=pointer["sha256"]:
                    raise RuntimeError("Protected best checkpoint changed before inference export")
                best=torch.load(pointer["path"],map_location="cpu",weights_only=True)
                exported={k:v for k,v in best.items() if k not in
                          ("optimizer_state_dict","rng_state","train_cursor","early_stopping_state")}
                exported.update(checkpoint_role="normal_only_ae_best_inference",selected_from=pointer,
                                inference_only=True,accepted_updates=best["global_updates"],
                                seen_normal_examples=best["training_examples_seen"])
                path=destination/"best_inference.pt"
                _checkpoint(path,exported)
                result["best_inference"]=dict(path=str(path),sha256=_sha(path),
                    global_updates=best["global_updates"],epoch=best["epoch"],
                    selection_reconstruction_nats_per_sequence=best["selection_metrics"]["reconstruction_nats_per_sequence"])
            except Exception as error:
                result.update(status="failed",export_error=f"{type(error).__name__}: {error}")
        result["elapsed_seconds"]=time.monotonic()-started
        _json(destination/"summary.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/normal_only_ae.json")
    parser.add_argument("--data-config",type=Path,default=ROOT/"config.json")
    parser.add_argument("--processed-dir",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--smoke-test",action="store_true",help="At most two updates, integration only")
    parser.add_argument("--_worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    cfg=validate_config(json.loads(args.config.read_text(encoding="utf-8")))
    if any(cfg[k]!=DEFAULTS[k] for k in ("embedding_dim","hidden_size","latent_dim","num_layers","dropout")):
        raise ValueError("The CLI retains the complete original architecture")
    if args.execute and not args._worker:
        command=[sys.executable,"-m","src.train_normal_ae",*(sys.argv[1:] if argv is None else argv),"--_worker"]
        try:
            return subprocess.run(command,check=False,timeout=cfg["max_seconds"]+30).returncode
        except subprocess.TimeoutExpired:
            print("Normal-only training exceeded hard time cap; committed checkpoints survive, outputs are partial",file=sys.stderr)
            return 124
    data=load_config(args.data_config)
    if args.processed_dir:
        data["processed_dir"]=args.processed_dir.resolve()
    result=run_normal_only(DataPipeline(data),args.output_dir,config=cfg,device=args.device,
                          execute=args.execute,smoke_test=args.smoke_test)
    print(json.dumps(result if not args.execute else dict(status=result["status"],reason=result.get("reason"),
        output_dir=str(args.output_dir),best_inference=result.get("best_inference")),indent=2))
    return 0 if result.get("status","completed")=="completed" else 1


if __name__=="__main__":
    raise SystemExit(main())
