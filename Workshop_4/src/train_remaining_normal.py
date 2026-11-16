"""One explicitly selected remaining normal-only AE/VAE experiment.

AE z32 is intentionally excluded: it has already been trained. VAE training
uses a sampled reconstruction plus beta*KL; model selection uses deterministic
reconstruction at mu, the same reconstruction criterion as the AE baseline.
No test data, historical optimizer, checkpoint continuation or grid is accepted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from .autoencoder import (Autoencoder, VariationalAutoencoder,
                          reconstruction_loss, kl_divergence)
from .benchmark_ae import _preserve_rng_and_flags
from .data import ROOT, DataPipeline, load_config
from .latent_diagnostics import compare_logits
from . import train_normal_ae as _legacy
from .train_ae import (GRADIENT_CLIPPING, _invalid_training_tensors,
                       _rng_state, seed_everything, stable_clip_grad_norm_)


EXPERIMENTS = {
    "ae_z64": dict(model="ae", latent_dim=64, beta=0.0),
    "vae_z32_b0p5": dict(model="vae", latent_dim=32, beta=0.5),
    "vae_z32_b1": dict(model="vae", latent_dim=32, beta=1.0),
    "vae_z32_b4": dict(model="vae", latent_dim=32, beta=4.0),
}
DEFAULTS = {**_legacy.DEFAULTS, "experiment_id": "ae_z64", **EXPERIMENTS["ae_z64"]}


def validate_config(overrides=None):
    supplied = {} if overrides is None else overrides
    if not isinstance(supplied, dict):
        raise ValueError("One experiment configuration mapping is required")
    identity = supplied.get("experiment_id", "ae_z64")
    if identity not in EXPERIMENTS:
        raise ValueError("Select ae_z64 or exactly one of the three VAE z32 beta runs; AE z32 is excluded")
    cfg = {**DEFAULTS, **EXPERIMENTS[identity], "experiment_id": identity, **supplied}
    if set(cfg) != set(DEFAULTS):
        raise ValueError("Unknown setting: a grid, source checkpoint, resume and test are unsupported")
    required = EXPERIMENTS[identity]
    if any(cfg[key] != value for key, value in required.items()) or type(cfg["latent_dim"]) is not int:
        raise ValueError("Experiment ID, model, latent width and beta must match the fixed declaration")
    if isinstance(cfg["beta"], bool) or not isinstance(cfg["beta"], (float, int)):
        raise ValueError("beta must be the declared numeric coefficient")
    # Preserve all original protections; its z32-only validation is projected
    # solely for validating common controls, not for creating/training a model.
    common = {key: cfg[key] for key in _legacy.DEFAULTS}
    common["latent_dim"] = 32
    _legacy.validate_config(common)
    return cfg


def _common_config(cfg):
    common = {key: cfg[key] for key in _legacy.DEFAULTS}
    common["latent_dim"] = 32
    return common


def training_decision(selection_ce, swap_damage, state, config):
    cfg = validate_config(config)
    return _legacy.training_decision(selection_ce, swap_damage, state, _common_config(cfg))


_deadline = _legacy._deadline
_json = _legacy._json
_checkpoint = _legacy._checkpoint
_sha = _legacy._sha


def build_model(config, vocab_size, sequence_length):
    cfg = validate_config(config)
    hparams = {key: cfg[key] for key in
               ("embedding_dim", "hidden_size", "latent_dim", "num_layers", "dropout")}
    kind = Autoencoder if cfg["model"] == "ae" else VariationalAutoencoder
    return kind(vocab_size=vocab_size, sequence_length=sequence_length, **hparams)


def training_terms(model, x, config):
    """Sample only for VAE training; no clipping or posterior modification."""
    cfg = validate_config(config)
    output = model(x, sample=True) if cfg["model"] == "vae" else model(x)
    reconstruction = reconstruction_loss(output.logits, x)
    kl = kl_divergence(output.mu, output.logvar) if cfg["model"] == "vae" else reconstruction.new_zeros(())
    weighted = cfg["beta"] * kl
    return dict(output=output, reconstruction=reconstruction, kl=kl,
                weighted_kl=weighted, total=reconstruction + weighted)


@torch.no_grad()
def evaluate_model_selection(model, x, batch_size=128, diagnostic_examples=128,
                             deadline=None, config=None):
    """Normal selection reconstruction at z (AE) or mu (VAE), no MC claim.

    The at-mu weighted objective is not a Monte Carlo negative ELBO. It is
    logged separately; only reconstruction controls model selection/early stop.
    """
    cfg = validate_config(config)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError("At least two normal model-selection examples are required")
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        device = next(model.parameters()).device
        recon_sum = kl_sum = 0.0
        mu_chunks, logvar_chunks, kl_dimension_chunks = [], [], []
        for start in range(0, len(x), batch_size):
            if deadline is not None:
                _deadline(deadline)
            batch = x[start:start + batch_size].to(device)
            output = model(batch, sample=False) if cfg["model"] == "vae" else model(batch)
            values = [output.logits, output.z]
            if cfg["model"] == "vae":
                values += [output.mu, output.logvar]
            if any(not torch.isfinite(v).all().item() for v in values):
                raise FloatingPointError("Nonfinite normal selection output/posterior")
            recon_sum += reconstruction_loss(output.logits.double(), batch, reduction="sum").item()
            if cfg["model"] == "vae":
                mu, logvar = output.mu.double(), output.logvar.double()
                per_dim = .5 * (mu.square() + logvar.exp() - 1 - logvar)
                if not torch.isfinite(per_dim).all().item():
                    raise FloatingPointError("Nonfinite posterior KL during model selection")
                kl_sum += per_dim.sum().item()
                mu_chunks.append(mu.cpu())
                logvar_chunks.append(logvar.cpu())
                kl_dimension_chunks.append(per_dim.cpu())
        probe = x[:min(diagnostic_examples, len(x))].to(device)
        captures = {}
        hook = model.decoder.register_forward_hook(
            lambda m, i, o: captures.update(hidden=o[0].detach(), h0=i[1].detach()))
        try:
            if cfg["model"] == "vae":
                z, _ = model.encode(probe)
            else:
                z = model.encode(probe)
            original = model.decode(z)
        finally:
            hook.remove()
        if not torch.isfinite(z).all().item():
            raise FloatingPointError("Nonfinite selection latent")
        permutation = torch.arange(len(probe), device=device).roll(len(probe)//2)
        swapped, zero = model.decode(z[permutation]), model.decode(torch.zeros_like(z))
        comparison = compare_logits(original, swapped, probe)
        zero_comparison = compare_logits(original, zero, probe)
        recon, kl = recon_sum/len(x), kl_sum/len(x)
        report = dict(scope="normal_model_selection_validation_only", examples=len(x),
            reconstruction_nats_per_sequence=recon, reconstruction_nats_per_character=recon/x.shape[1],
            kl_nats_per_sequence=kl, weighted_kl_nats_per_sequence=cfg["beta"]*kl,
            objective_at_mu_nats_per_sequence=recon+cfg["beta"]*kl,
            reconstruction_latent="posterior_mu" if cfg["model"] == "vae" else "deterministic_z",
            selection_definition="deterministic reconstruction; at-mu objective is not MC ELBO",
            diagnostic_examples=len(probe),
            swap_damage_nats_per_sequence=comparison["delta_nats_per_sequence"]["mean"],
            swap_probability_tv=comparison["probability_tv_mean"],
            swap_probability_js_nats=comparison["probability_js_nats_mean"],
            swap_top1_change_fraction=comparison["top1_change_fraction"],
            zero_damage_nats_per_sequence=zero_comparison["delta_nats_per_sequence"]["mean"],
            z_max_abs=z.abs().max().item(), z_rms=z.double().square().mean().sqrt().item(),
            z_between_example_std_rms=z.double().var(0,unbiased=False).mean().sqrt().item(),
            decoder_saturation_fraction=(captures["hidden"].abs()>=.99).double().mean().item(),
            initial_hidden_saturation_fraction=(captures["h0"].abs()>=.99).double().mean().item(),
            saturation_definition="fraction abs(hidden)>=.99; not gate probabilities",
            swap_source_target_token_agreement=(probe==probe[permutation]).double().mean().item(),
            swap_per_position=comparison["per_position"],
            utility_limit="descriptive normal subset; response at mu does not measure mutual information")
        if mu_chunks:
            mu, logvar = torch.cat(mu_chunks), torch.cat(logvar_chunks)
            per_dim = torch.cat(kl_dimension_chunks).mean(0)
            variance = mu.var(0,unbiased=False)
            report["posterior"] = dict(mu_rms=mu.square().mean().sqrt().item(),
                mu_between_example_std_rms=variance.mean().sqrt().item(),
                logvar_min=logvar.min().item(), logvar_max=logvar.max().item(),
                logvar_mean=logvar.mean().item(), kl_per_dimension=per_dim.tolist(),
                active_units=int((variance > .01).sum().item()),
                active_unit_definition="variance of posterior mean across selection examples > .01",
                mean_posterior_variance=logvar.exp().mean().item())
        else:
            report["posterior"] = None
        return report
    finally:
        for module, mode in modes.items():
            module.training = mode


def run_remaining_normal(pipeline, output_dir, *, config=None, device="cpu", execute=False, smoke_test=False):
    cfg=validate_config(config)
    if type(execute) is not bool or type(smoke_test) is not bool:
        raise ValueError("execute/smoke_test must be bool")
    from .normal_only_protocol import (role_partition, fit_normal_class, prepare_role_samples,
                                       role_tensor, iter_normal_train_batches, build_train_input_guard)
    partition=role_partition(pipeline,cfg["seed"])
    destination=Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Normal-only training requires a new output directory")
    plan=dict(operation="single_from_scratch_remaining_normal_model",experiment_id=cfg["experiment_id"],execute=execute,
              smoke_test=smoke_test,config=cfg,validation_partition=partition,
              initialization="new declared AE/VAE weights and new Adam; no historical checkpoint",
              model_selection="deterministic normal reconstruction at z (AE) or posterior mean (VAE); calibration/evaluation scores unavailable",
              vae_training="sampled reconstruction + beta*KL; at-mu selection objective is separately labeled, not MC ELBO",
              test_access="none",effective_max_updates=min(2,cfg["max_updates"]) if smoke_test else cfg["max_updates"],
              early_stopping_unit="model-selection evaluations, including epoch-end evaluations",
              grid="none; exactly one declared AE z64 or VAE z32 beta experiment; AE z32 excluded")
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
                initialization="from_scratch",smoke_test=smoke_test,test_accessed=False,experiment_id=cfg["experiment_id"],model_type=cfg["model"],beta=cfg["beta"])
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
            model=build_model(cfg,len(pipeline.vocab["itos"]),pipeline.config["window_size"]).to(selected)
            optimizer=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
            training_config={**cfg,"model":cfg["model"],"normal_class":normal["normal_class"],
                             "from_scratch":True,"smoke_test":smoke_test}
            protocol=dict(experiment_id=cfg["experiment_id"],model=cfg["model"],beta=cfg["beta"],from_scratch=True,train_filter="y_id == normal_class",normal_class=normal["normal_class"],
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
                    "train_kl_nats_per_sequence","train_weighted_kl_nats_per_sequence","train_objective_nats_per_sequence",
                    "selection_kl_nats_per_sequence","selection_weighted_kl_nats_per_sequence","selection_objective_at_mu_nats_per_sequence",
                    "posterior_active_units","posterior_logvar_min","posterior_logvar_max",
                    "selection_reconstruction_nats_per_sequence","selection_reconstruction_nats_per_character",
                    "swap_damage_nats_per_sequence","swap_probability_tv","z_max_abs",
                    "clipped_fraction","grad_norm_pre_max","grad_norm_post_max",
                    "stale_evaluations","warning","stop","elapsed_seconds"]
            writer=csv.DictWriter(history_stream,fieldnames=fields)
            writer.writeheader()
            total_loss=0.0
            total_kl=total_objective=0.0
            epoch_kl=epoch_objective=0.0
            epoch_loss=0.0
            epoch_examples=0
            clipped=0
            grad_pre_max=grad_post_max=0.0

            def snapshot(epoch,batch_index):
                nonlocal state,last_evaluated,stage
                stage="model_selection"
                _deadline(deadline)
                metrics=evaluate_model_selection(model,selection_x,cfg["batch_size"],cfg["diagnostic_examples"],deadline,config=cfg)
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
                payload=dict(format_version=1,checkpoint_role="remaining_normal_periodic_state",model_type=cfg["model"],experiment_id=cfg["experiment_id"],beta=cfg["beta"],
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
                                 selection_scope="deterministic normal reconstruction at z/mean; no calibration/evaluation selection")
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
                    train_kl_nats_per_sequence=epoch_kl/epoch_examples if epoch_examples else None,
                    train_weighted_kl_nats_per_sequence=cfg["beta"]*epoch_kl/epoch_examples if epoch_examples else None,
                    train_objective_nats_per_sequence=epoch_objective/epoch_examples if epoch_examples else None,
                    selection_kl_nats_per_sequence=metrics["kl_nats_per_sequence"],
                    selection_weighted_kl_nats_per_sequence=metrics["weighted_kl_nats_per_sequence"],
                    selection_objective_at_mu_nats_per_sequence=metrics["objective_at_mu_nats_per_sequence"],
                    posterior_active_units=metrics["posterior"]["active_units"] if metrics["posterior"] else None,
                    posterior_logvar_min=metrics["posterior"]["logvar_min"] if metrics["posterior"] else None,
                    posterior_logvar_max=metrics["posterior"]["logvar_max"] if metrics["posterior"] else None,
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
                        epoch_kl=epoch_objective=0.0
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
                                terms=training_terms(model,x,cfg)
                                output=terms["output"]
                                recon,kl,weighted_kl,loss=(terms[k] for k in ("reconstruction","kl","weighted_kl","total"))
                                if (any(not torch.isfinite(terms[k]).item() for k in ("reconstruction","kl","weighted_kl","total"))
                                        or not torch.isfinite(output.z).all().item()
                                        or (cfg["model"]=="vae" and (not torch.isfinite(output.mu).all().item() or not torch.isfinite(output.logvar).all().item()))):
                                    raise FloatingPointError("Nonfinite normal training loss/latent before Adam")
                                z_max=output.z.detach().abs().max().item()
                                if z_max>cfg["max_latent_abs"]:
                                    raise FloatingPointError("Normal training latent magnitude guard")
                                if recon.item()>max(state["best_reconstruction"],1e-12)*cfg["ce_severe_ratio"]*2:
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
                                total_loss+=recon.detach().item()*len(x)
                                epoch_loss+=recon.detach().item()*len(x)
                                total_kl+=kl.detach().item()*len(x)
                                total_objective+=loss.detach().item()*len(x)
                                epoch_kl+=kl.detach().item()*len(x)
                                epoch_objective+=loss.detach().item()*len(x)
                                clipped+=int(clipping["clipped"])
                                grad_pre_max=max(grad_pre_max,clipping["norm_before"])
                                grad_post_max=max(grad_post_max,clipping["norm_after"])
                                record=dict(epoch=epoch,batch_index=batch_index,global_updates=result["global_updates"],
                                    examples=len(x),normal_class=normal["normal_class"],all_labels_normal=True,
                                    gradient_reset_confirmed=True,
                                    reconstruction_nats_per_sequence=recon.item(),kl_nats_per_sequence=kl.item(),
                                    weighted_kl_nats_per_sequence=weighted_kl.item(),objective_nats_per_sequence=loss.item(),
                                    beta=cfg["beta"],training_reconstruction_latent="sampled_z" if cfg["model"]=="vae" else "deterministic_z",
                                    z_max_abs=z_max,clipping=clipping)
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
                exported.update(checkpoint_role="remaining_normal_best_inference",selected_from=pointer,
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
    parser.add_argument("--config",type=Path,required=True,help="Exactly one of the four remaining-model configs")
    parser.add_argument("--data-config",type=Path,default=ROOT/"config.json")
    parser.add_argument("--processed-dir",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--smoke-test",action="store_true",help="At most two updates; integration only")
    parser.add_argument("--_worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    cfg=validate_config(json.loads(args.config.read_text(encoding="utf-8")))
    if any(cfg[k]!=_legacy.DEFAULTS[k] for k in ("embedding_dim","hidden_size","num_layers","dropout")):
        raise ValueError("The CLI retains all original dimensions except the declared requested latent width")
    if args.execute and not args._worker:
        command=[sys.executable,"-m","src.train_remaining_normal",*(sys.argv[1:] if argv is None else argv),"--_worker"]
        try:return subprocess.run(command,check=False,timeout=cfg["max_seconds"]+30).returncode
        except subprocess.TimeoutExpired:
            print("Remaining-model training exceeded wall cap; committed checkpoints survive, outputs are partial",file=sys.stderr)
            return 124
    data=load_config(args.data_config)
    if args.processed_dir:data["processed_dir"]=args.processed_dir.resolve()
    result=run_remaining_normal(DataPipeline(data),args.output_dir,config=cfg,device=args.device,
                               execute=args.execute,smoke_test=args.smoke_test)
    print(json.dumps(result if not args.execute else dict(status=result["status"],reason=result.get("reason"),
        experiment_id=cfg["experiment_id"],output_dir=str(args.output_dir),best_inference=result.get("best_inference")),indent=2))
    return 0 if result.get("status","completed")=="completed" else 1


if __name__=="__main__":raise SystemExit(main())
