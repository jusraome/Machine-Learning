"""Common development-only cohorts for frozen normal AE/VAE comparisons.

This module has no training or test mode. VAE inference uses posterior mu;
NLL+KL is a separately named score, never MSE+KL or a Monte Carlo ELBO.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from .autoencoder import Autoencoder, VariationalAutoencoder
from .anomaly_evaluation import conformal_threshold, _score_summary, _preserve_random
from .evaluate_normal_ae import (complete_detection_metrics, _canonical, _write, _seal,
    _sha, _entries_record, _quarantine_best8_training_overlap, _digest_model)
from .normal_only_protocol import (role_partition, build_train_input_guard,
    prepare_role_samples, role_tensor, _row_hash)
from .latent_diagnostics import compare_logits
from .data import ROOT, DataPipeline, load_config


COMMON_SCORES = ("brier_mse", "nll_nats_per_sequence")
VAE_SCORES = ("nll_plus_kl_nats_per_sequence", "nll_plus_training_beta_kl_nats_per_sequence")
IDENTITIES = {"ae_z32": ("ae", 32, 0.), "ae_z64": ("ae", 64, 0.),
    "vae_z32_b0p5": ("vae", 32, .5), "vae_z32_b1": ("vae", 32, 1.),
    "vae_z32_b4": ("vae", 32, 4.), "best8": ("ae", 32, 0.)}
DEFAULTS = dict(seed=42, alpha=.05, primary_score="brier_mse", max_calibration=4096,
    max_evaluation=8192, per_speaker_cap=512, batch_size=128,
    read_batch_size=4096, max_seconds=1800, diagnostic_examples=128)


def validate_config(overrides=None):
    cfg = {**DEFAULTS, **(overrides or {})}
    if set(cfg) != set(DEFAULTS):
        raise ValueError("Unknown comparison setting; training, winner selection and test are unsupported")
    for name, lo, hi in (("seed",0,2**31-1),("max_calibration",1,8192),
            ("max_evaluation",1,16384),("per_speaker_cap",1,4096),("batch_size",1,256),
            ("read_batch_size",1,8192),("max_seconds",1,3600),("diagnostic_examples",2,128)):
        if type(cfg[name]) is not int or not lo <= cfg[name] <= hi:
            raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    if cfg["max_calibration"] + cfg["max_evaluation"] > 16384:
        raise ValueError("At most 16384 unique calibration/development examples are permitted")
    if (isinstance(cfg["alpha"],bool) or not isinstance(cfg["alpha"],(float,int))
            or not math.isfinite(cfg["alpha"]) or not 0 < cfg["alpha"] < 1):
        raise ValueError("alpha must be finite and in (0,1)")
    if cfg["primary_score"] not in COMMON_SCORES:
        raise ValueError("Predeclare a common reconstruction score before development evaluation")
    return cfg


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Common development comparison wall-time budget exhausted")


def read_reference_protocol(path):
    """Accept the existing AE-z32 sealed development artifact or this format."""
    path = Path(path).resolve()
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("Reference needs frozen_protocol.json and its SHA-256 sidecar")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if sidecar.read_text(encoding="ascii").strip() != digest:
        raise ValueError("Reference protocol SHA-256 mismatch")
    value = json.loads(raw.decode("utf-8"))
    if _sha(path) != digest:
        raise ValueError("Reference changed during reading")
    if (value.get("schema_version"), value.get("stage")) not in (
            (1,"calibrated_before_evaluation"), (2,"common_development_calibrated")):
        raise ValueError("Reference is not an existing sealed development cohort")
    if set(value.get("selected_rows",{})) != {"calibration","evaluation"}:
        raise ValueError("Reference must contain exact calibration and evaluation row records")
    if not value["selected_rows"]["calibration"] or not value["selected_rows"]["evaluation"]:
        raise ValueError("Reference populations must both be nonempty")
    if sum(len(x) for x in value["selected_rows"].values()) > 16384:
        raise ValueError("Reference exceeds the bounded comparison population")
    for role, records in value["selected_rows"].items():
        expected = value.get("selected_rows_sha256",{}).get(role)
        if expected != hashlib.sha256(_canonical(records)).hexdigest():
            raise ValueError("Reference selected-row fingerprint differs")
    return value, digest


def resolve_reference_rows(pipeline, reference, partition, normal, guard, deadline, read_batch_size=4096):
    """Resolve exact IDs, verify contents/roles and never refill a missing row."""
    if reference.get("data_contract") != pipeline.contract():
        raise ValueError("Reference data contract differs")
    if reference.get("normal_class") != normal or reference.get("role_digest") != partition["role_digest"]:
        raise ValueError("Reference class or role partition differs")
    if reference.get("train_input_guard") != guard["metadata"]:
        raise ValueError("Reference normal training inputs differ")
    declared = reference.get("partition", {})
    if declared.get("roles") != partition["roles"]:
        raise ValueError("Reference speaker roles differ")
    wanted = {}
    resolved = {"calibration":{}, "evaluation":{}}
    for role, records in reference["selected_rows"].items():
        for record in records:
            index = record.get("row_index")
            if type(index) is not int or index < 0 or index in wanted:
                raise ValueError("Reference IDs must be nonnegative and disjoint across roles")
            wanted[index] = (role, record)
    reader = pipeline._iter_rows_batches("val",batch_size=read_batch_size)
    index = 0
    try:
        for rows in reader:
            _deadline(deadline)
            _, labels = pipeline._validate_rows(rows,"val")
            for row,label in zip(rows,labels):
                pair = wanted.get(index)
                if pair:
                    role, record = pair
                    input_hash = hashlib.sha256(row["x"].encode("utf-8")).hexdigest()
                    if (record.get("row_sha256") != _row_hash(row)
                            or record.get("character_id") != row["character_id"]
                            or record.get("label") != label
                            or row["character_id"] not in partition["roles"][role]
                            or (record.get("input_sha256") is not None and record["input_sha256"] != input_hash)):
                        raise ValueError("Reference row content, label or speaker differs")
                    if role == "calibration" and label != normal:
                        raise ValueError("Reference calibration contains anomalies")
                    if input_hash in guard["input_hashes"]:
                        raise ValueError("Reference contains a normal-training input; cannot silently change cohort")
                    resolved[role][index] = dict(row=row,row_index=index,label=label,
                        row_sha256=_row_hash(row),input_sha256=input_hash)
                index += 1
    finally:
        reader.close()
    if sum(len(x) for x in resolved.values()) != len(wanted):
        raise ValueError("Some reference rows are missing; no resampling is allowed")
    return {role:[resolved[role][record["row_index"]] for record in reference["selected_rows"][role]]
            for role in resolved}


def _load_model(pipeline, label, path, device):
    model_type, latent, beta = IDENTITIES[label]
    path = Path(path).resolve()
    digest = _sha(path)
    checkpoint = torch.load(path,map_location="cpu",weights_only=True)
    pipeline.validate_checkpoint(checkpoint)
    hp = checkpoint["model_hparams"]
    if (checkpoint.get("model_type") != model_type or hp.get("latent_dim") != latent
            or any(hp.get(k) != v for k,v in dict(embedding_dim=32,hidden_size=128,num_layers=1,dropout=0.).items())
            or hp.get("latent_activation","identity") != "identity"):
        raise ValueError(f"Checkpoint architecture does not match {label}")
    if label == "best8":
        if checkpoint.get("epoch") != 8:
            raise ValueError("best8 must be the historical epoch8 general AE")
    else:
        training = checkpoint.get("training_config", {})
        proof = checkpoint.get("normal_only_protocol", {})
        definition = checkpoint.get("normal_definition", {})
        allowed_role = "normal_only_ae_best_inference" if label == "ae_z32" else "remaining_normal_best_inference"
        if checkpoint.get("checkpoint_role") != allowed_role:
            raise ValueError("Select a frozen trained inference checkpoint, not an initial/periodic model")
        if (not training.get("from_scratch") or not proof.get("from_scratch")
                or checkpoint.get("source_checkpoint") is not None or training.get("smoke_test")
                or proof.get("experiment_scope") == "smoke_integration"
                or type(checkpoint.get("global_updates")) is not int or checkpoint["global_updates"] < 1):
            raise ValueError("All normal-only candidates must be genuinely trained from scratch")
        normal = definition.get("normal_class")
        if (type(normal) is not int or not 0 <= normal < len(pipeline.vocab["itos"])
                or training.get("normal_class") != normal or proof.get("normal_class") != normal
                or definition.get("fitted_split") != "train"):
            raise ValueError("Normal class provenance differs")
        if label != "ae_z32" and checkpoint.get("experiment_id",training.get("experiment_id")) != label:
            raise ValueError("Saved experiment identity does not match the requested label")
        if model_type == "vae" and (isinstance(training.get("beta"),bool)
                or not isinstance(training.get("beta"),(float,int)) or training["beta"] != beta):
            raise ValueError("VAE beta does not match its experiment identity")
    cls = Autoencoder if model_type == "ae" else VariationalAutoencoder
    model = cls(**hp).to(device)
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    if any(not torch.isfinite(p).all().item() for p in model.parameters()):
        raise FloatingPointError("Nonfinite frozen model parameters")
    model.eval()
    return model, checkpoint, dict(path=str(path),sha256=digest,model_type=model_type,
        latent_dim=latent,beta=beta,epoch=checkpoint.get("epoch"),global_updates=checkpoint.get("global_updates"),
        checkpoint_role=checkpoint.get("checkpoint_role"),inference="posterior_mu" if model_type=="vae" else "deterministic_z")


@torch.no_grad()
def reconstruction_and_kl_scores(model, x, beta=0.):
    """Deterministic FP64 scores; NLL(mu)+KL is not an MC negative ELBO."""
    modes = {module:module.training for module in model.modules()}
    try:
        model.eval()
        output = model(x,sample=False) if isinstance(model,VariationalAutoencoder) else model(x)
        logits = output.logits.detach().double()
        if logits.ndim != 3 or logits.shape[:2] != x.shape or not torch.isfinite(logits).all().item():
            raise FloatingPointError("Invalid/nonfinite frozen reconstruction logits")
        logp = F.log_softmax(logits,dim=-1)
        nll = -logp.gather(-1,x.unsqueeze(-1)).squeeze(-1).sum(-1)
        p = logp.exp()
        mse = ((p.square().sum(-1)-2*p.gather(-1,x.unsqueeze(-1)).squeeze(-1)+1).mean(-1)/logits.shape[-1]).clamp_min(0)
        scores = dict(brier_mse=mse,nll_nats_per_sequence=nll)
        latent = output.z.detach().double()
        posterior = {}
        if isinstance(model,VariationalAutoencoder):
            if not torch.equal(output.z,output.mu):
                raise RuntimeError("VAE development inference must use posterior mu")
            mu, logvar = output.mu.detach().double(), output.logvar.detach().double()
            kl_dim = .5*(mu.square()+logvar.exp()-1-logvar)
            kl = kl_dim.sum(-1)
            scores.update(nll_plus_kl_nats_per_sequence=nll+kl,
                          nll_plus_training_beta_kl_nats_per_sequence=nll+beta*kl)
            posterior = dict(kl=kl,kl_dim=kl_dim,logvar=logvar)
        if (not torch.isfinite(latent).all().item() or any(not torch.isfinite(v).all().item() for v in scores.values())
                or any(not torch.isfinite(v).all().item() for v in posterior.values())):
            raise FloatingPointError("Nonfinite latent, KL or anomaly score")
        return scores, latent, posterior
    finally:
        for module, mode in modes.items():module.training=mode


def _score_population(pipeline,model,entries,normal,cfg,beta,deadline,diagnostics=False):
    rows, latents, kldims, logvars = [], [], [], []
    for start in range(0,len(entries),cfg["batch_size"]):
        _deadline(deadline)
        part = entries[start:start+cfg["batch_size"]]
        x = role_tensor(pipeline,part,split="val")["x"].to(next(model.parameters()).device)
        scores,z,posterior = reconstruction_and_kl_scores(model,x,beta)
        latents.append(z.cpu())
        if posterior:
            kldims.append(posterior["kl_dim"].cpu())
            logvars.append(posterior["logvar"].cpu())
        for i,entry in enumerate(part):
            rows.append(dict(row_index=entry["row_index"],row_sha256=entry["row_sha256"],input_sha256=entry["input_sha256"],
                character_id=entry["row"]["character_id"],y_id=entry["label"],is_anomaly=int(entry["label"]!=normal),
                **{name:values[i].item() for name,values in scores.items()},
                posterior_kl_nats=posterior["kl"][i].item() if posterior else None))
    if not rows:
        raise ValueError("No retained examples remain for scoring")
    z = torch.cat(latents)
    stats = dict(examples=len(rows),latent_dim=z.shape[-1],latent_between_example_std_rms=z.var(0,unbiased=False).mean().sqrt().item(),
        latent_rms=z.square().mean().sqrt().item(),active_latent_units_variance_gt_0p01=int((z.var(0,unbiased=False)>.01).sum()),
        active_unit_definition="coordinate variance >0.01 on this development population; scale-dependent diagnostic")
    if kldims:
        kl, lv = torch.cat(kldims), torch.cat(logvars)
        stats.update(active_mu_units_variance_gt_0p01=stats["active_latent_units_variance_gt_0p01"],
            mean_kl_nats_per_sequence=kl.sum(-1).mean().item(),mean_kl_per_dimension=kl.mean(0).tolist(),
            active_kl_units_mean_gt_0p01=int((kl.mean(0)>.01).sum()),logvar_mean=lv.mean().item(),
            logvar_min=lv.min().item(),logvar_max=lv.max().item(),
            objective_at_mu_nats_per_sequence=sum(r["nll_plus_training_beta_kl_nats_per_sequence"] for r in rows)/len(rows),
            objective_at_mu_definition="deterministic NLL(mu)+training_beta*KL, not MC ELBO")
    if diagnostics and len(entries)>1:
        subset = entries[:min(cfg["diagnostic_examples"],len(entries))]
        x = role_tensor(pipeline,subset,split="val")["x"].to(next(model.parameters()).device)
        with torch.no_grad():
            code=model.encode(x)
            code=code[0] if isinstance(model,VariationalAutoencoder) else code
            original=model.decode(code)
            permutation=torch.arange(len(code),device=code.device).roll(len(code)//2)
            swapped=model.decode(code[permutation])
            difference=compare_logits(original,swapped,x)
        stats.update(diagnostic_examples=len(subset),swap_damage_nats_per_sequence=difference["delta_nats_per_sequence"]["mean"],
            swap_probability_tv=difference["probability_tv_mean"],swap_per_position=difference["per_position"],
            swap_scope="first fixed retained development examples; posterior mu for VAE")
    return rows, stats


def run_development_comparison(pipeline,checkpoints,output_dir,*,config=None,reference_protocol=None,device="cpu",execute=False):
    cfg=validate_config(config)
    if (not isinstance(checkpoints,dict) or not checkpoints or set(checkpoints)-set(IDENTITIES)
            or set(checkpoints)=={"best8"} or type(execute) is not bool):
        raise ValueError("Provide uniquely named supported normal-only checkpoints, optionally best8")
    destination=Path(output_dir).resolve()
    if destination.exists():raise FileExistsError("Choose a new comparison output directory")
    for source in [*checkpoints.values(),reference_protocol]:
        if source is not None and (destination==Path(source).resolve() or destination in Path(source).resolve().parents):
            raise ValueError("Output cannot contain a source artifact")
    plan=dict(operation="common_normal_AE_VAE_development_comparison",execute=execute,checkpoints={k:str(v) for k,v in checkpoints.items()},
        reference_protocol=str(reference_protocol) if reference_protocol else None,config=cfg,device=device,
        test_access="closed; no test mode exists",optimizer_steps=0,primary_score=cfg["primary_score"],
        model_selection="normal-only model_selection role in independent completed training runs",
        cohort="exact sealed reference rows, no refill" if reference_protocol else "one shared new cohort for all supplied checkpoints",
        vae_inference="posterior mu; reconstruction and KL separate; NLL+KL secondary, never MSE+KL",
        winner_selection="none; development comparisons are descriptive")
    if not execute:return plan
    selected=torch.device(device)
    if selected.type not in ("cpu","cuda") or selected.type=="cuda" and not torch.cuda.is_available():raise ValueError("Unavailable device")
    destination.mkdir(parents=True,exist_ok=False)
    _write(destination/"plan.json",plan)
    started=time.monotonic();deadline=started+cfg["max_seconds"]
    sources={};report=dict(status="running",plan=plan,optimizer_steps=0,test_accessed=False)
    try:
        with _preserve_random():
            models={};saved={};info={}
            for label,path in checkpoints.items():
                _deadline(deadline)
                models[label],saved[label],info[label]=_load_model(pipeline,label,path,selected)
                sources[info[label]["path"]]=info[label]["sha256"]
            first=next(label for label in models if label!="best8")
            normal=saved[first]["normal_definition"]["normal_class"]
            partition=role_partition(pipeline,cfg["seed"])
            guard=build_train_input_guard(pipeline,normal,deadline=deadline,read_batch_size=cfg["read_batch_size"])
            for label,checkpoint in saved.items():
                if label=="best8":continue
                if (checkpoint["normal_definition"]["normal_class"]!=normal
                        or checkpoint.get("validation_partition",{}).get("role_digest")!=partition["role_digest"]
                        or checkpoint.get("validation_partition",{}).get("roles")!=partition["roles"]
                        or checkpoint.get("normal_only_protocol",{}).get("role_digest")!=partition["role_digest"]
                        or checkpoint.get("train_input_guard")!=guard["metadata"]):
                    raise ValueError("Models differ in normal definition, role partition or training input guard")
            reference=None;reference_digest=None
            if reference_protocol:
                reference,reference_digest=read_reference_protocol(reference_protocol)
                sources[str(Path(reference_protocol).resolve())]=reference_digest
                sidecar=Path(reference_protocol).resolve().with_suffix(".sha256")
                sources[str(sidecar)]=_sha(sidecar)
                for key in ("seed","alpha","primary_score"):
                    if reference.get("config",{}).get(key)!=cfg[key]:raise ValueError("Comparison changes a reference score/calibration/partition setting")
                samples=resolve_reference_rows(pipeline,reference,partition,normal,guard,deadline,cfg["read_batch_size"])
                reference_models=reference.get("models",{})
                ae32_key="normal_only" if "normal_only" in reference_models else "ae_z32"
                if "ae_z32" in info and ae32_key in reference_models:
                    if info["ae_z32"]["sha256"]!=reference_models[ae32_key]["sha256"]:
                        raise ValueError("AE z32 differs from the historical reference model")
                if "best8" in info and "best8" in reference.get("models",{}):
                    if info["best8"]["sha256"]!=reference["models"]["best8"]["sha256"]:raise ValueError("best8 differs from reference")
                sampling=dict(source="exact sealed reference IDs, order and hashes; no refill",reference_sha256=reference_digest)
            else:
                prepared=prepare_role_samples(pipeline,partition,normal,roles=("calibration","evaluation"),
                    max_per_role={"calibration":cfg["max_calibration"],"evaluation":cfg["max_evaluation"]},
                    per_speaker_cap=cfg["per_speaker_cap"],read_batch_size=cfg["read_batch_size"],deadline=deadline,
                    exclude_input_hashes=guard["input_hashes"])
                samples,sampling=prepared["samples"],prepared["sampling"]
            quarantine=dict(applied=False,reason="No all-class best8 supplied")
            if "best8" in models:
                filtered,quarantine=_quarantine_best8_training_overlap(pipeline,samples,normal,cfg,deadline)
                if reference is not None and any(len(filtered[r])!=len(samples[r]) for r in samples):
                    raise ValueError("Reference cohort is unsafe for this best8 comparison; refusing to alter exact reference rows")
                samples=filtered
            cohort=dict(data_contract=pipeline.contract(),normal_class=normal,normal_definition=saved[first]["normal_definition"],
                role_digest=partition["role_digest"],partition=partition,train_input_guard=guard["metadata"],
                selected_rows={r:_entries_record(entries) for r,entries in samples.items()},
                selected_rows_sha256={r:hashlib.sha256(_canonical(_entries_record(entries))).hexdigest() for r,entries in samples.items()},
                reference_protocol_sha256=reference_digest,exact_reference_reused=reference is not None,
                paired_best8_quarantine=quarantine)
            _write(destination/"sampling.json",sampling)
            _write(destination/"cohort.json",cohort)
            before={label:_digest_model(model) for label,model in models.items()}
            populations={};thresholds={};calibration_stats={}
            for label,model in models.items():
                rows,stats=_score_population(pipeline,model,samples["calibration"],normal,cfg,info[label]["beta"],deadline)
                if any(r["is_anomaly"] for r in rows):raise RuntimeError("Anomalies entered calibration")
                populations[label,"calibration"]=rows;calibration_stats[label]=stats
                score_names=(*COMMON_SCORES,*VAE_SCORES) if info[label]["model_type"]=="vae" else COMMON_SCORES
                thresholds[label]={score:conformal_threshold([r[score] for r in rows],cfg["alpha"]) for score in score_names}
            frozen=dict(schema_version=2,stage="common_development_calibrated",config=cfg,**cohort,
                primary_score=cfg["primary_score"],score_direction="higher is anomaly; strict score > threshold",
                models={label:{**info[label],"thresholds":thresholds[label]} for label in models},
                freeze_point="calibration complete before every mixed-development forward",test_access="closed",
                test_ready=False,reason="This delivery prepares development comparison only; test remains closed")
            frozen_digest=_seal(destination/"frozen_protocol.json",frozen)
            metrics={};reconstruction={};flat=[]
            for label,model in models.items():
                rows,stats=_score_population(pipeline,model,samples["evaluation"],normal,cfg,info[label]["beta"],deadline,True)
                populations[label,"evaluation"]=rows;reconstruction[label]=stats
                metrics[label]={}
                for score,threshold in thresholds[label].items():
                    value=complete_detection_metrics([r[score] for r in rows],[r["is_anomaly"] for r in rows],threshold["threshold"])
                    value.update(normal_score_distribution=_score_summary([r[score] for r in rows if not r["is_anomaly"]]),
                        anomaly_score_distribution=_score_summary([r[score] for r in rows if r["is_anomaly"]]))
                    metrics[label][score]=value
                    flat.append(dict(model=label,score=score,examples=value["examples"],anomaly_prevalence=value["anomaly_prevalence"],
                        roc_auc=value["roc_auc"],average_precision=value["average_precision"],pr_auc_trapezoidal=value["pr_auc_trapezoidal"],
                        precision=value["precision"],recall=value["recall"],f1=value["f1"],false_positive_rate=value["false_positive_rate"],
                        threshold=value["threshold"]))
            if any(before[label]!=_digest_model(model) or any(p.grad is not None for p in model.parameters()) for label,model in models.items()):
                raise RuntimeError("Frozen scoring mutated weights or gradients")
            with (destination/"scores.csv").open("x",newline="",encoding="utf-8") as stream:
                fields=["model","population","row_index","row_sha256","input_sha256","character_id","y_id","is_anomaly",*COMMON_SCORES,*VAE_SCORES,"posterior_kl_nats"]
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                for (label,population),rows in populations.items():writer.writerows(dict(model=label,population=population,**r) for r in rows)
            with (destination/"metrics.csv").open("x",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
            _write(destination/"metrics.json",dict(population="common_development",primary_score=cfg["primary_score"],models=metrics))
            _write(destination/"reconstruction_latent.json",dict(evaluation=reconstruction,calibration=calibration_stats,
                vae_score_convention="posterior mu; deterministic NLL+KL secondary; training-beta weighted alternative separately named; not MC ELBO"))
            report.update(status="completed",metrics=metrics,reconstruction_latent=reconstruction,
                evaluation_examples=len(samples["evaluation"]),calibration_examples=len(samples["calibration"]),
                exact_reference_reused=reference is not None,frozen_protocol_sha256=frozen_digest,
                frozen_protocol_path=str(destination/"frozen_protocol.json"),model_unchanged=True,
                calibration_status="finite_thresholds" if all(t["threshold"] is not None for scores in thresholds.values() for t in scores.values()) else "insufficient_normal_calibration",
                winner_selected=False,test_ready=False)
    except Exception as error:
        report.update(status="failed",reason=f"{type(error).__name__}: {error}")
    finally:
        unchanged=None if not sources else True
        for path,digest in sources.items():
            try:
                if not Path(path).is_file() or _sha(path)!=digest:unchanged=False
            except OSError:
                unchanged=False
        report["sources_unchanged"]=unchanged
        report["verified_source_count"]=len(sources)
        if unchanged is False:report.update(status="failed",reason="A frozen source changed during comparison")
        report["elapsed_seconds"]=time.monotonic()-started
        _write(destination/"summary.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config",type=Path,default=ROOT/"config.json")
    parser.add_argument("--processed-dir",type=Path)
    parser.add_argument("--checkpoint",action="append",required=True,help="MODEL_ID=PATH; repeat for common development comparison")
    parser.add_argument("--reference-protocol",type=Path)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--execute",action="store_true",help="Frozen development evaluation only")
    parser.add_argument("--_worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    checkpoints={}
    for item in args.checkpoint:
        if "=" not in item:parser.error("checkpoint requires MODEL_ID=PATH")
        label,path=item.split("=",1)
        if label in checkpoints:parser.error("duplicate model identity")
        checkpoints[label]=path
    cfg=validate_config(json.loads(args.config.read_text(encoding="utf-8")) if args.config else None)
    if args.execute and not args._worker:
        command=[sys.executable,"-m","src.evaluate_normal_models",*(sys.argv[1:] if argv is None else argv),"--_worker"]
        try:return subprocess.run(command,check=False,timeout=cfg["max_seconds"]+20).returncode
        except subprocess.TimeoutExpired:
            print("Comparison exceeded its wall cap; outputs are partial, no test or training was started",file=sys.stderr)
            return 124
    data=load_config(args.data_config)
    if args.processed_dir:data["processed_dir"]=args.processed_dir.resolve()
    result=run_development_comparison(DataPipeline(data),checkpoints,args.output_dir,config=cfg,
        reference_protocol=args.reference_protocol,device=args.device,execute=args.execute)
    print(json.dumps(result if not args.execute else {"status":result["status"],"output_dir":str(args.output_dir)},indent=2))
    return 0 if result.get("status","completed")=="completed" else 1


if __name__=="__main__":raise SystemExit(main())
