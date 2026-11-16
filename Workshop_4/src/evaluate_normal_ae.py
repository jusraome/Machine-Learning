"""Frozen normal-only AE evaluation, paired best8 baseline and sealed test stage.

No optimizer is constructed or restored. Model selection belongs exclusively to
the trainer's normal model_selection role. Thresholds use normal calibration;
mixed development/test scores never select a model, score or threshold.
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

from .autoencoder import Autoencoder
from .anomaly_evaluation import (SCORE_NAMES, reconstruction_scores,
    conformal_threshold, detection_metrics, _score_summary, _preserve_random)
from .data import ROOT, DataPipeline, load_config


DEFAULTS = dict(seed=42, alpha=.05, primary_score="brier_mse",
    max_calibration=4096, max_evaluation=8192, max_test=8192,
    per_speaker_cap=512, batch_size=128, read_batch_size=4096,
    max_seconds=1800)


def validate_config(overrides=None):
    cfg={**DEFAULTS, **(overrides or {})}
    if set(cfg)!=set(DEFAULTS):
        raise ValueError("Unknown evaluation setting; model/threshold tuning and training are unsupported")
    for name,lo,hi in (("seed",0,2**31-1),("max_calibration",1,8192),
            ("max_evaluation",1,16384),("max_test",1,16384),("per_speaker_cap",1,4096),
            ("batch_size",1,256),("read_batch_size",1,8192),("max_seconds",1,3600)):
        if type(cfg[name]) is not int or not lo<=cfg[name]<=hi:
            raise ValueError(f"{name} must be an integer in [{lo},{hi}]")
    if cfg["max_calibration"]+cfg["max_evaluation"]>16384:
        raise ValueError("At most 16384 unique development examples may be scored")
    alpha=cfg["alpha"]
    if isinstance(alpha,bool) or not isinstance(alpha,(int,float)) or not math.isfinite(alpha) or not 0<alpha<1:
        raise ValueError("alpha must be finite and in (0,1)")
    if cfg["primary_score"] not in SCORE_NAMES:
        raise ValueError("Predeclare brier_mse or nll_nats_per_sequence before evaluation")
    return cfg


def complete_detection_metrics(scores, labels, threshold=None):
    """Keep grouped-score AP distinct from trapezoidal PR-AUC.

    The PR curve starts at (recall=0, precision=1). With tied scores the linear
    interpolation convention can make trapezoidal PR-AUC differ greatly from AP.
    A constant-score AP equals anomaly prevalence, not necessarily PR-AUC.
    """
    result=detection_metrics(scores,labels,threshold)
    pr=result["precision_recall_curve"]
    if result["positives"]:
        previous_recall,previous_precision=0.0,1.0
        area=0.0
        for point in pr:
            recall,precision=point["recall"],point["precision"]
            area+=(recall-previous_recall)*(precision+previous_precision)/2
            previous_recall,previous_precision=recall,precision
    else:
        area=None
    n=result["examples"]
    prevalence=result["positives"]/n if n else None
    result.update(pr_auc_trapezoidal=area,
        pr_auc_definition="trapezoids over tied-score-group PR points, starting recall=0 precision=1; distinct from average_precision",
        anomaly_prevalence=prevalence,normal_prevalence=result["negatives"]/n if n else None,
        constant_score_average_precision=prevalence,
        random_roc_auc_reference=.5 if result["positives"] and result["negatives"] else None,
        confusion_orientation="positive=anomaly; high reconstruction error; score > frozen threshold",
        score_direction_fixed=True)
    return result


def _deadline(deadline):
    if time.monotonic()>=deadline:
        raise TimeoutError("Frozen evaluation wall-time budget exhausted")


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode("utf-8")


def _write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write("\n")


def _seal(path,value):
    path=Path(path)
    _write(path,value)
    raw=path.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    with path.with_suffix(".sha256").open("x",encoding="ascii") as stream:stream.write(digest+"\n")
    return digest


def read_frozen_protocol(path):
    path=Path(path).resolve()
    if not path.is_file() or not path.with_suffix(".sha256").is_file():
        raise ValueError("Test requires frozen_protocol.json and its SHA-256 sidecar")
    raw=path.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if path.with_suffix(".sha256").read_text(encoding="ascii").strip()!=digest:
        raise ValueError("Frozen protocol SHA-256 mismatch")
    value=json.loads(raw.decode("utf-8"))
    if _sha(path)!=digest:
        raise ValueError("Frozen protocol changed while being read")
    if value.get("schema_version")!=1 or value.get("stage")!="calibrated_before_evaluation":
        raise ValueError("Not a frozen normal-only detection protocol")
    if value.get("primary_score") not in SCORE_NAMES or set(value.get("models",{})) not in ({"normal_only"},{"normal_only","best8"}):
        raise ValueError("Invalid frozen model/score declaration")
    frozen_config=validate_config(value.get("config"))
    if frozen_config["primary_score"]!=value["primary_score"]:
        raise ValueError("Frozen primary score disagrees with frozen configuration")
    for model in value["models"].values():
        for score in SCORE_NAMES:
            threshold=model.get("thresholds",{}).get(score,{}).get("threshold")
            if threshold is None:
                raise ValueError("Test cannot use an insufficient-calibration protocol")
            if isinstance(threshold,bool) or not isinstance(threshold,(float,int)) or not math.isfinite(threshold) or threshold<0:
                raise ValueError("Test requires finite nonnegative frozen reconstruction thresholds")
    return value,digest


def _original_architecture(hparams):
    return (all(hparams.get(k)==v for k,v in dict(embedding_dim=32,hidden_size=128,latent_dim=32,num_layers=1,dropout=0.).items())
            and hparams.get("latent_activation","identity")=="identity")


def _load_model(pipeline,path,device,kind):
    path=Path(path).resolve()
    digest=_sha(path)
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    pipeline.validate_checkpoint(checkpoint)
    if checkpoint.get("model_type")!="ae" or not _original_architecture(checkpoint["model_hparams"]):
        raise ValueError("Evaluation keeps the original deterministic AE z32 architecture")
    if kind=="normal_only":
        if checkpoint.get("checkpoint_role")!="normal_only_ae_best_inference":
            raise ValueError("Select the trainer's frozen normal-only best_inference.pt")
        training=checkpoint.get("training_config",{})
        proof=checkpoint.get("normal_only_protocol",{})
        definition=checkpoint.get("normal_definition",{})
        if not training.get("from_scratch") or not proof.get("from_scratch") or checkpoint.get("source_checkpoint") is not None:
            raise ValueError("The detection AE must be trained from scratch, never continued from best8")
        if training.get("smoke_test") or proof.get("experiment_scope")=="smoke_integration":
            raise ValueError("A smoke checkpoint is not a trained anomaly-detector candidate")
        if type(checkpoint.get("global_updates")) is not int or checkpoint["global_updates"]<1:
            raise ValueError("An untrained initial checkpoint cannot be the final detection model")
        normal=definition.get("normal_class")
        if type(normal) is not int or not 0<=normal<len(pipeline.vocab["itos"]):
            raise ValueError("Missing valid train-derived normal definition")
        if training.get("normal_class")!=normal or proof.get("normal_class")!=normal or definition.get("fitted_split")!="train":
            raise ValueError("Normal-only provenance disagrees about the training class")
    elif checkpoint.get("epoch")!=8:
        raise ValueError("The exploratory baseline must be the historical epoch8 AE")
    model=Autoencoder(**checkpoint["model_hparams"]).to(device)
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    if any(not torch.isfinite(p).all().item() for p in model.parameters()):
        raise FloatingPointError("Nonfinite frozen model parameter")
    model.eval()
    return model,checkpoint,dict(path=str(path),sha256=digest,epoch=checkpoint.get("epoch"),
        checkpoint_role=checkpoint.get("checkpoint_role"),global_updates=checkpoint.get("global_updates"))


def _entries_record(entries):
    return [dict(row_index=entry["row_index"],row_sha256=entry["row_sha256"],
                 input_sha256=entry.get("input_sha256"),
                 character_id=entry["row"]["character_id"],label=entry["label"]) for entry in entries]


def _quarantine_best8_training_overlap(pipeline,samples,normal,cfg,deadline):
    """Drop selected inputs seen anywhere in best8's all-class training split.

    Only the bounded set of already retained input hashes is held in memory;
    the complete train stream is scanned without retaining its hash population.
    Filtering affects both models identically and never uses model scores.
    """
    candidates={entry.get("input_sha256") or hashlib.sha256(entry["row"]["x"].encode("utf-8")).hexdigest()
                for entries in samples.values() for entry in entries}
    if len(candidates)>16384:
        raise ValueError("Paired quarantine candidate population exceeds evaluation bound")
    matched=set()
    scanned=matching_rows=normal_matches=other_matches=0
    reader=pipeline._iter_rows_batches("train",batch_size=cfg["read_batch_size"])
    try:
        for rows in reader:
            _deadline(deadline)
            _,labels=pipeline._validate_rows(rows,"train")
            for row,label in zip(rows,labels):
                scanned+=1
                input_hash=hashlib.sha256(row["x"].encode("utf-8")).hexdigest()
                if input_hash in candidates:
                    matched.add(input_hash)
                    matching_rows+=1
                    normal_matches+=int(label==normal)
                    other_matches+=int(label!=normal)
    finally:
        if hasattr(reader,"close"):reader.close()
    filtered={}
    role_metadata={}
    for role,entries in samples.items():
        removed=[entry for entry in entries if (entry.get("input_sha256") or
                 hashlib.sha256(entry["row"]["x"].encode("utf-8")).hexdigest()) in matched]
        retained=[entry for entry in entries if (entry.get("input_sha256") or
                  hashlib.sha256(entry["row"]["x"].encode("utf-8")).hexdigest()) not in matched]
        filtered[role]=retained
        role_metadata[role]=dict(selected_before=len(entries),selected_after=len(retained),excluded_rows=len(removed),
            excluded_unique_inputs=len({entry.get("input_sha256") or hashlib.sha256(entry["row"]["x"].encode("utf-8")).hexdigest() for entry in removed}),
            normal_rows_after=sum(entry["label"]==normal for entry in retained),
            anomaly_rows_after=sum(entry["label"]!=normal for entry in retained),
            population_fingerprint=hashlib.sha256(_canonical(_entries_record(retained))).hexdigest())
    metadata=dict(applied=True,scope="all TRAIN inputs, regardless of class; relevant because best8 trained on all classes",
        action="exclude matched retained inputs from BOTH model views before any forward; no refill or score-dependent selection",
        scanned_train_rows=scanned,candidate_unique_inputs=len(candidates),matched_unique_inputs=len(matched),
        matching_train_rows=matching_rows,matching_normal_train_rows=normal_matches,
        matching_other_class_train_rows=other_matches,
        matched_inputs_sha256=hashlib.sha256(_canonical(sorted(matched))).hexdigest(),roles=role_metadata,
        population_fingerprint=hashlib.sha256(_canonical({role:value["population_fingerprint"] for role,value in role_metadata.items()})).hexdigest(),
        limitations=["Exact x equality only; does not identify paraphrases or near duplicates",
                     "Historical validation-based selection of best8 remains a separate methodological difference"])
    return filtered,metadata


def _digest_model(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _score_population(pipeline,model,entries,split,normal,cfg,deadline):
    from .normal_only_protocol import role_tensor
    rows=[]
    for start in range(0,len(entries),cfg["batch_size"]):
        _deadline(deadline)
        part=entries[start:start+cfg["batch_size"]]
        encoded=role_tensor(pipeline,part,split=split)
        x=encoded["x"] if isinstance(encoded,dict) else encoded[0]
        result=reconstruction_scores(model,x.to(next(model.parameters()).device))
        rows.extend(dict(row_index=entry["row_index"],row_sha256=entry["row_sha256"],
            character_id=entry["row"]["character_id"],y_id=entry["label"],
            is_anomaly=int(entry["label"]!=normal),
            **{score:result[score][i].item() for score in SCORE_NAMES}) for i,entry in enumerate(part))
    return rows


def _metrics(rows,thresholds):
    return {score:{**complete_detection_metrics([row[score] for row in rows],
                    [row["is_anomaly"] for row in rows],thresholds[score]["threshold"]),
            "normal_score_distribution":_score_summary([row[score] for row in rows if not row["is_anomaly"]]),
            "anomaly_score_distribution":_score_summary([row[score] for row in rows if row["is_anomaly"]])}
            for score in SCORE_NAMES}


def _write_scores(path,populations):
    with Path(path).open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=["model","population","row_index","row_sha256","character_id","y_id","is_anomaly",*SCORE_NAMES])
        writer.writeheader()
        for (model,population),rows in populations.items():
            writer.writerows(dict(model=model,population=population,**row) for row in rows)


def run_normal_evaluation(pipeline,checkpoint_path,output_dir,*,baseline_checkpoint=None,
                         config=None,stage="evaluate",frozen_protocol=None,device="cpu",execute=False):
    cfg=validate_config(config)
    if stage not in ("evaluate","test") or type(execute) is not bool:
        raise ValueError("Choose evaluate or sealed test explicitly")
    if stage=="test" and frozen_protocol is None:
        raise ValueError("Test requires an existing frozen_protocol.json; calibration is never refitted")
    destination=Path(output_dir).resolve()
    if destination.exists():raise FileExistsError("Choose a new evaluation output directory")
    for source in (checkpoint_path,baseline_checkpoint,frozen_protocol):
        if source is not None and (destination==Path(source).resolve() or destination in Path(source).resolve().parents):
            raise ValueError("Output cannot contain an immutable source")
    plan=dict(operation="normal_only_frozen_anomaly_evaluation",stage=stage,execute=execute,
        checkpoint=str(checkpoint_path),best8_baseline=str(baseline_checkpoint) if baseline_checkpoint else None,
        frozen_protocol=str(frozen_protocol) if frozen_protocol else None,config=cfg,device=device,
        primary_score=cfg["primary_score"],optimizer_steps=0,
        model_selection="Already fixed by trainer using normal model_selection validation speakers only",
        calibration="Normal calibration role only; thresholds sealed before mixed evaluation forward",
        test="Separate explicit stage requiring sealed source hashes, score and thresholds; no refit",
        limitations=["y next-character classes are proxy anomaly labels, not semantic novelty annotations",
            "AP must be interpreted against anomaly prevalence; trapezoidal PR-AUC is separately defined",
            "Paired best8 comparison uses identical new rows; earlier reported2021 metrics are unpaired unless row fingerprints match",
            "Reservoir samples are speaker-stratified subsets, not full validation or test",
            "Paired best8 comparison additionally removes retained inputs found anywhere in all-class TRAIN from BOTH views"])
    if stage=="test" and config is None:
        plan.update(config=None,primary_score="from_sealed_protocol",
                    configuration_source="frozen_protocol.json, validated on execution")
    if not execute:return plan
    selected=torch.device(device)
    if selected.type not in ("cpu","cuda") or selected.type=="cuda" and not torch.cuda.is_available():
        raise ValueError("Select an available CPU/CUDA device")
    destination.mkdir(parents=True,exist_ok=False)
    _write(destination/"plan.json",plan)
    deadline=time.monotonic()+cfg["max_seconds"]
    report=dict(status="running",stage=stage,optimizer_steps=0,checkpoint_sources_unchanged=None,plan=plan)
    sources={}
    started=time.monotonic()
    try:
        from .normal_only_protocol import role_partition,prepare_role_samples,build_train_input_guard
        with _preserve_random():
            frozen=None
            if stage=="test":
                frozen,frozen_digest=read_frozen_protocol(frozen_protocol)
                sources[str(Path(frozen_protocol).resolve())]=frozen_digest
                sidecar=Path(frozen_protocol).resolve().with_suffix(".sha256")
                sources[str(sidecar)]=_sha(sidecar)
                if config is None:cfg=validate_config(frozen["config"])
                else:
                    stable={k:v for k,v in cfg.items() if k!="max_seconds"}
                    expected={k:v for k,v in frozen["config"].items() if k!="max_seconds"}
                    if stable!=expected:raise ValueError("Test configuration differs from the sealed protocol")
                deadline=started+cfg["max_seconds"]
                report["effective_config"]=cfg
                if frozen["data_contract"]!=pipeline.contract():raise ValueError("Sealed data contract differs")
            model,checkpoint,info=_load_model(pipeline,checkpoint_path,selected,"normal_only")
            sources[info["path"]]=info["sha256"]
            models={"normal_only":model}
            model_info={"normal_only":info}
            normal=checkpoint["normal_definition"]["normal_class"]
            partition=role_partition(pipeline,cfg["seed"])
            declared=checkpoint.get("validation_partition",{})
            if declared.get("role_digest")!=partition["role_digest"] or declared.get("roles")!=partition["roles"]:
                raise ValueError("Evaluation roles differ from the trainer's model-selection/calibration/evaluation partition")
            if checkpoint["normal_only_protocol"].get("role_digest")!=partition["role_digest"]:
                raise ValueError("Normal-only checkpoint protocol digest differs")
            if frozen is not None:
                if frozen["role_digest"]!=partition["role_digest"] or frozen["normal_class"]!=normal:
                    raise ValueError("Sealed class or partition differs")
                if info["sha256"]!=frozen["models"]["normal_only"]["sha256"]:
                    raise ValueError("The selected model changed after freezing")
                if "best8" in frozen["models"] and baseline_checkpoint is None:
                    baseline_checkpoint=frozen["models"]["best8"]["path"]
                if (baseline_checkpoint is not None)!=("best8" in frozen["models"]):
                    raise ValueError("Test baseline selection differs from the sealed protocol")
            if baseline_checkpoint is not None:
                baseline,_,baseline_info=_load_model(pipeline,baseline_checkpoint,selected,"best8")
                sources[baseline_info["path"]]=baseline_info["sha256"]
                if frozen is not None and baseline_info["sha256"]!=frozen["models"]["best8"]["sha256"]:
                    raise ValueError("The best8 baseline changed after freezing")
                models["best8"],model_info["best8"]=baseline,baseline_info
            guard=build_train_input_guard(pipeline,normal,deadline=deadline,read_batch_size=cfg["read_batch_size"])
            if checkpoint.get("train_input_guard")!=guard["metadata"]:
                raise ValueError("Normal train input hashes differ from the trained checkpoint's leakage guard")
            if frozen is not None and frozen.get("train_input_guard")!=guard["metadata"]:
                raise ValueError("Normal train input hashes changed after calibration was frozen")
            _write(destination/"train_input_guard.json",guard["metadata"])
            before={name:_digest_model(value) for name,value in models.items()}
            populations={}
            if stage=="evaluate":
                prepared=prepare_role_samples(pipeline,partition,normal,roles=("calibration","evaluation"),
                    max_per_role={"calibration":cfg["max_calibration"],"evaluation":cfg["max_evaluation"]},
                    per_speaker_cap=cfg["per_speaker_cap"],read_batch_size=cfg["read_batch_size"],deadline=deadline,
                    exclude_input_hashes=guard["input_hashes"])
                samples=prepared["samples"]
                paired_guard=dict(applied=False,reason="No best8 baseline; normal-model train quarantine already applied")
                if "best8" in models:
                    samples,paired_guard=_quarantine_best8_training_overlap(pipeline,samples,normal,cfg,deadline)
                prepared["sampling"]["paired_baseline_quarantine"]=paired_guard
                _write(destination/"sampling.json",prepared["sampling"])
                _write(destination/"paired_input_quarantine.json",paired_guard)
                thresholds={}
                for name,value in models.items():
                    calibration=_score_population(pipeline,value,samples["calibration"],"val",normal,cfg,deadline)
                    if any(row["is_anomaly"] for row in calibration):raise RuntimeError("Anomalies entered threshold calibration")
                    populations[name,"calibration"]=calibration
                    thresholds[name]={score:conformal_threshold([row[score] for row in calibration],cfg["alpha"]) for score in SCORE_NAMES}
                frozen=dict(schema_version=1,stage="calibrated_before_evaluation",config=cfg,
                    primary_score=cfg["primary_score"],score_direction="higher is anomaly; strict score > threshold",
                    normal_class=normal,normal_definition=checkpoint["normal_definition"],
                    data_contract=pipeline.contract(),role_digest=partition["role_digest"],partition=partition,
                    train_input_guard=guard["metadata"],
                    paired_input_quarantine=paired_guard,
                    models={name:{**model_info[name],"thresholds":thresholds[name]} for name in models},
                    selected_rows={role:_entries_record(samples[role]) for role in ("calibration","evaluation")},
                    selected_rows_sha256={role:hashlib.sha256(_canonical(_entries_record(samples[role]))).hexdigest() for role in samples},
                    model_selection_scope="normal model_selection role; no calibration/evaluation metrics used to select checkpoint",
                    freeze_point="All calibration scores computed; no mixed evaluation or test forward has occurred",
                    test_ready=all(t["threshold"] is not None for by_score in thresholds.values() for t in by_score.values()))
                frozen_digest=_seal(destination/"frozen_protocol.json",frozen)
                for name,value in models.items():
                    populations[name,"evaluation"]=_score_population(pipeline,value,samples["evaluation"],"val",normal,cfg,deadline)
                population="evaluation"
            else:
                prepared=prepare_role_samples(pipeline,partition,normal,roles=("test",),
                    max_per_role={"test":cfg["max_test"]},per_speaker_cap=cfg["per_speaker_cap"],
                    read_batch_size=cfg["read_batch_size"],deadline=deadline,allow_test=True,
                    exclude_input_hashes=guard["input_hashes"])
                test_samples=prepared["samples"]
                paired_guard=dict(applied=False,reason="No best8 baseline; normal-model train quarantine already applied")
                if "best8" in models:
                    test_samples,paired_guard=_quarantine_best8_training_overlap(pipeline,test_samples,normal,cfg,deadline)
                prepared["sampling"]["paired_baseline_quarantine"]=paired_guard
                _write(destination/"sampling.json",prepared["sampling"])
                _write(destination/"paired_input_quarantine.json",paired_guard)
                for name,value in models.items():
                    populations[name,"test"]=_score_population(pipeline,value,test_samples["test"],"test",normal,cfg,deadline)
                population="test"
            if any(before[name]!=_digest_model(value) or any(p.grad is not None for p in value.parameters()) for name,value in models.items()):
                raise RuntimeError("Frozen inference changed parameters or gradients")
            metrics={name:_metrics(populations[name,population],frozen["models"][name]["thresholds"]) for name in models}
            comparison=[]
            if "best8" in metrics:
                for score in SCORE_NAMES:
                    for key in ("roc_auc","pr_auc_trapezoidal","average_precision","precision","recall","f1","false_positive_rate"):
                        current,baseline=metrics["normal_only"][score][key],metrics["best8"][score][key]
                        comparison.append(dict(score=score,metric=key,normal_only=current,best8=baseline,
                            normal_only_minus_best8=current-baseline if current is not None and baseline is not None else None))
            _write_scores(destination/"scores.csv",populations)
            _write(destination/"metrics.json",dict(population=population,primary_score=cfg["primary_score"],models=metrics))
            _write(destination/"comparison.json",dict(population=population,paired_rows=True,
                method="Same calibration/evaluation rows for both models; independently calibrated per-model/per-score thresholds",rows=comparison))
            report.update(status="completed",population=population,metrics=metrics,paired_best8="best8" in models,
                frozen_protocol_sha256=frozen_digest,frozen_protocol_path=str(Path(frozen_protocol).resolve()) if stage=="test" else str(destination/"frozen_protocol.json"),
                primary_score=cfg["primary_score"],test_ready=frozen["test_ready"],
                paired_input_quarantine=paired_guard,
                calibration_status="finite_thresholds" if frozen["test_ready"] else "insufficient_normal_calibration",
                model_unchanged=True,normal_class=normal,
                scored_unique_examples=len(next(rows for (name,pop),rows in populations.items() if pop==population)))
    except Exception as error:
        report.update(status="failed",reason=f"{type(error).__name__}: {error}")
    finally:
        unchanged=all(Path(path).is_file() and _sha(path)==expected for path,expected in sources.items())
        report["checkpoint_sources_unchanged"]=unchanged
        if not unchanged:report.update(status="failed",reason="A frozen checkpoint changed during evaluation")
        report["elapsed_seconds"]=time.monotonic()-started
        _write(destination/"summary.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config",type=Path,default=ROOT/"config.json")
    parser.add_argument("--processed-dir",type=Path)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--baseline-checkpoint",type=Path)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--stage",choices=("evaluate","test"),default="evaluate")
    parser.add_argument("--frozen-protocol",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--_worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    raw=json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    cfg=validate_config(raw)
    if args.execute and not args._worker:
        if args.stage=="test" and raw is None:
            sealed,_=read_frozen_protocol(args.frozen_protocol)
            cfg=validate_config(sealed["config"])
        command=[sys.executable,"-m","src.evaluate_normal_ae",*(sys.argv[1:] if argv is None else argv),"--_worker"]
        try:return subprocess.run(command,check=False,timeout=cfg["max_seconds"]+20).returncode
        except subprocess.TimeoutExpired:
            print("Evaluation exceeded hard wall cap; any existing files are partial",file=sys.stderr)
            return 124
    data=load_config(args.data_config)
    if args.processed_dir:data["processed_dir"]=args.processed_dir.resolve()
    result=run_normal_evaluation(DataPipeline(data),args.checkpoint,args.output_dir,
        baseline_checkpoint=args.baseline_checkpoint,config=raw,stage=args.stage,
        frozen_protocol=args.frozen_protocol,device=args.device,execute=args.execute)
    print(json.dumps(result if not args.execute else {"status":result["status"],"output_dir":str(args.output_dir)},indent=2))
    return 0 if result.get("status","completed")=="completed" else 1


if __name__=="__main__":raise SystemExit(main())
