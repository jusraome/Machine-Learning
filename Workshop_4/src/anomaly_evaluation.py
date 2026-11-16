"""Frozen best8 anomaly evaluation: no training or optimizer restoration.

The historical all-class representation model is an exploratory detector, not
the normal-only AE required by the final workshop protocol. Normal labels are
chosen from train only; normal val calibrates thresholds before held-out val or
explicit opt-in test evaluation. Primary score is probability/one-hot MSE.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from .autoencoder import Autoencoder
from .data import ROOT, DataPipeline, load_config


SCORE_NAMES = ("brier_mse", "nll_nats_per_sequence")
DEFAULTS = dict(seed=42, normal_class=None, alpha=.05, calibration_fraction=.5,
                max_calibration=2048, max_evaluation=2048, max_test=2048,
                per_speaker_cap=256, batch_size=128, read_batch_size=4096,
                max_seconds=600, expected_epoch=8, primary_score="brier_mse",
                evaluate_test=False)


def validate_config(overrides=None):
    cfg = {**DEFAULTS, **(overrides or {})}
    if set(cfg) != set(DEFAULTS):
        raise ValueError("Unknown anomaly setting; training and architecture options are unsupported")
    for name, low, high in (("seed", 0, 2**31-1), ("max_calibration", 1, 8192),
            ("max_evaluation", 1, 8192), ("max_test", 1, 8192), ("per_speaker_cap", 1, 2048),
            ("batch_size", 1, 256), ("read_batch_size", 1, 8192), ("max_seconds", 1, 600)):
        if type(cfg[name]) is not int or not low <= cfg[name] <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if cfg["normal_class"] is not None and (type(cfg["normal_class"]) is not int or cfg["normal_class"] < 0):
        raise ValueError("normal_class must be a nonnegative token ID or null")
    if type(cfg["expected_epoch"]) is not int or cfg["expected_epoch"] != 8:
        raise ValueError("This evaluation is scoped to historical epoch 8")
    for name in ("alpha", "calibration_fraction"):
        if isinstance(cfg[name], bool) or not isinstance(cfg[name], (int, float)) or not math.isfinite(cfg[name]) or not 0 < cfg[name] < 1:
            raise ValueError(f"{name} must lie strictly between zero and one")
    if type(cfg["evaluate_test"]) is not bool or cfg["primary_score"] not in SCORE_NAMES:
        raise ValueError("Invalid score selection or test opt-in")
    count = cfg["max_calibration"] + cfg["max_evaluation"] + (cfg["max_test"] if cfg["evaluate_test"] else 0)
    if count > 16384:
        raise ValueError("At most 16384 examples may be scored across calibration/evaluation/test")
    return cfg


def _deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Anomaly evaluation wall-time budget exhausted")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _finite_values(values, *, allow_empty=False):
    tensor = torch.as_tensor(values, dtype=torch.float64).detach().cpu()
    if tensor.ndim != 1 or (not allow_empty and not tensor.numel()):
        raise ValueError("scores must be a nonempty one-dimensional collection")
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError("Nonfinite anomaly score")
    return tensor


@contextmanager
def _preserve_random():
    state = random.getstate()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(state)


def fit_normal_class(pipeline, normal_class=None, *, batch_size=4096, deadline=None):
    """Choose the majority next-character y using all train, never val/test.

    y is a surrogate class label, not a semantic anomaly annotation. Ties are
    resolved by the smallest vocabulary ID, independently of held-out scores.
    """
    size = len(pipeline.vocab["itos"])
    if normal_class is not None and (type(normal_class) is not int or not 0 <= normal_class < size):
        raise ValueError("normal_class is outside the existing vocabulary")
    counts = Counter()
    digest = hashlib.sha256()
    count = 0
    raw = hasattr(pipeline, "_iter_rows_batches") and hasattr(pipeline, "_validate_rows")
    reader = (pipeline._iter_rows_batches("train", batch_size=batch_size) if raw else
              pipeline.iter_tensor_batches("train", batch_size=batch_size))
    try:
        for batch in reader:
            _deadline(deadline)
            labels = pipeline._validate_rows(batch, "train")[1] if raw else torch.as_tensor(batch["y"], dtype=torch.long).tolist()
            if any(type(value) is not int or not 0 <= value < size for value in labels):
                raise ValueError("Invalid train class label")
            counts.update(labels)
            count += len(labels)
            digest.update(torch.tensor(labels, dtype=torch.int64).numpy().astype("<i8", copy=False).tobytes())
    finally:
        if hasattr(reader, "close"):
            reader.close()
    if not count:
        raise ValueError("Cannot choose normality from an empty train split")
    selected = min(counts, key=lambda token: (-counts[token], token)) if normal_class is None else normal_class
    if not counts[selected]:
        raise ValueError("Declared normal class has no training examples")
    return dict(normal_class=selected, normal_token=pipeline.vocab["itos"][selected],
                source="most_frequent_train_y" if normal_class is None else "explicit_train_supported_class",
                train_examples=count, train_label_counts=[counts[i] for i in range(size)],
                train_y_sha256=digest.hexdigest(), fitted_split="train",
                label_definition="normal iff next-character y_id equals normal_class; other y labels are proxy anomalies")


def split_validation_speakers(pipeline, seed=42, calibration_fraction=.5):
    """Predeclare speaker-disjoint development groups without using labels/scores."""
    speakers = sorted(pipeline.groups["val"])
    if len(speakers) < 2:
        raise ValueError("At least two validation speakers are needed for disjoint calibration/evaluation")
    ordered = sorted(speakers, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
    cut = min(len(ordered)-1, max(1, math.floor(len(ordered)*calibration_fraction)))
    return dict(calibration_speakers=ordered[:cut], evaluation_speakers=ordered[cut:],
                split="val", method="seeded speaker hash, declared before scoring",
                leakage_control="no speaker is used for both threshold calibration and held-out development evaluation",
                limitation="Speaker separation does not prove exchangeability or independence of overlapping windows")


@torch.no_grad()
def reconstruction_scores(model, x):
    """Return FP64 per-sequence NLL and MSE of probabilities against one-hot x.

    brier_mse = mean_{t,v}(p[v|z,t]-one_hot(x[t])[v])**2. It is not an MSE
    between ordinal character IDs. Neither y nor test labels enter decoding.
    """
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with _preserve_random():
            output = model(x)
            logits = output.logits.detach().double()
            if logits.ndim != 3 or x.shape != logits.shape[:2] or x.dtype != torch.long:
                raise ValueError("Expected logits (B,T,V) and long reconstruction targets x (B,T)")
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError("Nonfinite reconstruction logits")
            logp = F.log_softmax(logits, dim=-1)
            nll = -logp.gather(-1, x.unsqueeze(-1)).squeeze(-1).sum(-1)
            p = logp.exp()
            mse = (p.square().sum(-1)-2*p.gather(-1, x.unsqueeze(-1)).squeeze(-1)+1).mean(-1)/logits.shape[-1]
            mse = mse.clamp_min(0)
            if not torch.isfinite(nll).all().item() or not torch.isfinite(mse).all().item():
                raise FloatingPointError("Nonfinite reconstruction score")
            return dict(nll_nats_per_sequence=nll, brier_mse=mse, brier_probability_mse=mse)
    finally:
        for module, mode in modes.items():
            module.training = mode


def conformal_threshold(normal_scores, alpha=.05):
    """Upper order statistic on normal calibration scores; strict exceedance.

    This order-statistic recipe does not give a nominal FPR guarantee here:
    speaker shift and overlapping windows invalidate an exchangeability claim.
    """
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    values = _finite_values(normal_scores, allow_empty=True)
    n = len(values)
    rank = math.ceil((n+1)*(1-alpha))
    threshold = values.sort().values[rank-1].item() if n and rank <= n else None
    return dict(threshold=threshold, rank=rank, n_normal=n, alpha=alpha,
                status="finite" if threshold is not None else "insufficient_normal_calibration",
                rule="anomaly iff score > threshold; equality remains normal",
                fitted_population="normal validation calibration only",
                fpr_guarantee="not claimed: grouped speakers and overlapping windows are not established exchangeable")


def detection_metrics(scores, labels, threshold=None):
    """Tie-aware rank AUC and grouped-threshold AP; higher score=more anomalous."""
    values = _finite_values(scores, allow_empty=True)
    y = torch.as_tensor(labels).detach().cpu()
    if y.ndim != 1 or len(y) != len(values) or ((y != 0) & (y != 1)).any().item():
        raise ValueError("Labels must be binary anomaly indicators matching the scores")
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    grouped = []
    pairs = sorted(zip(values.tolist(), y.tolist()), key=lambda pair: pair[0])
    for score, label in pairs:
        if not grouped or score != grouped[-1][0]:
            grouped.append([score, 0, 0])
        grouped[-1][1 if label else 2] += 1
    negatives_below, favorable = 0, 0.0
    for _, p, n in grouped:
        favorable += p*(negatives_below+.5*n)
        negatives_below += n
    auc = favorable/(positives*negatives) if positives and negatives else None
    tp = fp = 0
    ap = 0.0
    roc = [{"threshold":None,"fpr":0.0,"tpr":0.0}] if positives and negatives else []
    pr = []
    for score, p, n in reversed(grouped):
        tp += p
        fp += n
        precision = tp/(tp+fp)
        recall = tp/positives if positives else None
        if positives:
            ap += (p/positives)*precision
            pr.append(dict(threshold=score, precision=precision, recall=recall))
        if positives and negatives:
            roc.append(dict(threshold=score,fpr=fp/negatives,tpr=tp/positives))
    result = dict(examples=len(values), positives=positives, negatives=negatives,
                  roc_auc=auc, average_precision=ap if positives else None,
                  ranking_status="defined" if positives and negatives else "undefined_single_or_empty_class",
                  score_direction="higher is more anomalous", positive_label=1,
                  threshold=threshold, confusion=None, precision=None, recall=None, f1=None,
                  false_positive_rate=None, accuracy=None,
                  roc_curve=roc, precision_recall_curve=pr,
                  average_precision_definition="sum of recall increments times precision after each tied-score group")
    if threshold is not None:
        if isinstance(threshold, bool) or not math.isfinite(threshold):
            raise ValueError("threshold must be finite or None")
        predicted = values > threshold
        tp = int((predicted & (y == 1)).sum())
        fp = int((predicted & (y == 0)).sum())
        fn, tn = positives-tp, negatives-fp
        precision = tp/(tp+fp) if tp+fp else None
        recall = tp/positives if positives else None
        result.update(confusion=dict(tp=tp,fp=fp,tn=tn,fn=fn),
            precision=precision,recall=recall,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            false_positive_rate=fp/negatives if negatives else None,
            accuracy=(tp+tn)/len(values) if len(values) else None)
    return result


def _score_summary(values):
    scores = _finite_values(values, allow_empty=True)
    if not len(scores):
        return dict(examples=0,mean=None,std=None,minimum=None,median=None,p95=None,maximum=None)
    return dict(examples=len(scores),mean=scores.mean().item(),std=scores.std(unbiased=False).item(),
                minimum=scores.min().item(),median=scores.quantile(.5).item(),
                p95=scores.quantile(.95).item(),maximum=scores.max().item())


def _quotas(speakers, count, per_speaker_cap):
    quotient, remainder = divmod(count, len(speakers))
    return {speaker:min(per_speaker_cap, quotient+(i < remainder)) for i,speaker in enumerate(speakers)}


def _sample_rows(pipeline, split, speaker_groups, normal_class, cfg, deadline):
    """Speaker-stratified reservoirs scan split; only retained rows are scored."""
    members = {}
    reservoirs, seen, quotas, generators = {}, {}, {}, {}
    for role, speakers, count, only_normal in speaker_groups:
        if not speakers:
            raise ValueError(f"No speakers for {role}")
        role_quotas = _quotas(speakers, count, cfg["per_speaker_cap"])
        for speaker in speakers:
            key = (role,speaker)
            members[speaker] = (role,only_normal)
            quotas[key] = role_quotas[speaker]
            reservoirs[key], seen[key] = [], 0
            seed = int.from_bytes(hashlib.sha256(f"{cfg['seed']}:{split}:{role}:{speaker}".encode()).digest()[:8], "little")
            generators[key] = random.Random(seed)
    reader = pipeline._iter_rows_batches(split, batch_size=cfg["read_batch_size"])
    scanned, eligible = 0, Counter()
    try:
        for rows in reader:
            _deadline(deadline)
            _, targets = pipeline._validate_rows(rows,split)
            for row,label in zip(rows,targets):
                index = scanned
                scanned += 1
                role, only_normal = members[row["character_id"]]
                if only_normal and label != normal_class:
                    continue
                eligible[role] += 1
                key = (role,row["character_id"])
                seen[key] += 1
                capacity = quotas[key]
                if not capacity:
                    continue
                entry = dict(row=row,row_index=index,label=label)
                if len(reservoirs[key]) < capacity:
                    reservoirs[key].append(entry)
                else:
                    replacement = generators[key].randrange(seen[key])
                    if replacement < capacity:
                        reservoirs[key][replacement] = entry
    finally:
        reader.close()
    samples, coverage = {}, {}
    for role,speakers,_,_ in speaker_groups:
        selected = [entry for speaker in speakers for entry in reservoirs[role,speaker]]
        samples[role] = sorted(selected, key=lambda entry:entry["row_index"])
        coverage[role] = dict(eligible_rows=eligible[role],selected_rows=len(selected),
            speakers_declared=len(speakers),speakers_with_samples=sum(bool(reservoirs[role,s]) for s in speakers),
            per_speaker={s:dict(quota=quotas[role,s],eligible=seen[role,s],selected=len(reservoirs[role,s])) for s in speakers})
    return samples,dict(split=split,scanned_rows=scanned,roles=coverage,
            method="uniform reservoir within prespecified speaker quotas; full split scan, bounded scoring",
            weighting="speaker-stratified subset, not an unbiased estimate of full-split class prevalence")


def _model_digest(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_anomaly_evaluation(pipeline, checkpoint_path, output_dir, *, config=None, device="cpu", execute=False):
    cfg=validate_config(config)
    if type(execute) is not bool:
        raise ValueError("execute must be bool")
    destination,source=Path(output_dir).resolve(),Path(checkpoint_path).resolve()
    if destination.exists():
        raise FileExistsError("Use a new anomaly evaluation output directory")
    if destination==source or destination in source.parents:
        raise ValueError("Output cannot contain the historical checkpoint")
    partition=split_validation_speakers(pipeline,cfg["seed"],cfg["calibration_fraction"])
    plan=dict(operation="frozen_best8_anomaly_evaluation",checkpoint=str(source),config=cfg,
        execute=execute,device=device,partition=partition,optimizer_steps=0,
        primary_score=cfg["primary_score"],secondary_scores=[s for s in SCORE_NAMES if s!=cfg["primary_score"]],
        score_definitions=dict(brier_mse="mean over positions and vocabulary of (softmax(logits)-one_hot(x))^2; not ordinal token-ID MSE",
                               nll_nats_per_sequence="categorical NLL of x, summed over all sequence positions including UNK"),
        threshold_fit="normal calibration subset of validation only; frozen before held-out val and optional test",
        test_access="explicit opt-in" if cfg["evaluate_test"] else "none",
        limitations=["general all-class best8 is exploratory, not the workshop normal-only-trained detector",
            "next-character y classes are proxy labels, not semantic anomaly annotations",
            "speaker-disjoint sampling does not establish nominal conformal false-positive guarantee",
            "bounded speaker-stratified subset differs from complete validation/test and from prior prefix128"])
    if not execute:
        return plan
    selected=torch.device(device)
    if selected.type not in ("cpu","cuda") or (selected.type=="cuda" and not torch.cuda.is_available()):
        raise ValueError("Select an available CPU/CUDA device")
    destination.mkdir(parents=True,exist_ok=False)
    _write(destination/"plan.json",plan)
    result=dict(status="running",plan=plan,optimizer_steps=0,checkpoint_modified=False)
    started=time.monotonic()
    deadline=started+cfg["max_seconds"]
    original_hash=None
    try:
        original_hash=_sha256(source)
        checkpoint=torch.load(source,map_location="cpu",weights_only=True)
        pipeline.validate_checkpoint(checkpoint)
        hp=checkpoint["model_hparams"]
        if (checkpoint.get("model_type")!="ae" or hp.get("latent_dim")!=32 or
                hp.get("latent_activation","identity")!="identity" or checkpoint.get("epoch")!=cfg["expected_epoch"]):
            raise ValueError("Expected the original deterministic AE z32 historical epoch8 checkpoint")
        normal=fit_normal_class(pipeline,cfg["normal_class"],batch_size=cfg["read_batch_size"],deadline=deadline)
        training_config=checkpoint.get("training_config")
        training_regime_known=isinstance(training_config,dict) and "normal_class" in training_config
        training_normal=training_config.get("normal_class") if training_regime_known else None
        if training_normal is not None and training_normal!=normal["normal_class"]:
            raise ValueError("Checkpoint normal-only class differs from the declared detection normal class")
        trained_normal_only=training_regime_known and training_normal==normal["normal_class"]
        result.update(checkpoint_sha256=original_hash,checkpoint_epoch=checkpoint.get("epoch"),
            checkpoint_role=checkpoint.get("checkpoint_role"),normal_definition=normal,
            checkpoint_training_normal_class=training_normal,
            detector_status=("normal_only_checkpoint_declaration" if trained_normal_only else
                "exploratory_not_normal_only_trained" if training_regime_known else "exploratory_training_regime_unverified"),
            normal_only_training_evidence="checkpoint declaration only; historical provenance must be retained" if trained_normal_only else
                "checkpoint declares general representation training; this does not fulfill normal-only training requirement"
                if training_regime_known else "checkpoint lacks a normal_class declaration; normal-only provenance is unverified")
        _write(destination/"normal_definition.json",normal)
        val_groups=[("calibration",partition["calibration_speakers"],cfg["max_calibration"],True),
                    ("evaluation",partition["evaluation_speakers"],cfg["max_evaluation"],False)]
        samples,sampling=_sample_rows(pipeline,"val",val_groups,normal["normal_class"],cfg,deadline)
        _write(destination/"sampling.json",sampling)
        with _preserve_random():
            model=Autoencoder(**hp).to(selected)
            model.load_state_dict(checkpoint["state_dict"],strict=True)
            if any(not torch.isfinite(p).all().item() for p in model.parameters()):
                raise FloatingPointError("Nonfinite checkpoint model parameters")
            before=_model_digest(model)
            rows=[]
            def score_population(role,entries,split):
                scored=[]
                for start in range(0,len(entries),cfg["batch_size"]):
                    _deadline(deadline)
                    part=entries[start:start+cfg["batch_size"]]
                    encoded=pipeline.encode_tensor_rows([entry["row"] for entry in part],split)
                    scores=reconstruction_scores(model,encoded["x"].to(selected))
                    for i,entry in enumerate(part):
                        scored.append(dict(population=role,split=split,row_index=entry["row_index"],
                            character_id=entry["row"]["character_id"],y_id=entry["label"],
                            is_anomaly=int(entry["label"]!=normal["normal_class"]),
                            **{name:scores[name][i].item() for name in SCORE_NAMES}))
                rows.extend(scored)
                return scored
            calibration=score_population("calibration",samples["calibration"],"val")
            thresholds={name:conformal_threshold([r[name] for r in calibration],cfg["alpha"]) for name in SCORE_NAMES}
            # Threshold artifact is committed before held-out or test scoring.
            _write(destination/"thresholds.json",thresholds)
            evaluation=score_population("evaluation",samples["evaluation"],"val")
            populations={"evaluation":evaluation}
            if cfg["evaluate_test"]:
                test_speakers=sorted(pipeline.groups["test"],key=lambda s:hashlib.sha256(f"{cfg['seed']}:{s}".encode()).hexdigest())
                test_samples,test_sampling=_sample_rows(pipeline,"test",[("test",test_speakers,cfg["max_test"],False)],
                    normal["normal_class"],cfg,deadline)
                _write(destination/"test_sampling.json",test_sampling)
                populations["test"]=score_population("test",test_samples["test"],"test")
            if before!=_model_digest(model) or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Frozen inference mutated model weights or gradients")
            metrics={role:{name:{**detection_metrics([r[name] for r in population],[r["is_anomaly"] for r in population],thresholds[name]["threshold"]),
                        "normal_score_distribution":_score_summary([r[name] for r in population if not r["is_anomaly"]]),
                        "anomaly_score_distribution":_score_summary([r[name] for r in population if r["is_anomaly"]])}
                       for name in SCORE_NAMES} for role,population in populations.items()}
        with (destination/"scores.csv").open("x",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=["population","split","row_index","character_id","y_id","is_anomaly",*SCORE_NAMES])
            writer.writeheader()
            writer.writerows(rows)
        _write(destination/"metrics.json",metrics)
        result.update(status="completed",thresholds=thresholds,metrics=metrics,sampling=sampling,
                      scored_examples=len(rows),model_unchanged=True,
                      calibration_status="finite_thresholds" if all(t["threshold"] is not None for t in thresholds.values()) else
                          "insufficient_normal_calibration_thresholds_undefined",
                      test_evaluated=cfg["evaluate_test"])
    except Exception as error:
        result.update(status="failed",reason=f"{type(error).__name__}: {error}")
    finally:
        if original_hash is not None:
            try:
                unchanged=_sha256(source)==original_hash
            except OSError:
                unchanged=False
            result["checkpoint_modified"]=not unchanged
            if not unchanged:
                result.update(status="failed",reason="Historical checkpoint changed during evaluation")
        result["elapsed_seconds"]=time.monotonic()-started
        _write(destination/"summary.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/anomaly_evaluation.json")
    parser.add_argument("--data-config",type=Path,default=ROOT/"config.json")
    parser.add_argument("--processed-dir",type=Path)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--evaluate-test",action="store_true",help="Opt in to mixed test AFTER normal val threshold calibration")
    parser.add_argument("--execute",action="store_true",help="Read-only scoring only, never training")
    parser.add_argument("--_worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    cfg=json.loads(args.config.read_text(encoding="utf-8"))
    if args.evaluate_test:
        cfg["evaluate_test"]=True
    cfg=validate_config(cfg)
    if args.execute and not args._worker:
        command=[sys.executable,"-m","src.anomaly_evaluation",*(sys.argv[1:] if argv is None else argv),"--_worker"]
        try:
            return subprocess.run(command,check=False,timeout=cfg["max_seconds"]+20).returncode
        except subprocess.TimeoutExpired:
            print("Anomaly evaluation exceeded wall cap; existing output is partial, not completed",file=sys.stderr)
            return 124
    data=load_config(args.data_config)
    if args.processed_dir:
        data["processed_dir"]=args.processed_dir.resolve()
    result=run_anomaly_evaluation(DataPipeline(data),args.checkpoint,args.output_dir,config=cfg,device=args.device,execute=args.execute)
    print(json.dumps(result if not args.execute else {"status":result["status"],"output_dir":str(args.output_dir)},indent=2))
    return 0 if result.get("status","completed")=="completed" else 1


if __name__=="__main__":
    raise SystemExit(main())
