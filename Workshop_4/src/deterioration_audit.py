"""Phase 4: read-only, provenance-aware investigation of epochs 8--15.

Only stable copied logs are parsed. Missing evidence is never represented as
zero events. This module imports no model, optimizer, torch or training code.
Quantiles use a deterministic bounded reservoir and are marked approximate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import zipfile

from . import audit_logs
from .data import ROOT


CONFIRMED_CHECKPOINTS = (
    {"epoch": 8, "reconstruction_nats_per_sequence": 160.400189,
     "baseline_marginal_nats_per_sequence": 252.388629,
     "swap_damage_nats_per_sequence": 359.958162, "probability_tv": 0.662012},
    {"epoch": 30, "reconstruction_nats_per_sequence": 250.129855,
     "baseline_marginal_nats_per_sequence": 252.388629,
     "swap_damage_nats_per_sequence": 3.640303, "probability_tv": 0.013355},
)
DEFAULT_REPORTED_HISTORY = ROOT / "audit/evidence/user_reported_history.json"
RESERVOIR_SIZE = 2048
MAX_DISTINCT_BATCHES = 65536
MAX_GROUPS = 128


def confirmed_checkpoint_comparison():
    first, last = [dict(row) for row in CONFIRMED_CHECKPOINTS]
    for row in (first, last):
        row["reconstruction_nats_per_character"] = row["reconstruction_nats_per_sequence"] / 80
        row["gain_over_marginal_nats_per_sequence"] = (
            row["baseline_marginal_nats_per_sequence"] - row["reconstruction_nats_per_sequence"])
    return {
        "provenance": "user_confirmed_checkpoint_evaluation; not reevaluated from local weights",
        "validation_scope": "same fixed 128 validation sequences, 80 positions each; not full validation",
        "rows": [first, last],
        "reconstruction_increase_nats_per_sequence": last["reconstruction_nats_per_sequence"] - first["reconstruction_nats_per_sequence"],
        "reconstruction_relative_increase": last["reconstruction_nats_per_sequence"] / first["reconstruction_nats_per_sequence"] - 1,
        "swap_damage_retained_fraction": last["swap_damage_nats_per_sequence"] / first["swap_damage_nats_per_sequence"],
        "probability_tv_retained_fraction": last["probability_tv"] / first["probability_tv"],
        "conclusion": "Example-specific reconstruction and latent dependence deteriorated markedly; root cause remains unestablished.",
    }


def inventory_local_logs(workshop_dir=ROOT):
    """Inventory only; never select a smoke log as a production trace."""
    workshop_dir = Path(workshop_dir).resolve()
    result, errors = [], []
    excluded = {"tmp", "exports", ".git", ".venv", "venv", ".pytest_cache", "__pycache__"}
    wanted = {"history.csv", "gradient_norms.csv", "nonfinite_events.jsonl"}
    for directory, subdirs, names in os.walk(workshop_dir, onerror=lambda error: errors.append(str(error))):
        subdirs[:] = sorted(name for name in subdirs if name not in excluded and not (Path(directory) / name).is_symlink())
        for name in sorted(set(names) & wanted):
            path = Path(directory) / name
            result.append({"path": str(path), "kind": name,
                           "scope": "smoke_only" if "_smoke_real" in path.parts else "unverified_run_scope",
                           "automatically_used_as_production": False})
    return {"files": result, "unreadable_directories": errors,
            "scope": "Workshop_4 excluding tmp/exports/environments; no external Drive inspection"}


def _source_status(path):
    if path is None:
        return {"status": "not_provided", "path": None}
    path = Path(path).resolve()
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    return {"status": "available", "path": str(path), **audit_logs._fingerprint(path)}


def _history_summary(rows, start, end, provenance):
    selected = [row for row in rows if start <= row["epoch"] <= end]
    counts = Counter(row["epoch"] for row in selected)
    all_counts = Counter(row["epoch"] for row in rows)
    by_epoch = {row["epoch"]: row for row in rows if all_counts[row["epoch"]] == 1}
    reference = by_epoch.get(8)
    normalized = []
    for row in selected:
        train = row.get("train_reconstruction")
        val = row.get("val_reconstruction")
        train_field, val_field = "train_reconstruction", "val_reconstruction"
        if train is None:
            train, train_field = row.get("train_total"), "train_total_fallback"
        if val is None:
            val, val_field = row.get("val_total"), "val_total_fallback"
        ref_train = reference.get("train_reconstruction") if reference else None
        ref_val = reference.get("val_reconstruction") if reference else None
        if reference and ref_train is None:
            ref_train = reference.get("train_total")
        if reference and ref_val is None:
            ref_val = reference.get("val_total")
        normalized.append({**row, "train_nats_per_sequence": train, "val_nats_per_sequence": val,
                           "train_field_used": train_field, "val_field_used": val_field,
                           "val_relative_to_epoch8": val / ref_val - 1 if val is not None and ref_val else None,
                           "train_relative_to_epoch8": train / ref_train - 1 if train is not None and ref_train else None})
    transitions = []
    unique = sorted((row for row in normalized if counts[row["epoch"]] == 1), key=lambda row: row["epoch"])
    for previous, current in zip(unique, unique[1:]):
        def ratio(field):
            left, right = previous[field], current[field]
            return right / left if left and right is not None else None
        transitions.append({"from_epoch": previous["epoch"], "to_epoch": current["epoch"],
                            "adjacent_epochs": current["epoch"] == previous["epoch"] + 1,
                            "train_ratio": ratio("train_nats_per_sequence"),
                            "val_ratio": ratio("val_nats_per_sequence"),
                            "interpretation": "descriptive recorded transition, not a causal change-point estimate"})
    return {"provenance": provenance, "rows": normalized,
            "missing_epochs": [epoch for epoch in range(start, end+1) if epoch not in counts],
            "duplicate_epochs": sorted(epoch for epoch, count in counts.items() if count > 1),
            "chronological_order": all(a["epoch"] < b["epoch"] for a,b in zip(selected, selected[1:])),
            "transitions": transitions,
            "large_adjacent_increases": [t for t in transitions if t["adjacent_epochs"] and
                ((t["val_ratio"] is not None and t["val_ratio"] >= 1.5) or
                 (t["train_ratio"] is not None and t["train_ratio"] >= 1.5))],
            "epoch8_reference_available": reference is not None,
            "scope_limit": "History is an epoch aggregate; scope/counts must be checked before comparing with the separate 128-sequence checkpoint diagnostic."}


def _quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    # Weighted sum avoids overflow in upper-lower for very large positive data.
    return values[lower] * (upper-index) + values[upper] * (index-lower) if lower != upper else values[lower]


def _gradient_details(path, start, end, assumed_clip):
    groups = {}
    total_rows = 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            total_rows += 1
            epoch = audit_logs._epoch(row.get("epoch"))
            if not start <= epoch <= end:
                continue
            key = (row.get("session") or "unknown", epoch)
            if key not in groups:
                if len(groups) >= MAX_GROUPS:
                    raise ValueError("Too many focused session/epoch groups for bounded audit")
                seed = int.from_bytes(hashlib.sha256(f"{key}".encode()).digest()[:8], "little")
                groups[key] = {"session": key[0], "epoch": epoch, "logged_rows": 0,
                    "before_mean": 0.0, "before_max": 0.0, "after_max": 0.0, "clipped_rows": 0,
                    "scale_count": 0, "scale_mean": 0.0, "scale_min": None, "zero_scales": 0,
                    "norm_above_1e8_rows": 0, "fp32_square_reduction_risk_rows": 0,
                    "postclip_limit_violations_under_assumed_clip": 0,
                    "min_batch_index": None, "max_batch_index": None,
                    "_indices": set(), "_indices_capped": False, "_sample": [], "_rng": random.Random(seed)}
            group = groups[key]
            before = audit_logs._number(row.get("norm_before"), "norm_before")
            after = audit_logs._number(row.get("norm_after"), "norm_after")
            if min(before, after) < 0:
                raise ValueError("Gradient norms must be nonnegative")
            batch = audit_logs._number(row.get("batch_index"), "batch_index")
            if batch < 1 or not batch.is_integer():
                raise ValueError("batch_index must be a positive integer")
            batch = int(batch)
            clipped = str(row.get("clipped", "")).lower()
            if clipped not in ("true", "false", "1", "0"):
                raise ValueError("clipped must be a recorded boolean")
            group["logged_rows"] += 1
            n = group["logged_rows"]
            group["before_mean"] += (before-group["before_mean"]) / n
            group["before_max"] = max(group["before_max"], before)
            group["after_max"] = max(group["after_max"], after)
            group["clipped_rows"] += int(clipped in ("true", "1"))
            group["norm_above_1e8_rows"] += before > 1e8
            group["fp32_square_reduction_risk_rows"] += before > math.sqrt(3.4028234663852886e38)
            group["postclip_limit_violations_under_assumed_clip"] += after > assumed_clip*(1+1e-6)
            group["min_batch_index"] = batch if group["min_batch_index"] is None else min(group["min_batch_index"],batch)
            group["max_batch_index"] = batch if group["max_batch_index"] is None else max(group["max_batch_index"],batch)
            if len(group["_indices"]) < MAX_DISTINCT_BATCHES:
                group["_indices"].add(batch)
            elif batch not in group["_indices"]:
                group["_indices_capped"] = True
            sample = group["_sample"]
            if len(sample) < RESERVOIR_SIZE:
                sample.append(before)
            else:
                replace = group["_rng"].randrange(n)
                if replace < RESERVOIR_SIZE:
                    sample[replace] = before
            scale = audit_logs._number(row.get("scale"), "scale", optional=True)
            if scale is not None:
                if scale < 0 or scale > 1:
                    raise ValueError("recorded clipping scale must be in [0,1]")
                group["scale_count"] += 1
                group["scale_mean"] += (scale-group["scale_mean"]) / group["scale_count"]
                group["scale_min"] = scale if group["scale_min"] is None else min(group["scale_min"],scale)
                group["zero_scales"] += scale == 0
    return groups, total_rows


def _event_details(path, start, end):
    groups = {}
    focused = 0
    total = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream,1):
            if not line.strip():
                continue
            event = audit_logs._load_json(line)
            if not isinstance(event, dict):
                raise ValueError("Each event must be a JSON object")
            total += 1
            epoch = audit_logs._epoch(event.get("epoch"))
            if not start <= epoch <= end:
                continue
            focused += 1
            key = (event.get("session") or "unknown",epoch)
            if key not in groups:
                if len(groups) >= MAX_GROUPS:
                    raise ValueError("Too many focused session/epoch event groups")
                groups[key] = {"session": key[0], "epoch": epoch, "events": 0,
                    "reasons": Counter(), "actions": Counter(), "categories": Counter(),
                    "first_file_line": line_number, "minimum_recorded_batch_index": None, "_batches": set()}
            group = groups[key]
            group["events"] += 1
            reason = str(event.get("reason", "unknown"))
            group["reasons"][reason] += 1
            group["actions"][str(event.get("action", "unknown"))] += 1
            category = ("finite_gradient_clipping" if reason == "large_finite_gradient_clipped" else
                        "nonfinite_detected" if "nonfinite" in reason else
                        "rejection_budget" if "skipped" in reason or "consecutive" in reason else "other")
            group["categories"][category] += 1
            batch = audit_logs._number(event.get("batch_index"), "batch_index", optional=True)
            if batch is not None:
                if batch < 1 or not batch.is_integer():
                    raise ValueError("event batch_index must be positive integer or null")
                batch = int(batch)
                group["minimum_recorded_batch_index"] = batch if group["minimum_recorded_batch_index"] is None else min(batch,group["minimum_recorded_batch_index"])
                if len(group["_batches"]) < MAX_DISTINCT_BATCHES:
                    group["_batches"].add(batch)
    return groups, total, focused


def analyze_deterioration_logs(*, history=None, gradient_norms=None, events=None, metadata=None,
                               reported_history=DEFAULT_REPORTED_HISTORY,
                               start_epoch=8, end_epoch=15, assumed_clip=5.0):
    if type(start_epoch) is not int or type(end_epoch) is not int or not 1 <= start_epoch <= end_epoch <= 10000:
        raise ValueError("Invalid epoch window")
    if isinstance(assumed_clip,bool) or not math.isfinite(assumed_clip) or assumed_clip <= 0:
        raise ValueError("assumed_clip must be finite and positive")
    sources = {name:_source_status(path) for name,path in (("history",history),
        ("gradient_norms",gradient_norms),("events",events),("metadata",metadata),("reported_history",reported_history))}
    existing = {name:Path(status["path"]) for name,status in sources.items() if status["status"] == "available"}
    result = {"schema_version":1, "epoch_window":[start_epoch,end_epoch], "sources":sources,
        "confirmed_checkpoint_comparison":confirmed_checkpoint_comparison(),
        "history":None, "gradient_norms":None, "events":None, "metadata":None,"cross_log_alignment":[],
        "assumed_clip_norm":assumed_clip, "optimizer_steps":0,
        "limitations":["Missing logs are unavailable evidence, not zero events or zero gradients.",
            "Empty event files contain zero logged events; instrumentation/coverage is not established by emptiness.",
            "nonfinite_events.jsonl can contain accepted finite-gradient clipping events; filename is not event classification.",
            "Clipping frequency alone does not establish the cause of model deterioration.",
            "History epoch aggregates and the fixed 128-sequence checkpoint diagnostic are distinct validation scopes.",
            "Time alignment supports descriptive co-occurrence only; no causal claim or reconstruction of missing batches."]}
    if "history" in existing:
        parsed = audit_logs.audit_logs(history=existing["history"])["history"]
        result["history"] = _history_summary(parsed["rows"],start_epoch,end_epoch,"parsed_stable_history_csv")
        result["history"]["file_recorded_epochs"] = parsed["recorded_epochs"]
        sources["history"]["status"] = "present_empty" if not parsed["rows"] else "parsed"
    elif "reported_history" in existing:
        payload = audit_logs._load_json(existing["reported_history"].read_text(encoding="utf-8"))
        rows = [{"epoch":audit_logs._epoch(row.get("epoch")),
                 "train_reconstruction":audit_logs._number(row.get("train_loss"),"train_loss",optional=True),
                 "val_reconstruction":audit_logs._number(row.get("val_loss"),"val_loss",optional=True)}
                for row in payload.get("rows",[])]
        result["history"] = _history_summary(rows,start_epoch,end_epoch,"earlier_user_reported_history; not raw Colab logs")
        sources["reported_history"]["status"] = "parsed_user_report"
    if "metadata" in existing:
        result["metadata"] = audit_logs._load_json(existing["metadata"].read_text(encoding="utf-8-sig"))
        if not isinstance(result["metadata"],dict):
            raise ValueError("metadata.json must contain an object")
        sources["metadata"]["status"] = "parsed"
    gradient_groups, gradient_total = _gradient_details(existing["gradient_norms"],start_epoch,end_epoch,assumed_clip) if "gradient_norms" in existing else ({},None)
    event_groups, event_total, event_focused = _event_details(existing["events"],start_epoch,end_epoch) if "events" in existing else ({},None,None)
    if "gradient_norms" in existing:
        sources["gradient_norms"]["status"] = "present_empty" if gradient_total == 0 else "parsed" if gradient_groups else "no_rows_in_epoch_window"
        result["gradient_norms"] = []
        for key, group in sorted(gradient_groups.items()):
            exported = {k:v for k,v in group.items() if not k.startswith("_")}
            exported.update(clipped_ratio_logged_rows=group["clipped_rows"]/group["logged_rows"],
                distinct_logged_batches=len(group["_indices"]) if not group["_indices_capped"] else None,
                duplicate_batch_rows=group["logged_rows"]-len(group["_indices"]) if not group["_indices_capped"] else None,
                batch_index_tracking_capped=group["_indices_capped"],
                norm_before_quantiles={str(q):_quantile(group["_sample"],q) for q in (.5,.9,.99)},
                quantile_sample_size=len(group["_sample"]),
                quantiles_exact=group["logged_rows"]<=RESERVOIR_SIZE,
                quantile_method="deterministic reservoir, at most 2048 finite logged norms per session/epoch",
                coverage="logged rows only; repeated/sampled attempts cannot establish full-epoch clipping")
            if not group["scale_count"]:
                exported["scale_mean"] = None
            result["gradient_norms"].append(exported)
    if "events" in existing:
        sources["events"]["status"] = "present_empty" if event_total == 0 else "parsed"
        result["events"] = {"total_logged_events":event_total,"focused_logged_events":event_focused,
            "groups":[{k:v for k,v in group.items() if not k.startswith("_")} for _,group in sorted(event_groups.items())],
            "periods":{name:sum(g["events"] for g in event_groups.values() if predicate(g["epoch"]))
                       for name,predicate in (("before_epoch12",lambda e:e<12),("epoch12_onward",lambda e:e>=12))}}
    for key, group in sorted(gradient_groups.items()):
        matched = event_groups.get(key)
        exact_session = key[0] != "unknown"
        result["cross_log_alignment"].append({"session":key[0],"epoch":key[1],
            "session_alignment_recorded":exact_session,
            "gradient_logged_rows":group["logged_rows"],
            "events_same_recorded_session_epoch":matched["events"] if matched and exact_session else None,
            "overlapping_logged_event_batch_indices":len(group["_indices"] & matched["_batches"]) if matched and exact_session else None,
            "history_alignment":"epoch only; history lacks a recorded session identity, no causal linkage established"})
    for name,path in existing.items():
        before = {key:sources[name][key] for key in ("sha256","size_bytes","mtime_ns")}
        if audit_logs._fingerprint(path) != before:
            raise RuntimeError(f"Source changed during focused audit: {path}")
    result["sources_unchanged"] = True
    result["root_cause_established"] = False
    # Finite inputs can still overflow a derived ratio; reject before creating
    # any result directory or emitting non-standard JSON numbers.
    json.dumps(result,allow_nan=False)
    return result


def stage_log_bundle(source_dir, destination_dir, zip_path=None):
    """Copy exactly four stable historical logs into a new folder and ZIP.

    Run after Drive mount against the user-specified completed run directory.
    No checkpoint directory is inspected or copied. A changed source aborts;
    partially copied new files are not presented as an audited bundle.
    """
    source_dir, destination_dir = Path(source_dir).resolve(),Path(destination_dir).resolve()
    if destination_dir.exists():
        raise FileExistsError("Stable log copies require a new destination folder")
    if source_dir == destination_dir or source_dir in destination_dir.parents:
        raise ValueError("Log copies must live outside the historical run directory")
    archive_path = Path(zip_path).resolve() if zip_path is not None else destination_dir/"historical_log_bundle.zip"
    if archive_path.exists():
        raise FileExistsError("Log ZIP must use a new path")
    if source_dir == archive_path or source_dir in archive_path.parents:
        raise ValueError("Log ZIP cannot be written inside the historical run directory")
    filenames = {"history":"history.csv","gradient_norms":"gradient_norms.csv",
                 "events":"nonfinite_events.jsonl","metadata":"metadata.json"}
    original = {}
    for key,name in filenames.items():
        path = source_dir/name
        if not path.is_file():
            raise FileNotFoundError(f"Missing required historical log: {path}")
        if path.is_symlink():
            raise ValueError(f"Historical log sources must be regular files, not links: {path}")
        original[key] = audit_logs._fingerprint(path)
    destination_dir.mkdir(parents=True,exist_ok=False)
    copies = {}
    for key,name in filenames.items():
        destination = destination_dir/name
        with (source_dir/name).open("rb") as incoming,destination.open("xb") as outgoing:
            for chunk in iter(lambda:incoming.read(1024*1024),b""):
                outgoing.write(chunk)
        if audit_logs._fingerprint(destination)["sha256"] != original[key]["sha256"]:
            raise RuntimeError(f"Copied log differs from initial source: {name}")
        copies[key] = str(destination)
    for key,name in filenames.items():
        if audit_logs._fingerprint(source_dir/name) != original[key]:
            raise RuntimeError(f"Historical log changed during copy: {name}; partial destination is not a valid bundle")
    archive_path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive_path,"x",compression=zipfile.ZIP_DEFLATED) as archive:
        for key,name in filenames.items():
            archive.write(Path(copies[key]),name)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(filenames.values()):
            raise RuntimeError("Log ZIP integrity failed")
        for key,name in filenames.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != original[key]["sha256"]:
                raise RuntimeError("Log ZIP bytes differ from the stable source")
    # Detect changes in either the copies or sources while building the archive.
    for key,name in filenames.items():
        if audit_logs._fingerprint(source_dir/name) != original[key] or audit_logs._fingerprint(Path(copies[key]))["sha256"] != original[key]["sha256"]:
            raise RuntimeError(f"Log changed during ZIP creation: {name}; bundle is not validated")
    return {"status":"validated_stable_log_copies","source_dir":str(source_dir),
            "destination_dir":str(destination_dir),"copied_files":copies,
            "source_fingerprints":original,"zip_path":str(archive_path),
            "zip_sha256":audit_logs._fingerprint(archive_path)["sha256"],
            "checkpoints_included":False}


def write_deterioration_audit(output_dir, *, execute=False, **sources):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Choose a new Phase 4 audit output directory")
    for key in ("history","gradient_norms","events","metadata","reported_history"):
        path = sources.get(key)
        if path is not None and output_dir in Path(path).resolve().parents:
            raise ValueError("Output cannot contain an input source")
    plan = {"phase":4,"operation":"read_only_focused_log_audit","execute":execute,
            "epoch_window":[sources.get("start_epoch",8),sources.get("end_epoch",15)],
            "inputs":{key:str(value) if value is not None else None for key,value in sources.items()},
            "optimizer_steps":0,"outputs":"new report.json only; no historical files modified"}
    if not execute:
        return plan
    report = analyze_deterioration_logs(**sources)
    output_dir.mkdir(parents=True,exist_ok=False)
    with (output_dir/"report.json").open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history",type=Path)
    parser.add_argument("--gradient-norms",type=Path)
    parser.add_argument("--events",type=Path)
    parser.add_argument("--metadata",type=Path)
    parser.add_argument("--reported-history",type=Path,default=DEFAULT_REPORTED_HISTORY)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--start-epoch",type=int,default=8)
    parser.add_argument("--end-epoch",type=int,default=15)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args(argv)
    report = write_deterioration_audit(args.output_dir,execute=args.execute,
        history=args.history,gradient_norms=args.gradient_norms,events=args.events,metadata=args.metadata,
        reported_history=args.reported_history,start_epoch=args.start_epoch,end_epoch=args.end_epoch)
    print(json.dumps(report if not args.execute else {"output":str(args.output_dir/"report.json"),
        "root_cause_established":False},indent=2,ensure_ascii=False,allow_nan=False))


if __name__ == "__main__":
    main()
