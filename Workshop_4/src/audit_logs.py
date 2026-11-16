"""Streaming audit of stable COPIES of training logs; no model or training access.

Use copies captured after a run/epoch, not files being written by Colab. Hashes
and stat signatures before/after parsing reject sources that change mid-audit.
Ratios from gradient logs describe logged rows, never an assumed full epoch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


def _fingerprint(path):
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant is not permitted: {value}")


def _number(value, field, optional=False):
    if value in (None, "") and optional:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite number in {field}")
    return result


def _epoch(value):
    number = _number(value, "epoch")
    if number < 1 or not number.is_integer():
        raise ValueError("epoch must be a positive integer")
    return int(number)


def _load_json(text):
    # parse_constant catches NaN/Infinity; parse_float also catches e.g. 1e999.
    return json.loads(text, parse_constant=_reject_constant,
                      parse_float=lambda value: _number(value, "JSON number"))


def _read_history(path):
    fields = ("train_reconstruction", "train_kl", "train_total", "val_reconstruction", "val_kl", "val_total",
              "train_examples", "val_examples", "train_attempted_batches", "train_accepted_batches",
              "train_skipped_batches", "train_attempted_examples", "train_skipped_examples", "train_skipped_fraction",
              "train_grad_norm_pre_mean", "train_grad_norm_pre_max", "train_grad_norm_post_max",
              "train_clipped_batches", "train_large_grad_batches")
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append({"epoch": _epoch(row.get("epoch")),
                         **{field: _number(row.get(field), field, optional=True) for field in fields}})
    counts = Counter(row["epoch"] for row in rows)
    candidates = [row for row in rows if row["val_total"] is not None]
    best = min(candidates, key=lambda row: row["val_total"]) if candidates else None
    first_val = candidates[0]["val_total"] if candidates else None
    last_val = candidates[-1]["val_total"] if candidates else None
    ratio = last_val / first_val if first_val and last_val is not None else None
    return {"rows": rows, "recorded_epochs": len(rows), "best_epoch_by_val_total": best["epoch"] if best else None,
            "best_val_total": best["val_total"] if best else None,
            "last_to_first_val_ratio": ratio if ratio is not None and math.isfinite(ratio) else None,
            "duplicate_epochs": [epoch for epoch, count in counts.items() if count > 1],
            "chronological_order": all(left["epoch"] < right["epoch"] for left, right in zip(rows, rows[1:])),
            "missing_audit_values": "null means absent/uninstrumented, not zero"}


def _read_gradient_norms(path):
    groups = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row.get("session") or "unknown", _epoch(row.get("epoch")))
            group = groups.setdefault(key, {"session": key[0], "epoch": key[1], "logged_rows": 0,
                                           "norm_before_mean": 0.0, "norm_before_max": 0.0,
                                           "norm_after_max": 0.0, "clipped_rows": 0, "batch_indices": set()})
            before = _number(row.get("norm_before"), "norm_before")
            after = _number(row.get("norm_after"), "norm_after")
            if min(before, after) < 0:
                raise ValueError("Gradient norms must be nonnegative")
            clipped = str(row.get("clipped", "")).lower()
            if clipped not in ("true", "false", "1", "0"):
                raise ValueError("clipped must be a recorded boolean")
            batch_index = _number(row.get("batch_index"), "batch_index")
            if batch_index < 1 or not batch_index.is_integer():
                raise ValueError("batch_index must be a positive integer")
            group["logged_rows"] += 1
            group["norm_before_mean"] += (before - group["norm_before_mean"]) / group["logged_rows"]
            group["norm_before_max"] = max(group["norm_before_max"], before)
            group["norm_after_max"] = max(group["norm_after_max"], after)
            group["clipped_rows"] += int(clipped in ("true", "1"))
            group["batch_indices"].add(int(batch_index))
    result = []
    for group in groups.values():
        group["distinct_logged_batches"] = len(group.pop("batch_indices"))
        group["duplicate_batch_rows"] = group["logged_rows"] - group["distinct_logged_batches"]
        group["clipped_ratio_logged_rows"] = group["clipped_rows"] / group["logged_rows"]
        group["coverage"] = "logged_rows_only; sampling and interrupted/repeated attempts prevent full-epoch inference"
        result.append(group)
    return result


def _read_events(path):
    groups = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = _load_json(line)
            if not isinstance(event, dict):
                raise ValueError("Each event must be a JSON object")
            key = (event.get("session") or "unknown", _epoch(event.get("epoch")))
            group = groups.setdefault(key, {"session": key[0], "epoch": key[1], "events": 0,
                                           "reasons": Counter(), "actions": Counter()})
            group["events"] += 1
            group["reasons"][str(event.get("reason", "unknown"))] += 1
            group["actions"][str(event.get("action", "unknown"))] += 1
    return list(groups.values())


def audit_logs(history=None, gradient_norms=None, events=None, metadata=None):
    sources = {name: Path(path).resolve() for name, path in
               (("history", history), ("gradient_norms", gradient_norms), ("events", events), ("metadata", metadata))
               if path is not None}
    if not sources:
        raise ValueError("Provide at least one copied log file")
    fingerprints = {name: _fingerprint(path) for name, path in sources.items()}
    result = {"sources": {name: {"path": str(path), **fingerprints[name]} for name, path in sources.items()},
              "limitations": ["read-only copied logs; no active Colab process or Drive checkpoint inspected",
                              "gradient rows grouped by session and epoch; no merging repeated attempts",
                              "logged-row ratios do not imply full-epoch coverage or a causal diagnosis"]}
    for name, path in sources.items():
        if name == "history":
            result[name] = _read_history(path)
        elif name == "gradient_norms":
            result[name] = _read_gradient_norms(path)
        elif name == "events":
            result[name] = _read_events(path)
        else:
            result[name] = _load_json(path.read_text(encoding="utf-8"))
    for name, path in sources.items():
        if _fingerprint(path) != fingerprints[name]:
            raise RuntimeError(f"Source changed during audit; use a stable copy: {path}")
    return result


def write_audit(output, **sources):
    output = Path(output).resolve()
    if output in {Path(path).resolve() for path in sources.values() if path is not None} or output.exists():
        raise FileExistsError("Audit output must be a new file distinct from all inputs")
    report = audit_logs(**sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history", "gradient_norms", "events", "metadata"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_audit(args.output, **{name: getattr(args, name) for name in ("history", "gradient_norms", "events", "metadata")})
    print(args.output)


if __name__ == "__main__":
    main()
