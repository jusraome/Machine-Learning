"""Read-only checkpoint/Adam provenance inspection; never takes an optimizer step.

An inference checkpoint may contain a full Adam mapping without proving that
its moments and RNG belong to the saved best weights. Structural restoration,
temporal alignment and historical replay are deliberately separate conclusions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import torch

from .autoencoder import Autoencoder
from .benchmark_ae import _preserve_rng_and_flags
from .data import ROOT, DataPipeline, load_config


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree(value):
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous()
        return {"tensor_shape": list(data.shape), "dtype": str(data.dtype),
                "sha256": hashlib.sha256(data.numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(k): _tree(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_tree(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return {"unsupported_type": type(value).__name__}


def value_fingerprint(value):
    return hashlib.sha256(json.dumps(_tree(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _normalized_optimizer(state):
    groups = state.get("param_groups", [])
    ordered = [identifier for group in groups for identifier in group.get("params", [])]
    mapping = {identifier: f"parameter_{i}" for i, identifier in enumerate(ordered)}
    return {"param_groups": [{**group, "params": [mapping[i] for i in group["params"]]} for group in groups],
            "state": {mapping[i]: value for i, value in state.get("state", {}).items() if i in mapping}}


def _valid_counter(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 and number.is_integer() else None
    except (TypeError, ValueError, RuntimeError):
        return None


def _rng_inventory(state):
    issues = []
    if not isinstance(state, dict):
        return {"complete": False, "issues": ["rng_state mapping missing"]}
    try:
        random.Random().setstate(state["python"])
    except (KeyError, TypeError, ValueError):
        issues.append("Python RNG missing or invalid")
    try:
        torch.Generator(device="cpu").set_state(state["torch"].cpu())
    except (KeyError, TypeError, RuntimeError, AttributeError):
        issues.append("CPU Torch RNG missing or invalid")
    cuda = state.get("cuda")
    if not isinstance(cuda, list) or any(not isinstance(v, torch.Tensor) or v.dtype != torch.uint8 or v.ndim != 1 or not v.numel() for v in cuda):
        issues.append("CUDA RNG inventory missing or invalid")
    return {"complete": not issues, "issues": issues, "cuda_generators": len(cuda) if isinstance(cuda, list) else None,
            "fingerprint": value_fingerprint(state), "restoration_check": "isolated Python Random and CPU Torch Generator; no global RNG mutation"}


def _adam_inventory(checkpoint, model):
    state = checkpoint.get("optimizer_state_dict")
    report = {"present": isinstance(state, dict), "structurally_complete": False,
              "parameter_mapping": "original trainer uses one Adam group in model.parameters() order",
              "issues": [], "group_hyperparameters": [], "parameter_states": [], "step_min": None, "step_max": None}
    if not isinstance(state, dict) or not isinstance(state.get("state"), dict) or not isinstance(state.get("param_groups"), list):
        report["issues"].append("optimizer_state_dict requires state and param_groups")
        return report
    groups = state["param_groups"]
    if len(groups) != 1:
        report["issues"].append("Original trainer requires one group; arbitrary parameter ordering cannot be proven")
    identifiers = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            report["issues"].append("Invalid Adam parameter group")
            continue
        identifiers.extend(group["params"])
        hyper = {key: group.get(key) for key in ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused")}
        report["group_hyperparameters"].append(_tree(hyper))
        for name in ("lr", "eps", "weight_decay"):
            value = group.get(name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0 or (name == "eps" and value <= 0):
                report["issues"].append(f"Invalid/missing Adam group {name}")
        betas = group.get("betas")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2 or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v < 1 for v in betas):
            report["issues"].append("Invalid/missing Adam betas")
        if type(group.get("amsgrad")) is not bool:
            report["issues"].append("Invalid/missing Adam amsgrad flag")
    named = list(model.named_parameters())
    if len(identifiers) != len(named) or len(set(identifiers)) != len(identifiers):
        report["issues"].append("Adam IDs must uniquely cover every model parameter")
    if set(state["state"]) != set(identifiers):
        report["issues"].append("Adam state keys do not exactly match grouped parameter IDs")
    steps = []
    amsgrad = any(group.get("amsgrad") for group in groups)
    for (name, parameter), identifier in zip(named, identifiers):
        entry = state["state"].get(identifier)
        problems = []
        if not isinstance(entry, dict):
            problems.append("missing parameter state")
            entry = {}
        step = _valid_counter(entry.get("step"))
        if step is None:
            problems.append("missing/invalid integer Adam step")
        else:
            steps.append(step)
        for key in ("exp_avg", "exp_avg_sq") + (("max_exp_avg_sq",) if amsgrad else ()):
            tensor = entry.get(key)
            if not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                problems.append(f"{key} missing or shape/dtype mismatch")
            elif not torch.isfinite(tensor).all().item() or (key != "exp_avg" and (tensor < 0).any().item()):
                problems.append(f"{key} nonfinite or invalid negative second moment")
        report["parameter_states"].append({"name": name, "saved_id": str(identifier), "step": step, "issues": problems})
        report["issues"].extend(f"{name}: {problem}" for problem in problems)
    report.update(structurally_complete=not report["issues"], step_min=min(steps) if steps else None,
                  step_max=max(steps) if steps else None, fingerprint=value_fingerprint(state))
    return report


def _temporal_inventory(checkpoint, adam, comparison=None):
    issues, unresolved = [], []
    epoch = checkpoint.get("epoch")
    history = checkpoint.get("history")
    epochs = [row.get("epoch") for row in history if isinstance(row, dict)] if isinstance(history, list) else []
    valid_epochs = [value for value in epochs if type(value) is int and value >= 1]
    if len(valid_epochs) != len(epochs) or len(set(valid_epochs)) != len(valid_epochs):
        issues.append("history epoch sequence is invalid or duplicated")
    if valid_epochs and type(epoch) is int and max(valid_epochs) != epoch:
        issues.append("saved weight epoch differs from final history epoch; recovered-best envelope may carry later Adam/RNG")
    if not valid_epochs:
        unresolved.append("No complete epoch history establishes optimizer time")
    expected = None
    if type(epoch) is int and epoch >= 1 and sorted(valid_epochs) == list(range(1, epoch + 1)):
        accepted = [_valid_counter(row.get("train_accepted_batches")) for row in history]
        if all(value is not None for value in accepted):
            expected = int(sum(accepted))
            if adam["step_min"] != expected or adam["step_max"] != expected:
                issues.append("Adam parameter steps disagree with summed accepted training batches")
        else:
            unresolved.append("History lacks accepted-batch counts; step-to-epoch alignment unproven")
    if checkpoint.get("checkpoint_role") == "best_inference_only":
        unresolved.append("Inference-only role does not certify an aligned resumable training state")
    other = None
    if comparison is not None:
        same_adam = value_fingerprint(checkpoint.get("optimizer_state_dict")) == value_fingerprint(comparison.get("optimizer_state_dict"))
        same_rng = value_fingerprint(checkpoint.get("rng_state")) == value_fingerprint(comparison.get("rng_state"))
        same_weights = value_fingerprint(checkpoint.get("state_dict")) == value_fingerprint(comparison.get("state_dict"))
        other = dict(epoch=comparison.get("epoch"), same_adam=same_adam, same_rng=same_rng, same_weights=same_weights)
        if type(epoch) is int and type(comparison.get("epoch")) is int and comparison["epoch"] > epoch and same_adam and same_rng and not same_weights:
            issues.append("Earlier weights share Adam and RNG exactly with later comparison checkpoint: temporal mixing evidence")
    return dict(status="inconsistent" if issues else "unproven" if unresolved else "consistent_epoch_boundary",
                issues=issues, unresolved=unresolved, weight_epoch=epoch,
                final_history_epoch=max(valid_epochs) if valid_epochs else None,
                expected_steps_from_accepted_batches=expected, comparison=other)


def inspect_checkpoint(checkpoint, pipeline=None, metadata=None, comparison_checkpoint=None):
    """Inspect a loaded mapping and test restoration on disposable objects only."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a mapping")
    report = {"schema_version": 1, "optimizer_steps_performed": 0, "checkpoint_role": checkpoint.get("checkpoint_role"),
              "model_type": checkpoint.get("model_type"), "epoch": checkpoint.get("epoch"),
              "weights_fingerprint": value_fingerprint(checkpoint.get("state_dict")),
              "availability": "inspected", "historical_faithful_continuation_eligible": False}
    hparams = checkpoint.get("model_hparams", {})
    expected = dict(embedding_dim=32, hidden_size=128, latent_dim=32, num_layers=1, dropout=0.0)
    report["original_ae_z32"] = checkpoint.get("model_type") == "ae" and all(hparams.get(k) == v for k, v in expected.items()) and hparams.get("latent_activation", "identity") == "identity"
    report["configuration"] = {"model_hparams": _tree(hparams), "training_config": _tree(checkpoint.get("training_config")),
        "runtime": _tree(checkpoint.get("runtime")), "gradient_clipping": _tree(checkpoint.get("gradient_clipping")),
        "stability_policy_present": isinstance(checkpoint.get("stability_policy"), dict),
        "stability_state_present": isinstance(checkpoint.get("stability_state"), dict)}
    if pipeline is not None:
        pipeline.validate_checkpoint(checkpoint)
    report["data_contract_validated"] = pipeline is not None
    report["data_manifest_present"] = isinstance(checkpoint.get("data_manifest"), dict)
    manifest = checkpoint.get("data_manifest")
    manifest_files = manifest.get("files", {}) if isinstance(manifest, dict) else {}
    report["data_manifest_complete"] = (isinstance(manifest, dict)
        and manifest.get("data_contract") == checkpoint.get("data_contract")
        and isinstance(manifest.get("tokens"), dict)
        and all(isinstance(manifest_files.get(split), dict)
                and isinstance(manifest_files[split].get("sha256"), str)
                and len(manifest_files[split]["sha256"]) == 64
                and _valid_counter(manifest_files[split].get("rows")) not in (None, 0)
                and _valid_counter(manifest_files[split].get("size_bytes")) is not None
                for split in ("train", "val")))
    report["data_bytes_reverified"] = False
    with _preserve_rng_and_flags():
        if checkpoint.get("model_type") != "ae":
            raise ValueError("Inspection targets deterministic AE checkpoints only")
        model = Autoencoder(**hparams)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        report["model_parameters_finite"] = all(torch.isfinite(p).all().item() for p in model.parameters())
        adam = _adam_inventory(checkpoint, model)
        report["adam"] = adam
        report["rng"] = _rng_inventory(checkpoint.get("rng_state"))
        report["temporal_alignment"] = _temporal_inventory(checkpoint, adam, comparison_checkpoint)
        cfg = checkpoint.get("training_config", {})
        cfg = cfg if isinstance(cfg, dict) else {}
        config_issues = []
        if cfg.get("model") != "ae":
            config_issues.append("Missing/non-AE original training configuration")
        for name in expected:
            if cfg.get(name) != hparams.get(name):
                config_issues.append(f"training_config/model_hparams mismatch or missing {name}")
        for name in ("batch_size", "seed", "learning_rate", "shuffle_buffer_batches", "grad_clip", "normal_class"):
            if name not in cfg:
                config_issues.append(f"training_config lacks {name}")
        if isinstance(metadata, dict):
            metadata_cfg = metadata.get("training_config")
            if metadata_cfg is not None and metadata_cfg != checkpoint.get("training_config"):
                config_issues.append("External metadata training_config differs from checkpoint")
        groups = checkpoint.get("optimizer_state_dict", {}).get("param_groups", []) if isinstance(checkpoint.get("optimizer_state_dict"), dict) else []
        if "learning_rate" in cfg and any(group.get("lr") != cfg["learning_rate"] for group in groups):
            config_issues.append("Saved Adam learning rate differs from training_config")
        original_adam_defaults = dict(betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False, maximize=False)
        for group in groups:
            for key, expected_value in original_adam_defaults.items():
                actual = group.get(key)
                if key == "betas" and isinstance(actual, (list, tuple)):
                    actual = tuple(actual)
                if actual != expected_value:
                    config_issues.append(f"Saved Adam {key} differs from original trainer defaults")
        runtime = checkpoint.get("runtime")
        runtime_complete = isinstance(runtime, dict) and all(key in runtime for key in
            ("tensor_loader", "prefetch_batches", "pin_memory", "profile_batches", "gradient_log_every"))
        clipping = checkpoint.get("gradient_clipping")
        fp64_clipping = isinstance(clipping, dict) and clipping.get("algorithm") == "global_l2_fp64_v1" and clipping.get("norm_dtype") == clipping.get("scaling_dtype") == "float64" and clipping.get("max_norm") == 5.0
        report["resumable_envelope"] = dict(runtime_complete=runtime_complete, fp64_clipping_recorded=fp64_clipping,
            stability_policy_complete=isinstance(checkpoint.get("stability_policy"), dict),
            stability_state_complete=isinstance(checkpoint.get("stability_state"), dict))
        report["configuration"]["issues"] = config_issues
        restored = {"attempted": False, "weights_verified": False, "adam_verified": False,
                    "optimizer_step_called": False, "global_rng_preserved": True}
        if adam["structurally_complete"] and report["model_parameters_finite"]:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            restored.update(attempted=True, weights_verified=value_fingerprint(model.state_dict()) == report["weights_fingerprint"],
                adam_verified=value_fingerprint(_normalized_optimizer(optimizer.state_dict())) == value_fingerprint(_normalized_optimizer(checkpoint["optimizer_state_dict"])))
        report["disposable_restore"] = restored
    report["structural_restore_possible"] = restored["weights_verified"] and restored["adam_verified"]
    complete = (adam["structurally_complete"] and restored["weights_verified"] and restored["adam_verified"] and report["rng"]["complete"]
                and not config_issues and report["data_manifest_complete"] and all(report["resumable_envelope"].values()))
    report["epoch_boundary_state_restorable"] = bool(complete and checkpoint.get("checkpoint_role") == "last_resumable" and report["temporal_alignment"]["status"] == "consistent_epoch_boundary")
    report["configuration"]["environment"] = _tree(checkpoint.get("environment") or (metadata.get("environment") if isinstance(metadata, dict) else None))
    report["continuation_verdict"] = (
        "inconsistent_state_do_not_resume" if report["temporal_alignment"]["status"] == "inconsistent" or not adam["structurally_complete"] else
        "structure_restores_but_historical_fidelity_unproven")
    report["limitations"] = [
        "Full Adam fields and a successful no-step roundtrip do not prove those moments belong to best8 weights.",
        "Phase4 continue restores only Phase4 cursor/order/config/Adam/RNG and a fixed training prefix, not the original epoch stream.",
        "Original epoch-boundary replay additionally requires verified data bytes, original stream/runtime, backend/version and trustworthy aligned RNG.",
        "This auditor never authorizes inference-only best checkpoints as resumable or starts a training window.",
        "A weights_only branch resets Adam and data cursor; its deterioration cannot establish the cause of the original epoch12 failure."]
    return report


def audit_checkpoint(path, pipeline=None, metadata_path=None, comparison_path=None):
    path = Path(path).resolve()
    if not path.is_file():
        return {"availability": "missing", "path": str(path), "inspected": False,
                "historical_faithful_continuation_eligible": False,
                "optimizer_steps_performed": 0,
                "limitation": "Actual checkpoint bytes unavailable here; inspect after mounting Drive, do not infer Adam presence from code."}
    sources = {"checkpoint": path}
    if metadata_path is not None:
        sources["metadata"] = Path(metadata_path).resolve()
    if comparison_path is not None:
        sources["comparison_checkpoint"] = Path(comparison_path).resolve()
    fingerprints = {name: file_sha256(source) for name, source in sources.items()}
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    comparison = torch.load(sources["comparison_checkpoint"], map_location="cpu", weights_only=True) if "comparison_checkpoint" in sources else None
    metadata = json.loads(sources["metadata"].read_text(encoding="utf-8-sig")) if "metadata" in sources else None
    report = inspect_checkpoint(checkpoint, pipeline=pipeline, metadata=metadata, comparison_checkpoint=comparison)
    for name, source in sources.items():
        if file_sha256(source) != fingerprints[name]:
            raise RuntimeError(f"Source changed during checkpoint inspection: {source}")
    report.update(path=str(path), source_sha256=fingerprints, sources_unchanged=True, inspected=True)
    return report


def write_checkpoint_audit(output_dir, *, checkpoint, pipeline=None, metadata_path=None, comparison_path=None, execute=False):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Checkpoint audit needs a new output directory")
    for source in (checkpoint, metadata_path, comparison_path):
        if source is not None and output in Path(source).resolve().parents:
            raise ValueError("Output cannot contain an input checkpoint/log")
    plan = dict(operation="read_only_checkpoint_fidelity_audit", execute=execute, checkpoint=str(checkpoint),
                metadata=str(metadata_path) if metadata_path else None, comparison_checkpoint=str(comparison_path) if comparison_path else None,
                optimizer_steps=0, starts_training=False)
    if not execute:
        return plan
    report = audit_checkpoint(checkpoint, pipeline=pipeline, metadata_path=metadata_path, comparison_path=comparison_path)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--comparison-checkpoint", type=Path)
    parser.add_argument("--data-config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    pipeline = None
    if args.execute and args.checkpoint.is_file():
        config = load_config(args.data_config)
        if args.processed_dir:
            config["processed_dir"] = args.processed_dir.resolve()
        pipeline = DataPipeline(config)
    report = write_checkpoint_audit(args.output_dir, checkpoint=args.checkpoint, pipeline=pipeline,
        metadata_path=args.metadata, comparison_path=args.comparison_checkpoint, execute=args.execute)
    print(json.dumps(report if not args.execute else {"availability": report["availability"], "output": str(args.output_dir / "report.json"),
        "historical_faithful_continuation_eligible": report.get("historical_faithful_continuation_eligible", False)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
