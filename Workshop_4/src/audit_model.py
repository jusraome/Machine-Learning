"""Read-only checkpoint comparison and small gradient probes.

No optimizer step is performed. Train/validation are the only allowed splits.
Probes use evaluation mode and the reference GRU backend (cuDNN disabled) so
that backward is available on CUDA as well as CPU. They are diagnostic probes,
not a throughput benchmark or reproduction of the running Colab process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from .autoencoder import Autoencoder, VariationalAutoencoder, reconstruction_loss, vae_loss
from .data import ROOT, DataPipeline, load_config


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_summary(value):
    value = value.detach().double()
    finite = torch.isfinite(value)
    clean = value[finite]
    norm = torch.linalg.vector_norm(clean).item()
    return {"shape": list(value.shape), "finite": bool(finite.all().item()),
            "nan": int(torch.isnan(value).sum().item()), "inf": int(torch.isinf(value).sum().item()),
            "max_abs_finite": clean.abs().max().item() if clean.numel() else None,
            "mean_abs_finite": clean.abs().mean().item() if clean.numel() else None,
            "finite_part_l2": norm if math.isfinite(norm) else None}


def recurrent_spectra(model):
    """Gate matrix norms are indicators, not spectral norms of GRU Jacobians."""
    result = {}
    for name, parameter in model.named_parameters():
        if "weight_hh" not in name:
            continue
        gates = {}
        for gate, matrix in zip(("reset", "update", "candidate"), parameter.detach().double().chunk(3, dim=0)):
            finite = torch.isfinite(matrix).all().item()
            gates[gate] = {"spectral_norm": torch.linalg.svdvals(matrix).max().item() if finite else None,
                           "finite": bool(finite)}
        result[name] = gates
    return result


def _gradient_blocks(named_parameters, gradients):
    blocks = {}
    for (name, _), gradient in zip(named_parameters, gradients):
        if gradient is None:
            continue
        block = name.split(".", 1)[0]
        entry = blocks.setdefault(block, {"parameters": {}, "nonfinite_elements": 0, "l2": 0.0})
        summary = tensor_summary(gradient)
        entry["parameters"][name] = summary
        entry["nonfinite_elements"] += summary["nan"] + summary["inf"]
        entry["l2"] = math.hypot(entry["l2"], summary["finite_part_l2"] or 0.0)
    return blocks


@torch.no_grad()
def _input_dependence(model, z, targets, reference_logits):
    """Perturb only z on this same batch; not a generative quality metric."""
    reference = reference_logits.detach().double()
    denominator = max(torch.linalg.vector_norm(reference).item(), 1e-12)
    result = {"permutation": "reverse batch order", "permutation_meaningful": len(z) > 1}
    for name, latent in (("original_latent", z), ("permuted_latent", z.flip(0)),
                         ("zero_latent", torch.zeros_like(z))):
        logits = reference_logits.detach() if name == "original_latent" else model.decode(latent.detach())
        loss = reconstruction_loss(logits, targets)
        probabilities = logits.double().softmax(dim=-1)
        entropy = -torch.special.xlogy(probabilities, probabilities).sum(dim=-1).mean()
        difference = torch.linalg.vector_norm(logits.double() - reference).item()
        result[name] = {
            "reconstruction_nats_per_sequence": loss.item() if torch.isfinite(loss).item() else None,
            "relative_logits_l2_change": difference / denominator if math.isfinite(difference) else None,
            "mean_probability_variance_between_examples": probabilities.var(dim=0, unbiased=False).mean().item()
                if len(z) > 1 and torch.isfinite(probabilities).all().item() else None,
            "mean_prediction_entropy_nats_per_character": entropy.item() if torch.isfinite(entropy).item() else None,
            "top1_token_histogram": torch.bincount(logits.argmax(dim=-1).reshape(-1),
                                                   minlength=model.vocab_size).cpu().tolist(),
        }
    result["interpretation_limit"] = "small-batch sensitivity diagnostic; similar losses alone do not prove global latent collapse"
    return result


def probe_model(model, x, beta=1.0, horizons=None):
    """Compute loss and gradients without mutating weights, .grad fields or RNG.

    VAE reconstruction is evaluated at mu for deterministic comparison, and is
    explicitly not the Monte Carlo training ELBO. Decoder sensitivity is a
    fixed-direction vector-Jacobian product at selected output positions.
    """
    if x.ndim != 2 or not len(x):
        raise ValueError("x must be a nonempty token batch")
    previous_modes = {module: module.training for module in model.modules()}
    device = x.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    captures = {}
    def capture_decoder(module, inputs, output):
        captures["decoder_input"] = inputs[0]
        captures["decoder_initial_hidden"] = inputs[1]
        captures["decoder_hidden_sequence"] = output[0]
    handle = model.decoder.register_forward_hook(capture_decoder)
    raw_handle = None
    if isinstance(model, Autoencoder):
        def capture_raw_latent(module, inputs, output):
            captures["raw_latent"] = output
        raw_handle = model.to_latent.register_forward_hook(capture_raw_latent)
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.backends.cudnn.flags(enabled=False):
            with torch.enable_grad():
                output = model(x, sample=False) if isinstance(model, VariationalAutoencoder) else model(x)
                loss = reconstruction_loss(output.logits, x)
                objective = (vae_loss(output.logits, x, output.mu, output.logvar, beta=beta).total
                             if isinstance(model, VariationalAutoencoder) else loss)
                report = {"reconstruction_nats_per_sequence": loss.item() if torch.isfinite(loss).item() else None,
                          "objective_at_mu": objective.item() if torch.isfinite(objective).item() else None,
                          "objective_definition": "CE(x) + beta*KL at mu; diagnostic, not MC ELBO" if
                              isinstance(model, VariationalAutoencoder) else "CE summed over sequence, mean batch",
                          "latent": tensor_summary(output.z), "logits": tensor_summary(output.logits),
                          "latent_activation": getattr(model, "latent_activation", "gaussian_posterior"),
                          "batch_size": len(x), "sequence_length": x.shape[1],
                          "decoder_input_stride": list(captures["decoder_input"].stride()),
                          "decoder_initial_hidden": tensor_summary(captures["decoder_initial_hidden"]),
                          "decoder_hidden_sequence": tensor_summary(captures["decoder_hidden_sequence"])}
                if "raw_latent" in captures:
                    report["raw_latent"] = tensor_summary(captures["raw_latent"])
                report["initial_hidden_tanh_near_saturation_fraction"] = (
                    captures["decoder_initial_hidden"].detach().abs() >= 0.99).double().mean().item()
                report["decoder_hidden_near_saturation_fraction"] = (
                    captures["decoder_hidden_sequence"].detach().abs() >= 0.99).double().mean().item()
                report["decoder_saturation_definition"] = (
                    "fraction of hidden-sequence elements with abs(h)>=0.99; a hidden-state proxy, not GRU gate probabilities")
                if not torch.isfinite(objective).item():
                    report["gradient_probe"] = "not run: nonfinite objective"
                    return report
                parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
                derivatives = torch.autograd.grad(objective, [parameter for _, parameter in parameters] +
                                                  [output.z, output.logits], allow_unused=True)
                report["gradient_blocks"] = _gradient_blocks(parameters, derivatives[:-2])
                report["latent_gradient"] = tensor_summary(derivatives[-2])
                report["logit_gradient"] = tensor_summary(derivatives[-1])
                global_norm = math.hypot(*[block["l2"] for block in report["gradient_blocks"].values()])
                report["global_parameter_grad_l2"] = (global_norm if math.isfinite(global_norm) and not any(
                    block["nonfinite_elements"] for block in report["gradient_blocks"].values()) else None)
                # CE sum over positions and mean over batch has dL/dlogit =
                # (softmax(logits)-one_hot(x))/B, so every entry is bounded by 1/B.
                report["ce_logit_gradient_max_bound"] = 1.0 / len(x)
                handle.remove()  # Do not retain the extra decoder graph via the hook.
                if raw_handle is not None:
                    raw_handle.remove()
                report["input_dependence"] = _input_dependence(model, output.z.detach(), x, output.logits.detach())
                sensitivity_z = output.z.detach().clone().requires_grad_(True)
                sensitivity_logits = model.decode(sensitivity_z)
                selected = sorted(set(horizons or (1, 20, 40, x.shape[1])))
                selected = [h for h in selected if 1 <= h <= x.shape[1]]
                direction = torch.ones(model.vocab_size, device=device, dtype=sensitivity_logits.dtype)
                direction[1::2] = -1
                direction /= math.sqrt(model.vocab_size)
                sensitivities = {}
                for index, horizon in enumerate(selected):
                    score = (sensitivity_logits[:, horizon - 1, :] * direction).sum() / len(x)
                    derivative = torch.autograd.grad(score, sensitivity_z, retain_graph=index < len(selected) - 1)[0]
                    sensitivities[str(horizon)] = tensor_summary(derivative)
                report["decoder_position_vjp"] = sensitivities
                report["decoder_position_vjp_definition"] = "d(mean_batch(dot(alternating_unit_vector, logits[t])))/dz; not a Jacobian spectral norm"
                return report
    finally:
        handle.remove()
        if raw_handle is not None:
            raw_handle.remove()
        for module, was_training in previous_modes.items():
            module.training = was_training


def compare_checkpoints(paths, pipeline, split="val", batch_size=8, device="cpu", horizons=None):
    if split not in ("train", "val"):
        raise ValueError("Checkpoint audit is restricted to train/val; never test")
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("Audit batch_size must be between 1 and 128")
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    result = {"split": split, "data_contract": pipeline.contract(), "checkpoints": [],
              "backend": "reference GRU, cuDNN disabled during probes", "optimizer_steps": 0,
              "limitations": ["small fixed batch, not full validation or a throughput benchmark",
                              "GRU gate weight spectra do not establish recurrent Jacobian instability",
                              "no active Google Drive file is modified"]}
    devices = [selected_device.index or 0] if selected_device.type == "cuda" else []
    batch = None
    with torch.random.fork_rng(devices=devices):
        for label, raw_path in paths.items():
            path = Path(raw_path)
            entry = {"label": label, "path": str(path.resolve()), "available": path.is_file()}
            result["checkpoints"].append(entry)
            if not entry["available"]:
                continue
            original_hash = _sha256(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            pipeline.validate_checkpoint(checkpoint)
            model_type = {"ae": Autoencoder, "vae": VariationalAutoencoder}.get(checkpoint.get("model_type"))
            if model_type is None:
                raise ValueError(f"Unknown model_type in {path}")
            model = model_type(**checkpoint["model_hparams"]).to(selected_device)
            model.load_state_dict(checkpoint["state_dict"])
            if batch is None:
                reader = pipeline.iter_batches(split, batch_size=batch_size)
                try:
                    batch = torch.tensor(next(reader)["x"], dtype=torch.long, device=selected_device)
                finally:
                    reader.close()
            entry.update(sha256=original_hash, epoch=checkpoint.get("epoch"),
                         best_epoch=checkpoint.get("best_epoch"), model_hparams=model.hparams,
                         training_config=checkpoint.get("training_config"),
                         recurrent_spectra=recurrent_spectra(model),
                         probe=probe_model(model, batch, beta=checkpoint.get("training_config", {}).get("beta", 1.0),
                                           horizons=horizons))
            entry["parameter_summaries"] = {name: tensor_summary(parameter)
                                              for name, parameter in model.named_parameters()}
            optimizer_state = checkpoint.get("optimizer_state_dict", {}).get("state", {})
            entry["optimizer_state_summaries"] = {
                str(parameter): {name: tensor_summary(value) for name, value in state.items()
                                 if isinstance(value, torch.Tensor)}
                for parameter, state in optimizer_state.items()}
            entry["optimizer_nonfinite_tensors"] = [
                {"parameter_id": str(parameter), "state": name, "summary": tensor_summary(value)}
                for parameter, state in optimizer_state.items() for name, value in state.items()
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item()]
            if _sha256(path) != original_hash:
                raise RuntimeError(f"Checkpoint changed while being read: {path}")
    if len([entry for entry in result["checkpoints"] if entry["available"]]) < 2:
        result["limitations"].append("fewer than two checkpoints available; no best/last causal comparison established")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best", type=Path)
    parser.add_argument("--last", type=Path)
    parser.add_argument("--data-config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.best is None and args.last is None:
        parser.error("Provide --best and/or --last checkpoint paths")
    if args.output.exists():
        raise FileExistsError(f"Audit output already exists: {args.output}")
    config = load_config(args.data_config)
    if args.processed_dir is not None:
        config["processed_dir"] = args.processed_dir.resolve()
    result = compare_checkpoints({name: path for name, path in (("best", args.best), ("last", args.last))
                                  if path is not None}, DataPipeline(config), args.split, args.batch_size, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
