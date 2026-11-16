"""Read-only latent information diagnostics; categorical metrics accumulate in FP64.

The marginal baseline is fitted on x positions from train only (never next-token
y). Evaluation holds targets and positions fixed while replacing decoder z. A
small response of raw logits is not an estimate of mutual information: logit
offsets, scale, the latent population and target alignment all matter.
"""
from __future__ import annotations

import hashlib
import math

import torch
from torch.nn import functional as F


def _tokens(x, vocab_size):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.long)
    if x.dtype != torch.long or x.ndim != 2 or not all(x.shape):
        raise ValueError("x must be a nonempty torch.long (B,T) tensor")
    if ((x < 0) | (x >= vocab_size)).any().item():
        raise ValueError("token outside vocabulary support")
    return x


def _finite(x, name):
    if not torch.isfinite(x).all().item():
        raise FloatingPointError(f"nonfinite {name}; diagnostic stopped")
    return x


def _json_finite(value):
    """Refuse silently serialized NaN/Inf (unsmoothed undefined losses use None)."""
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("nonfinite diagnostic result")
    if isinstance(value, dict):
        for child in value.values():
            _json_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _json_finite(child)
    return value


def _regions(length):
    # torch.tensor_split gives nonoverlapping, exhaustive thirds for any T.
    chunks = torch.tensor_split(torch.arange(length), 3)
    return {name: chunk.tolist() for name, chunk in zip(("initial", "middle", "final"), chunks)}


def fit_character_baseline(pipeline, *, batch_size=4096, alpha=0.5, max_examples=None):
    """Stream train x only; default covers the entire split with O(T*V) memory.

    Counts weight every reconstruction position, including overlapping windows
    and UNK. This matches the AE's target weighting, not unique raw characters.
    max_examples is an explicit prefix approximation and recorded as such.
    alpha=0 is allowed; impossible held-out targets then yield null loss plus a
    count rather than invalid JSON Infinity. alpha>0 is the primary baseline.
    """
    if isinstance(alpha, bool) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if max_examples is not None and (type(max_examples) is not int or max_examples <= 0):
        raise ValueError("max_examples must be a positive integer or None")
    vocab_size = len(pipeline.vocab["itos"])
    length = pipeline.config["window_size"]
    counts = torch.zeros(vocab_size, dtype=torch.int64)
    position_counts = torch.zeros(length * vocab_size, dtype=torch.int64)
    total = 0
    digest = hashlib.sha256()
    iterator = pipeline.iter_tensor_batches("train", batch_size=batch_size)
    try:
        for batch in iterator:
            x = _tokens(batch["x"], vocab_size).detach().cpu()
            if x.shape[1] != length:
                raise ValueError("train sequence length mismatch")
            if max_examples is not None:
                x = x[:max_examples - total]
            if not len(x):
                break
            counts += torch.bincount(x.flatten(), minlength=vocab_size)
            position_ids = torch.arange(length).unsqueeze(0) * vocab_size + x
            position_counts += torch.bincount(position_ids.flatten(), minlength=length * vocab_size)
            digest.update(x.contiguous().numpy().astype("<i8", copy=False).tobytes())
            total += len(x)
            if max_examples is not None and total >= max_examples:
                break
    finally:
        if hasattr(iterator, "close"):
            iterator.close()
    if not total:
        raise ValueError("empty train split")
    probabilities = (counts.double() + alpha) / (counts.sum().double() + alpha * vocab_size)
    position_counts = position_counts.reshape(length, vocab_size)
    positional = (position_counts.double() + alpha) / (total + alpha * vocab_size)
    result = {
        "fitted_split": "train", "targets": "x, all positions including UNK; no y",
        "alpha": float(alpha), "smoothing": "additive alpha per vocabulary category",
        "vocab_size": vocab_size, "sequence_length": length,
        "train_examples": total, "train_tokens": int(counts.sum()),
        "scope": "full_train" if max_examples is None else "explicit_train_prefix",
        "max_examples": max_examples, "train_x_sha256": digest.hexdigest(),
        "counts": counts.tolist(), "probabilities": probabilities.tolist(),
        "position_counts": position_counts.tolist(), "position_probabilities": positional.tolist(),
        "weighting_limit": "counts reconstruction windows; overlapping raw characters may be counted repeatedly",
    }
    return _json_finite(result)


def _nll_summary(per_token):
    """Allow +Inf only as an explicit unsupported-target baseline outcome."""
    unsupported = torch.isinf(per_token)
    if torch.isnan(per_token).any().item() or (per_token < 0).any().item():
        raise FloatingPointError("invalid token NLL")
    def average(values):
        return values.mean().item() if values.numel() and torch.isfinite(values).all().item() else None
    per_sequence = per_token.sum(-1)
    length = per_token.shape[1]
    result = {
        "nats_per_sequence": average(per_sequence), "nats_per_character": average(per_token),
        "unsupported_target_tokens": int(unsupported.sum().item()),
        "status": "undefined_unsmoothed_support" if unsupported.any().item() else "finite",
        "per_sequence_nats": [value if math.isfinite(value) else None for value in per_sequence.tolist()],
        "per_position_nats_per_character": [average(per_token[:, position]) for position in range(length)],
        "regions": {name: {"positions_0based": indices,
                            "nats_per_character": average(per_token[:, indices]),
                            "nats_per_sequence_region": average(per_token[:, indices].sum(-1)) if indices else None}
                    for name, indices in _regions(length).items()},
    }
    return result


def evaluate_baseline(baseline, x):
    """Evaluate global marginal and train-position marginals on identical x."""
    x = _tokens(x, baseline["vocab_size"])
    if x.shape[1] != baseline["sequence_length"]:
        raise ValueError("baseline sequence length mismatch")
    probabilities = torch.as_tensor(baseline["probabilities"], dtype=torch.float64, device=x.device)
    _finite(probabilities, "baseline probabilities")
    if (probabilities < 0).any().item() or not torch.isclose(probabilities.sum(), probabilities.new_tensor(1.0)).item():
        raise ValueError("invalid baseline probabilities")
    marginal = _nll_summary(-probabilities.log()[x])
    result = {"marginal": marginal, "fitted_split": baseline["fitted_split"],
              "fit_scope": baseline["scope"], "alpha": baseline["alpha"],
              "examples": len(x), "positions": x.shape[1]}
    if "position_probabilities" in baseline:
        positional = torch.as_tensor(baseline["position_probabilities"], dtype=torch.float64, device=x.device)
        _finite(positional, "position baseline probabilities")
        if positional.shape != (x.shape[1], len(probabilities)) or (positional < 0).any().item():
            raise ValueError("invalid position baseline probabilities")
        if not torch.allclose(positional.sum(-1), torch.ones(x.shape[1], device=x.device, dtype=torch.float64)):
            raise ValueError("position probabilities must sum to one")
        result["position_marginal"] = _nll_summary(-positional.log().unsqueeze(0).expand(len(x), -1, -1).gather(-1, x.unsqueeze(-1)).squeeze(-1))
    return _json_finite(result)


def _logits(logits, targets):
    if logits.ndim != 3 or not all(logits.shape) or logits.shape[-1] < 2:
        raise ValueError("logits must have shape (B,T,V>=2)")
    targets = _tokens(targets, logits.shape[-1])
    if targets.shape != logits.shape[:2] or targets.device != logits.device:
        raise ValueError("targets must match logits shape and device")
    logits = _finite(logits.detach().double(), "logits")
    logp = _finite(F.log_softmax(logits, dim=-1), "log probabilities")
    return logits, logp, -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def summarize_prediction(logits, targets):
    logits, logp, per_token = _logits(logits, targets)
    probabilities = logp.exp()
    predictions = logits.argmax(-1)
    entropy = -(probabilities * logp).sum(-1)
    centered = logits - logits.mean(-1, keepdim=True)
    summary = _nll_summary(per_token)
    summary.update({
        "accuracy": (predictions == targets).double().mean().item(),
        "prediction_entropy_nats_per_character": entropy.mean().item(),
        "top1_token_histogram": torch.bincount(predictions.flatten(), minlength=logits.shape[-1]).cpu().tolist(),
        "raw_logits_rms": logits.square().mean().sqrt().item(),
        "centered_logits_rms": centered.square().mean().sqrt().item(),
        "between_examples": {
            "probability_variance_mean": probabilities.var(0, unbiased=False).mean().item(),
            "centered_logits_std_rms": centered.var(0, unbiased=False).mean().sqrt().item(),
            "top1_disagreement_vs_rolled_example": (predictions != predictions.roll(1, 0)).double().mean().item() if len(logits) > 1 else None,
            "probability_tv_vs_rolled_example": ((probabilities - probabilities.roll(1, 0)).abs().sum(-1) * 0.5).mean().item() if len(logits) > 1 else None,
        },
    })
    summary["per_position_accuracy"] = (predictions == targets).double().mean(0).cpu().tolist()
    summary["per_position_prediction_entropy"] = entropy.mean(0).cpu().tolist()
    if len(logits) > 1:
        position_counts = F.one_hot(predictions, num_classes=logits.shape[-1]).double().sum(0)
        disagreement = (len(logits) ** 2 - position_counts.square().sum(-1)) / (len(logits) * (len(logits) - 1))
        summary["between_examples"]["top1_disagreement_all_distinct_pairs"] = disagreement.mean().item()
    return _json_finite(summary)


def _paired_summary(delta):
    values = delta.detach().double()
    n = values.numel()
    std = values.std(unbiased=True).item() if n > 1 else None
    return {"mean": values.mean().item(), "std": std,
            "standard_error_descriptive": std / math.sqrt(n) if std is not None else None,
            "minimum": values.min().item(), "maximum": values.max().item(),
            "positive_fraction": (values > 0).double().mean().item(),
            "per_example": values.cpu().tolist()}


def compare_logits(original, altered, targets):
    """Positive loss delta means the intervention damaged reconstruction.

    Centering logits removes the per-position constant softmax gauge. TV/JS
    measure predictive distributions; top1 alone misses sub-argmax information.
    """
    original, logp, original_nll = _logits(original, targets)
    altered, logq, altered_nll = _logits(altered, targets)
    if altered.shape != original.shape:
        raise ValueError("comparison logits shapes differ")
    p, q = logp.exp(), logq.exp()
    raw_delta = altered - original
    centered_delta = (altered - altered.mean(-1, keepdim=True)) - (original - original.mean(-1, keepdim=True))
    tv = (p - q).abs().sum(-1) * 0.5
    logm = torch.logaddexp(logp, logq) - math.log(2.0)
    js = 0.5 * ((p * (logp - logm)).sum(-1) + (q * (logq - logm)).sum(-1))
    top1 = (original.argmax(-1) != altered.argmax(-1)).double()
    loss_delta = altered_nll - original_nll
    result = {
        "delta_nats_per_sequence": _paired_summary(loss_delta.sum(-1)),
        "delta_nats_per_character": loss_delta.mean().item(),
        "raw_logits_rms_change": raw_delta.square().mean().sqrt().item(),
        "raw_logits_l2_change": raw_delta.norm().item(),
        "relative_raw_logits_l2_change": (raw_delta.norm() / original.norm().clamp_min(1e-12)).item(),
        "centered_logits_rms_change": centered_delta.square().mean().sqrt().item(),
        "centered_logits_l2_change": centered_delta.norm().item(),
        "relative_centered_logits_l2_change": (centered_delta.norm() / (original - original.mean(-1, keepdim=True)).norm().clamp_min(1e-12)).item(),
        "probability_tv_mean": tv.mean().item(), "probability_js_nats_mean": js.mean().item(),
        "top1_change_fraction": top1.mean().item(),
        "per_position": {
            "delta_nats_per_character": loss_delta.mean(0).cpu().tolist(),
            "raw_logits_rms_change": raw_delta.square().mean((0, 2)).sqrt().cpu().tolist(),
            "centered_logits_rms_change": centered_delta.square().mean((0, 2)).sqrt().cpu().tolist(),
            "probability_tv": tv.mean(0).cpu().tolist(),
            "probability_js_nats": js.mean(0).cpu().tolist(),
            "top1_change_fraction": top1.mean(0).cpu().tolist(),
        },
        "regions": {},
    }
    for name, indices in _regions(original.shape[1]).items():
        result["regions"][name] = {
            "positions_0based": indices,
            "delta_nats_per_character": loss_delta[:, indices].mean().item() if indices else None,
            "probability_tv_mean": tv[:, indices].mean().item() if indices else None,
            "top1_change_fraction": top1[:, indices].mean().item() if indices else None,
        }
    return _json_finite(result)


@torch.no_grad()
def evaluate_latent_dependence(model, x, baseline, *, permutation=None):
    """Evaluate one fixed population globally, without parameter/gradient writes.

    Default swap shifts half the population, a derangement including odd B.
    This reduces adjacency in ordered windows but does not guarantee semantic
    dissimilarity. A supplied integer
    permutation must be a bijection without fixed points. Reverse-order legacy
    sensitivity is also reported exactly; odd B leaves its middle row unchanged.
    z=0 may be off the encoder manifold and is a secondary intervention. Loss
    changes under an on-population swap are the stronger alignment diagnostic.
    """
    x = _tokens(x, baseline["vocab_size"])
    if len(x) < 2:
        raise ValueError("at least two examples are required for an informative swap")
    permutation_source = "half_population_cyclic_shift" if permutation is None else "explicit"
    if permutation is None:
        permutation = torch.arange(len(x), device=x.device).roll(len(x) // 2)
    else:
        permutation = torch.as_tensor(permutation, device=x.device)
    if permutation.dtype != torch.long or permutation.shape != (len(x),):
        raise ValueError("permutation must contain B integer indices")
    indices = torch.arange(len(x), device=x.device)
    if not torch.equal(permutation.sort().values, indices) or (permutation == indices).any().item():
        raise ValueError("permutation must be a bijection without fixed points")
    modes_before = {module: module.training for module in model.modules()}
    devices = [x.device.index if x.device.index is not None else torch.cuda.current_device()] if x.device.type == "cuda" else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            z = model.encode(x)
            if not isinstance(z, torch.Tensor) or z.ndim != 2 or len(z) != len(x):
                raise TypeError("diagnostic requires deterministic AE encode(x) -> (B,Z)")
            _finite(z, "latent")
            original = model.decode(z)
            swapped = model.decode(z[permutation])
            zero = model.decode(torch.zeros_like(z))
            reverse = model.decode(z.flip(0))
            baseline_result = evaluate_baseline(baseline, x)
            original_summary = summarize_prediction(original, x)
            swapped_summary = summarize_prediction(swapped, x)
            zero_summary = summarize_prediction(zero, x)
            comparisons = {"swapped_minus_original": compare_logits(original, swapped, x),
                           "zero_minus_original": compare_logits(original, zero, x)}
            raw_reverse = _finite(reverse.detach().double(), "reverse logits")
            raw_original = original.detach().double()
            historical = ((raw_reverse - raw_original).norm() / raw_original.norm().clamp_min(1e-12)).item()
            z64 = z.detach().double()
            center = z64 - z64.mean(0, keepdim=True)
            covariance = center.T @ center / len(z)
            eig = torch.linalg.eigvalsh(covariance).clamp_min(0)
            variance = eig.sum()
            effective_rank = (variance.square() / eig.square().sum()).item() if variance.item() > 0 else 0.0
            result = {
                "schema_version": 1, "examples": len(x), "sequence_length": x.shape[1],
                "metrics_dtype": "float64", "targets": "same x, same positions; no y, no PAD mask",
                "permutation": permutation.cpu().tolist(), "permutation_fixed_points": 0,
                "permutation_source": permutation_source,
                "swapped_source_target_token_agreement": (x == x[permutation]).double().mean().item(),
                "swapped_source_target_exact_sequence_agreement": (x == x[permutation]).all(-1).double().mean().item(),
                "modes": {"original": original_summary, "swapped": swapped_summary, "zero": zero_summary},
                "baseline": baseline_result, "comparisons": comparisons,
                "historical_sensitivity": {
                    "relative_raw_logits_l2_reverse": historical,
                    "definition": "||decode(reverse(z))-decode(z)||_2 / max(||decode(z)||_2,1e-12)",
                    "population": "entire supplied x; historical pilots used first fixed validation batch",
                    "reverse_fixed_points": len(x) % 2,
                    "limitation": "depends on raw logit offsets/scale and batch latent spread; not mutual information or reconstruction utility",
                },
                "latent": {"max_abs": z64.abs().max().item(), "rms": z64.square().mean().sqrt().item(),
                           "between_example_std_rms": center.square().mean().sqrt().item(),
                           "mean_pairwise_rolled_distance": (z64 - z64.roll(1, 0)).norm(dim=-1).mean().item(),
                           "mean_swapped_pair_distance": (z64 - z64[permutation]).norm(dim=-1).mean().item(),
                           "covariance_effective_rank": effective_rank},
                "interpretation_limits": [
                    "Paired deltas are descriptive on this fixed subset, not independent-sample confidence intervals.",
                    "Small swap damage plus small TV supports little useful sequence-specific dependence on this population; does not estimate global mutual information.",
                    "Zero latent may be off-manifold; low top1 diversity alone cannot establish collapse.",
                    "Position marginals can explain position-only behavior without sequence information.",
                ],
            }
            for baseline_name in ("marginal", "position_marginal"):
                if baseline_name not in baseline_result:
                    continue
                baseline_loss = baseline_result[baseline_name]["nats_per_sequence"]
                result.setdefault("baseline_comparison", {})[baseline_name] = {
                    "ae_minus_baseline_nats_per_sequence": original_summary["nats_per_sequence"] - baseline_loss if baseline_loss is not None else None,
                    "ae_minus_baseline_nats_per_character": (original_summary["nats_per_sequence"] - baseline_loss) / x.shape[1] if baseline_loss is not None else None,
                    "per_position_ae_minus_baseline_nats": [ae - base if base is not None else None for ae, base in zip(original_summary["per_position_nats_per_character"], baseline_result[baseline_name]["per_position_nats_per_character"])],
                }
                # Loss proximity alone does not show a marginal predictor:
                # compare its actual distribution with the fitted prior too.
                key = "probabilities" if baseline_name == "marginal" else "position_probabilities"
                prior = torch.as_tensor(baseline[key], dtype=torch.float64, device=x.device)
                if (prior > 0).all().item():
                    prior_logits = prior.log().expand_as(original)
                    comparison = compare_logits(original, prior_logits, x)
                    result["baseline_comparison"][baseline_name]["distribution_distance"] = {
                        metric: comparison[metric] for metric in
                        ("probability_tv_mean", "probability_js_nats_mean", "top1_change_fraction",
                         "centered_logits_rms_change", "per_position")}
                else:
                    result["baseline_comparison"][baseline_name]["distribution_distance"] = None
            return _json_finite(result)
    finally:
        for module, training in modes_before.items():
            module.training = training
