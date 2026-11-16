"""Recurrent character AE/VAE; no data loading, training or file writes.

Inputs are token IDs (B, T); outputs are categorical logits (B, T, V).
The decoder receives only z, never the original tokens or next-character y.
Reconstruction NLL sums over T and KL sums over z before averaging the batch,
so beta weights the two terms of the per-sequence negative ELBO consistently.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class AEOutput:
    logits: Tensor
    z: Tensor


@dataclass
class VAEOutput(AEOutput):
    mu: Tensor
    logvar: Tensor


@dataclass
class VAELoss:
    total: Tensor
    reconstruction: Tensor
    kl: Tensor


def _reduce(values: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction must be 'none', 'mean' or 'sum'")


def _check_posterior(mu: Tensor, logvar: Tensor) -> None:
    if mu.ndim != 2 or mu.shape != logvar.shape or not all(mu.shape):
        raise ValueError("mu and logvar must have the same nonempty (B, latent_dim) shape")
    if not mu.is_floating_point() or not logvar.is_floating_point():
        raise TypeError("mu and logvar must be floating-point tensors")
    if mu.device != logvar.device or mu.dtype != logvar.dtype:
        raise ValueError("mu and logvar must share device and dtype")


def reparameterize(mu: Tensor, logvar: Tensor, eps: Tensor | None = None) -> Tensor:
    """Sample z = mu + exp(0.5 * logvar) * eps with differentiable parameters.

    eps is optional for deterministic tests; normally it is sampled from N(0,I).
    No clipping is applied: nonfinite losses must be diagnosed during training.
    """
    _check_posterior(mu, logvar)
    if eps is None:
        eps = torch.randn_like(mu)
    elif eps.shape != mu.shape or eps.device != mu.device or eps.dtype != mu.dtype:
        raise ValueError("eps must match mu's shape, device and dtype")
    return mu + torch.exp(0.5 * logvar) * eps


def kl_divergence(mu: Tensor, logvar: Tensor, reduction: str = "mean") -> Tensor:
    """Analytic KL(q(z|x) || N(0,I)), summed over latent coordinates."""
    _check_posterior(mu, logvar)
    per_sequence = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(dim=-1)
    return _reduce(per_sequence, reduction)


def reconstruction_loss(logits: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    """Categorical NLL of x (not y), summed across each character sequence.

    The default is nats/sequence; divide by T to log nats/character. UNK (0)
    remains a genuine category, not padding. MSE on ordinal token IDs is invalid;
    later evaluation can use probabilities versus one-hot targets for MSE.
    """
    if logits.ndim != 3 or not all(logits.shape) or logits.shape[-1] < 2:
        raise ValueError("logits must have nonempty shape (B, T, vocab_size >= 2)")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must have shape (B, T) matching logits")
    if targets.dtype != torch.long:
        raise TypeError("targets must be torch.long token IDs")
    if ((targets < 0) | (targets >= logits.shape[-1])).any().item():
        raise ValueError("targets must contain only IDs in [0, vocab_size)")
    per_token = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return _reduce(per_token.sum(dim=-1), reduction)


def vae_loss(logits: Tensor, targets: Tensor, mu: Tensor, logvar: Tensor,
             beta: float = 1.0, reduction: str = "mean") -> VAELoss:
    """Negative beta-ELBO: categorical reconstruction NLL + beta * KL.

    beta=0.5, 1.0 and 4.0 support the required future runs; beta=0 supports
    optional KL annealing. With reduction='none', each term has shape (B,).
    """
    if isinstance(beta, bool) or not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    _check_posterior(mu, logvar)
    if logits.ndim != 3 or logits.shape[0] != mu.shape[0]:
        raise ValueError("posterior and reconstruction batch sizes must match")
    recon = reconstruction_loss(logits, targets, reduction)
    kl = kl_divergence(mu, logvar, reduction)
    return VAELoss(total=recon + beta * kl, reconstruction=recon, kl=kl)


class _SequenceAutoencoder(nn.Module):
    """Shared recurrent encoder and decoder; all dimensions are checkpointable."""

    def __init__(self, vocab_size: int = 76, sequence_length: int = 80,
                 embedding_dim: int = 32, hidden_size: int = 128,
                 latent_dim: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        dimensions = dict(vocab_size=vocab_size, sequence_length=sequence_length,
                          embedding_dim=embedding_dim, hidden_size=hidden_size,
                          latent_dim=latent_dim, num_layers=num_layers)
        if any(type(value) is not int or value <= 0 for value in dimensions.values()):
            raise ValueError("all architecture dimensions must be positive integers")
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if isinstance(dropout, bool) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.hparams = {**dimensions, "dropout": dropout}
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # GRU dropout is inter-layer only; one-layer models do not use it.
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers,
                              batch_first=True, dropout=recurrent_dropout)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size * num_layers)
        self.decoder = nn.GRU(latent_dim, hidden_size, num_layers=num_layers,
                              batch_first=True, dropout=recurrent_dropout)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def _encode_hidden(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.sequence_length:
            raise ValueError(f"x must have shape (B > 0, {self.sequence_length})")
        if x.dtype != torch.long:
            raise TypeError("x must be torch.long token IDs")
        _, hidden = self.encoder(self.embedding(x))
        return hidden[-1]

    def decode(self, z: Tensor) -> Tensor:
        """Reconstruct from z alone; recurrent state determines sequence position."""
        if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] != self.latent_dim:
            raise ValueError(f"z must have shape (B > 0, {self.latent_dim})")
        initial = torch.tanh(self.latent_to_hidden(z))
        initial = initial.reshape(z.shape[0], self.num_layers, self.hidden_size)
        initial = initial.transpose(0, 1).contiguous()
        repeated_z = z.unsqueeze(1).expand(-1, self.sequence_length, -1)
        decoded, _ = self.decoder(repeated_z, initial)
        return self.output_projection(decoded)


class Autoencoder(_SequenceAutoencoder):
    """Embedding -> GRU -> deterministic bottleneck -> GRU -> token logits.

    ``latent_activation='tanh'`` is an explicit experimental architecture
    variant. It bounds the projected code before both decoder routes; identity
    preserves the historical model and checkpoint hyperparameter envelope.
    This does not establish stability of either architecture over a long run.
    """

    def __init__(self, latent_activation: str = "identity", **kwargs):
        if latent_activation not in ("identity", "tanh"):
            raise ValueError("latent_activation must be 'identity' or 'tanh'")
        super().__init__(**kwargs)
        self.to_latent = nn.Linear(self.hidden_size, self.latent_dim)
        self.latent_activation = latent_activation
        if latent_activation != "identity":
            self.hparams["latent_activation"] = latent_activation

    def encode_raw(self, x: Tensor) -> Tensor:
        """Uncontrolled projection, exposed for magnitude diagnostics."""
        return self.to_latent(self._encode_hidden(x))

    def encode(self, x: Tensor) -> Tensor:
        raw = self.encode_raw(x)
        return raw.tanh() if self.latent_activation == "tanh" else raw

    def forward(self, x: Tensor) -> AEOutput:
        z = self.encode(x)
        return AEOutput(logits=self.decode(z), z=z)


class VariationalAutoencoder(_SequenceAutoencoder):
    """Diagonal Gaussian posterior; evaluation uses mu unless sample=True."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mu_head = nn.Linear(self.hidden_size, self.latent_dim)
        self.logvar_head = nn.Linear(self.hidden_size, self.latent_dim)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self._encode_hidden(x)
        return self.mu_head(hidden), self.logvar_head(hidden)

    reparameterize = staticmethod(reparameterize)

    def forward(self, x: Tensor, sample: bool | None = None) -> VAEOutput:
        if sample is not None and not isinstance(sample, bool):
            raise TypeError("sample must be bool or None")
        mu, logvar = self.encode(x)
        should_sample = self.training if sample is None else sample
        z = reparameterize(mu, logvar) if should_sample else mu
        return VAEOutput(logits=self.decode(z), z=z, mu=mu, logvar=logvar)
