"""Autoencoder (AE) and Variational Autoencoder (VAE) for 80-character text windows.

Semantic compression: each window of 80 characters is compressed into a small
latent vector z and then reconstructed from it.

    x (batch, 80) character ids
    -> Embedding(76, 16), flatten          (batch, 1280)
    -> Linear 1280 -> 512, ReLU
    -> Linear 512 -> 256, ReLU
    -> AE:  Linear 256 -> z                (batch, z)
       VAE: Linear 256 -> mu, Linear 256 -> log_var, z = mu + sigma * eps
    -> Linear z -> 256, ReLU -> Linear 256 -> 512, ReLU -> Linear 512 -> 80 * 76
    -> (batch, 80, 76) logits, one distribution over the 76 characters per position

The windows always have 80 characters, so we treat them as fixed-size inputs and
use fully-connected layers (simple and fast). The output is categorical, so the
reconstruction loss is the cross-entropy of every position.

This file also has the data loading and the Workshop 2 Transformer (used frozen
for the comparison of representations).
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

SEED = 42
WINDOW = 80
ROOT = Path(__file__).resolve().parents[1]  # Workshop_4/
WORKSHOP1 = ROOT.parent / "Workshop_1"
W2_CHECKPOINT = ROOT.parent / "Workshop_2" / "checkpoints" / "model_best.pt"
CHECKPOINTS_DIR = ROOT / "checkpoints"
RUNS_DIR = ROOT / "runs" / "workshop4"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# fixed random subsamples of the Workshop 1 splits
N_SAMPLES = {"train": 200_000, "val": 20_000, "test": 20_000}


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_vocab():
    return json.loads((WORKSHOP1 / "data" / "vocab.json").read_text(encoding="utf-8"))


def load_split(split, vocab):
    """Fixed subsample of a Workshop 1 split: x (n, 80) character ids and y (n,) next character."""
    df = pd.read_parquet(WORKSHOP1 / "data" / "processed" / f"{split}.parquet", columns=["x", "y_id"])
    idx = np.sort(np.random.default_rng(SEED).choice(len(df), N_SAMPLES[split], replace=False))
    df = df.iloc[idx]
    stoi = vocab["stoi"]
    x = np.array([[stoi.get(c, 0) for c in w] for w in df["x"]], dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(df["y_id"].to_numpy().astype(np.int64))


def decode(ids, itos):
    """Character ids -> text (<UNK> shown as '?')."""
    return "".join("?" if i == 0 else itos[i] for i in ids)


# ---------------------------------------------------------------- models
class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size, z_dim=64, variational=False, emb_dim=16):
        super().__init__()
        self.hparams = {"vocab_size": vocab_size, "z_dim": z_dim, "variational": variational,
                        "emb_dim": emb_dim}
        self.vocab_size = vocab_size
        self.variational = variational
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.Sequential(
            nn.Linear(WINDOW * emb_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.to_z = nn.Linear(256, z_dim)  # AE: z, VAE: mu
        if variational:
            self.to_log_var = nn.Linear(256, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, WINDOW * vocab_size),
        )

    def encode(self, x):
        """AE: (z, None). VAE: (mu, log_var)."""
        h = self.encoder(self.embedding(x).flatten(1))
        if self.variational:
            return self.to_z(h), self.to_log_var(h)
        return self.to_z(h), None

    def decode(self, z):
        return self.decoder(z).view(-1, WINDOW, self.vocab_size)

    def forward(self, x):
        mu, log_var = self.encode(x)
        if self.variational and self.training:
            z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)  # reparameterization trick
        else:
            z = mu  # evaluation: use the mean
        return self.decode(z), mu, log_var


def kl_divergence(mu, log_var):
    """KL( N(mu, sigma^2) || N(0, I) ) for every example: -1/2 * sum(1 + log s^2 - mu^2 - s^2)."""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)


def reconstruction_loss(logits, x):
    """Cross-entropy summed over the 80 positions, for every example."""
    ce = nn.functional.cross_entropy(logits.transpose(1, 2), x, reduction="none")  # (batch, 80)
    return ce.sum(dim=1)


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TextAutoencoder(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


# ---------------------------------------------------------------- Workshop 2 model (frozen)
class CharTransformer(nn.Module):
    """Same architecture as Workshop 2 (inference only)."""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, dropout=0.1, max_len=80):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        block = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, n_layers, norm=nn.LayerNorm(d_model),
                                            enable_nested_tensor=False)
        self.fc = nn.Linear(d_model, vocab_size)

    def features(self, x):
        """Representation of the window: last position after the Transformer blocks (128 values)."""
        positions = torch.arange(x.size(1), device=x.device)
        h = self.token_embedding(x) + self.position_embedding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        return self.blocks(h, mask=mask, is_causal=True)[:, -1]

    def forward(self, x):
        return self.fc(self.features(x))


def load_w2_transformer(device):
    ckpt = torch.load(W2_CHECKPOINT, map_location="cpu", weights_only=False)
    model = CharTransformer(**ckpt["model_hparams"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def batched(fn, x, device, batch_size=2048):
    """Apply fn to x in batches and concatenate the results on the CPU."""
    return torch.cat([fn(x[i:i + batch_size].to(device)).cpu() for i in range(0, len(x), batch_size)])
