"""Character-level Transformer language model for next-character prediction.

Architecture (a small decoder-only Transformer, like GPT, at character level):

    x (batch, 80) character ids
    -> token embedding (76 -> 128) + position embedding (80 -> 128)   (batch, 80, 128)
    -> 2 Transformer blocks: causal self-attention (4 heads) + feed-forward (128 -> 512 -> 128),
       each with LayerNorm, residual connections and dropout             (batch, 80, 128)
    -> LayerNorm, take the LAST position                                (batch, 128)
    -> Linear(128 -> 76)                                                (batch, 76) logits

Why a Transformer: our team's assignment for Workshop 2 is a Transformer-based
language model. Self-attention lets every position look directly at all the
previous characters of the window (no information has to pass step by step
like in an RNN), and all positions are processed in parallel, so it trains fast
on a GPU. The causal mask makes each position see only the characters before it,
as in a language model.

Alternatives we considered:
- LSTM / GRU: read the text one character at a time; slower to train and the
  information from far characters must pass through many steps.
- 1D CNN: only sees as much context as its kernel size allows.
- Pre-trained Transformers (GPT-2, BERT): they use sub-word tokens, not our
  76-character vocabulary, and are too large for our GPU (GTX 1650, 4 GB),
  so the model is trained from scratch (no transfer learning).

Run `python -m src.model` to print the architecture and test a forward pass.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

SEED = 42
WINDOW_SIZE = 80  # characters in every input window
UNK_INDEX = 0  # index of <UNK> in the vocabulary

ROOT = Path(__file__).resolve().parents[1]  # Workshop_2/
WORKSHOP1 = ROOT.parent / "Workshop_1"  # data, split and vocabulary come from Workshop 1
BEST_CHECKPOINT = ROOT / "checkpoints" / "model_best.pt"


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(name):
    """Load the train / val / test split saved by Workshop 1 (we never re-split)."""
    return pd.read_parquet(WORKSHOP1 / "data" / "processed" / f"{name}.parquet")


def read_vocab():
    """Load the Workshop 1 vocabulary (built on the train split only)."""
    return json.loads((WORKSHOP1 / "data" / "vocab.json").read_text(encoding="utf-8"))


def encode_text(text, stoi):
    """Characters -> ids. Characters not in the vocabulary become <UNK> (0)."""
    return [stoi.get(c, UNK_INDEX) for c in text]


class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, dropout=0.1,
                 max_len=WINDOW_SIZE):
        super().__init__()
        self.hparams = {"vocab_size": vocab_size, "d_model": d_model, "n_heads": n_heads,
                        "n_layers": n_layers, "dropout": dropout, "max_len": max_len}
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        block = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, n_layers, norm=nn.LayerNorm(d_model),
                                            enable_nested_tensor=False)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        h = self.token_embedding(x) + self.position_embedding(positions)
        # causal mask: position i can only attend to positions <= i
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.fc(h[:, -1])  # the last position predicts the next character


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    set_seeds()
    vocab_size = read_vocab()["size"]
    model = CharTransformer(vocab_size)
    print(model)
    print("\nParameters per layer:")
    for name, module in [("token_embedding", model.token_embedding),
                         ("position_embedding", model.position_embedding),
                         ("transformer_blocks", model.blocks), ("fc", model.fc)]:
        print(f"  {name:20s} {count_parameters(module):>9,}")
    print(f"  {'total':20s} {count_parameters(model):>9,} (all trainable)")

    # Forward pass on a random batch to check the shapes.
    batch = torch.randint(0, vocab_size, (4, WINDOW_SIZE))
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    print(f"\nInput shape:  {tuple(batch.shape)}")
    print(f"Output shape: {tuple(logits.shape)}  (batch, vocab_size)")


if __name__ == "__main__":
    main()
