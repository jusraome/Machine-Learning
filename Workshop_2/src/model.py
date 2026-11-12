"""Character-level LSTM for next-character prediction.

Architecture (same as the LEAF reference model for Shakespeare):

    x (batch, 80) character ids
    -> Embedding(76, 8)          -> (batch, 80, 8)
    -> LSTM(8 -> 256)            -> (batch, 80, 256)
    -> Dropout
    -> LSTM(256 -> 256)          -> (batch, 80, 256)
    -> last hidden state         -> (batch, 256)
    -> Dropout
    -> Linear(256 -> 76)         -> (batch, 76) logits (softmax is inside the loss)

Why an LSTM: text is a sequence, and the next character depends on the
previous ones. The LSTM keeps a hidden state while it reads the 80 characters,
and its gates let it remember information for many steps.

Alternatives we considered:
- Simple RNN: the gradient vanishes over 80 steps, so it forgets quickly.
- GRU: very similar to the LSTM; we kept the LSTM because it is the reference
  model of the benchmark, so our results can be compared with LEAF.
- Transformer: too expensive for our hardware (GTX 1650 with 4 GB).
- 1D CNN: only sees as much context as its kernel size allows.

No transfer learning: there is no pre-trained model for this 76-character
vocabulary, so the model is trained from scratch.

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


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=8, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.hparams = {"vocab_size": vocab_size, "embedding_dim": embedding_dim,
                        "hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout}
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Two single-layer LSTMs with dropout in between. This is the same as
        # nn.LSTM(num_layers=2, dropout=p), but that version crashed on our
        # Windows + CUDA setup at the end of training.
        self.lstm = nn.ModuleList(
            [nn.LSTM(embedding_dim if i == 0 else hidden_size, hidden_size, batch_first=True)
             for i in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        """Xavier for input weights, orthogonal for recurrent weights,
        forget-gate bias = 1 so the LSTM remembers by default at the start."""
        h = self.hparams["hidden_size"]
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                for gate in param.data.chunk(4, dim=0):  # 4 gates stacked
                    nn.init.orthogonal_(gate)
            elif "bias" in name:
                nn.init.zeros_(param)
                param.data[h:2 * h] = 0.5  # forget gate (bias_ih + bias_hh = 1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out = self.embedding(x)
        for i, layer in enumerate(self.lstm):
            if i > 0:
                out = self.dropout(out)
            out, (h_n, _) = layer(out)
        return self.fc(self.dropout(h_n[-1]))  # last hidden state of the top layer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    set_seeds()
    vocab_size = read_vocab()["size"]
    model = CharLSTM(vocab_size)
    print(model)
    print("\nParameters per layer:")
    for name, module in [("embedding", model.embedding), ("lstm", model.lstm), ("fc", model.fc)]:
        print(f"  {name:10s} {count_parameters(module):>9,}")
    print(f"  {'total':10s} {count_parameters(model):>9,} (all trainable)")

    # Forward pass on a random batch to check the shapes.
    batch = torch.randint(0, vocab_size, (4, WINDOW_SIZE))
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    print(f"\nInput shape:  {tuple(batch.shape)}")
    print(f"Output shape: {tuple(logits.shape)}  (batch, vocab_size)")


if __name__ == "__main__":
    main()
