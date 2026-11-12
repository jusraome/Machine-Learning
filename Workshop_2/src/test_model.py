"""Load checkpoints/model_best.pt and predict on new data.

    python -m src.test_model
    python -m src.test_model --text "To be, or not to be, that is the questio"

Shows the top-5 predicted next characters for a few test windows and for a
hand-written text. It only needs the checkpoint (weights, hyperparameters and
vocabulary are saved inside it), not train.py.
"""

import argparse

import numpy as np
import torch

from src.model import (BEST_CHECKPOINT, SEED, WINDOW_SIZE, CharLSTM, encode_text, get_device,
                       load_split, set_seeds)

DEFAULT_TEXT = "Shall I compare thee to a summer's day? Thou art more lovely and more temperate. Rough wi"


def load_checkpoint(path=BEST_CHECKPOINT):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CharLSTM(**ckpt["model_hparams"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(get_device()).eval(), ckpt


@torch.no_grad()
def top5(model, text, vocab):
    """Top-5 next characters (with probabilities) after the last 80 characters of text."""
    ids = torch.tensor([encode_text(text[-WINDOW_SIZE:], vocab["stoi"])], device=get_device())
    probs = model(ids).softmax(-1)[0].cpu()
    values, indices = probs.topk(5)
    return [(vocab["itos"][i], p.item()) for p, i in zip(values, indices)]


def print_prediction(text, preds, true=None):
    print(f"\ncontext: ...{text[-50:]!r}")
    if true is not None:
        print(f"true next char: {true!r}")
    for rank, (char, p) in enumerate(preds, 1):
        print(f"  {rank}. {char!r:8} {p:6.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    args = parser.parse_args()

    set_seeds()
    model, ckpt = load_checkpoint()
    vocab = ckpt["vocab"]
    print(f"loaded model_best.pt: {ckpt['run_name']}, epoch {ckpt['epoch']}, "
          f"val macro-F1 {ckpt['val_metric']['value']:.4f}")

    print("\n=== 5 random test windows ===")
    test = load_split("test")
    rng = np.random.default_rng(SEED)
    for i in rng.choice(len(test), size=5, replace=False):
        x, y = test["x"].iat[i], test["y"].iat[i]
        print_prediction(x, top5(model, x, vocab), true=y)

    print("\n=== hand-written text ===")
    print_prediction(args.text, top5(model, args.text, vocab))


if __name__ == "__main__":
    main()
