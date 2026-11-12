"""Train the character-level LSTM.

Hyperparameter search (3 learning rates x 2 regularisation settings, short
runs on a subsample of train, evaluated on validation):

    python -m src.train --search --epochs 4 --max-train-samples 300000 --max-val-samples 100000

Final training (the two variants: without and with class weights):

    python -m src.train --run-name lstm_noweights --epochs 10 --max-train-samples 1500000
    python -m src.train --run-name lstm_weighted  --epochs 10 --max-train-samples 1500000 --class-weights

Each run logs loss and metrics per epoch to TensorBoard (runs/<run-name>).
The final runs also save their learning curves in reports/figures/. The best epoch (highest validation macro-F1) is saved to
checkpoints/<run-name>.pt, and copied to checkpoints/model_best.pt if it is
better than the current one. The test split is not used here.
"""

import argparse
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.model import (BEST_CHECKPOINT, ROOT, SEED, UNK_INDEX, WINDOW_SIZE, CharLSTM, get_device,
                       load_split, read_vocab, set_seeds)

PROCESSED_DIR = ROOT / "data" / "processed"  # cache of the encoded data (not in git)
CHECKPOINTS_DIR = ROOT / "checkpoints"
RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PATIENCE = 3  # early stopping: epochs without improvement of val macro-F1
CLIP_NORM = 5.0  # gradient clipping
AUC_VAL_SAMPLES = 50_000  # per-epoch AUC on a fixed val subsample (full AUC is slow)
TRAIN_EVAL_SAMPLES = 200_000  # final train metrics on a fixed subsample

# Grid for --search
SEARCH_LRS = [3e-3, 1e-3, 3e-4]
SEARCH_REG = [{"dropout": 0.1, "weight_decay": 1e-4},   # light regularisation
              {"dropout": 0.3, "weight_decay": 1e-2}]   # strong regularisation


# ---------------------------------------------------------------- data
def load_encoded(split, vocab):
    """Return (x, y) as uint8 tensors. Encoded once and cached in data/processed/."""
    path = PROCESSED_DIR / f"{split}.pt"
    if path.exists():
        data = torch.load(path)
        return data["x"], data["y"]

    print(f"encoding the {split} split...")
    df = load_split(split)
    # Lookup table byte -> id (the text is ASCII); unknown characters -> <UNK>.
    table = np.full(256, UNK_INDEX, dtype=np.uint8)
    for char, idx in vocab["stoi"].items():
        if len(char) == 1 and ord(char) < 256:
            table[ord(char)] = idx
    raw = np.frombuffer("".join(df["x"]).encode("latin-1"), dtype=np.uint8)
    x = torch.from_numpy(table[raw].reshape(-1, WINDOW_SIZE))
    y = torch.from_numpy(df["y_id"].to_numpy().astype(np.uint8))  # y_id comes from Workshop 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y, "itos": vocab["itos"]}, path)
    return x, y


def subsample(n, size):
    """Fixed random subset of row indices (same seed every time), or None for all rows."""
    if size is None or size >= n:
        return None
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(n, size=size, replace=False))


def class_weights(y, vocab_size):
    """Weight of each class = 1 / sqrt(frequency in TRAIN), normalised so the
    average weight per training row is 1. Classes never seen in train get 0."""
    counts = torch.bincount(y.long(), minlength=vocab_size).double()
    weights = torch.where(counts > 0, counts.rsqrt(), torch.zeros_like(counts))
    weights = weights * counts.sum() / (weights * counts).sum()
    return weights.float()


# ---------------------------------------------------------------- metrics
@torch.no_grad()
def predict(model, x, device, batch_size=4096):
    """Logits for every row of x (returned on the CPU)."""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        out.append(model(x[i:i + batch_size].to(device).long()).cpu())
    return torch.cat(out)


def auc_ovr(y_true, probs):
    """Macro one-vs-rest AUC-ROC, only over the classes present in y_true
    (same rule as Workshop 1)."""
    present = np.unique(y_true)
    y_bin = (y_true[:, None] == present[None, :]).astype(int)
    p = probs[:, present]
    p = p / p.sum(axis=1, keepdims=True)
    return float(roc_auc_score(y_bin, p, average="macro"))


def compute_metrics(y_true, logits, with_auc=True):
    """Same metrics as Workshop 1: accuracy, precision, recall, F1 and AUC-ROC."""
    pred = logits.argmax(1).numpy()
    m = {"accuracy": float((pred == y_true).mean())}
    for avg in ("macro", "weighted"):
        m[f"precision_{avg}"] = float(precision_score(y_true, pred, average=avg, zero_division=0))
        m[f"recall_{avg}"] = float(recall_score(y_true, pred, average=avg, zero_division=0))
        m[f"f1_{avg}"] = float(f1_score(y_true, pred, average=avg, zero_division=0))
    if with_auc:
        m["auc_roc_ovr_macro"] = auc_ovr(y_true, logits.softmax(1).numpy())
    return m


# ---------------------------------------------------------------- training
def train(args, vocab, data, device, save=True):
    """Train one configuration and return its history and best epoch."""
    set_seeds()
    vocab_size = vocab["size"]
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]
    auc_idx = subsample(len(x_val), AUC_VAL_SAMPLES)
    auc_idx = slice(None) if auc_idx is None else auc_idx

    model = CharLSTM(vocab_size, dropout=args.dropout).to(device)
    weights = class_weights(y_train, vocab_size).to(device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                           eta_min=args.lr * 0.01)
    writer = SummaryWriter(RUNS_DIR / args.run_name)

    x_train, y_train = x_train.to(device), y_train.to(device).long()
    n = len(x_train)
    generator = torch.Generator().manual_seed(SEED)
    history, best, best_state, bad_epochs = [], None, None, 0
    start_time = time.time()
    print(f"{args.run_name}: {n:,} train rows, {len(x_val):,} val rows, device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        order = torch.randperm(n, generator=generator).to(device)
        total_loss, correct, n_batches = 0.0, 0, 0
        for i in range(0, n, args.batch_size):
            idx = order[i:i + args.batch_size]
            xb, yb = x_train[idx].long(), y_train[idx]
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()
            n_batches += 1
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # validation at the end of every epoch
        val_logits = predict(model, x_val, device)
        val = compute_metrics(y_val.numpy(), val_logits, with_auc=False)
        val_loss = nn.functional.cross_entropy(
            val_logits, y_val.long(), weight=None if weights is None else weights.cpu()).item()
        val_auc = auc_ovr(y_val.numpy()[auc_idx], val_logits[auc_idx].softmax(1).numpy())

        row = {"epoch": epoch, "lr": lr,
               "train_loss": total_loss / n_batches, "val_loss": val_loss,
               "train_accuracy": correct / n, "val_accuracy": val["accuracy"],
               "val_f1_macro": val["f1_macro"], "val_auc_ovr_subsample": val_auc}
        history.append(row)
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)

        improved = best is None or row["val_f1_macro"] > best["val_f1_macro"] + 1e-4
        print(f"epoch {epoch}/{args.epochs}  loss {row['train_loss']:.4f}/{val_loss:.4f}  "
              f"acc {row['train_accuracy']:.4f}/{row['val_accuracy']:.4f}  "
              f"val macro-F1 {row['val_f1_macro']:.4f}  val AUC {val_auc:.4f}  "
              f"lr {lr:.2e}  ({time.time() - t0:.0f}s){'  *best*' if improved else ''}")
        if improved:
            best, bad_epochs = row, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("early stopping")
                break

    writer.close()
    result = {"history": history, "best": best, "n_train": n,
              "train_seconds": time.time() - start_time}

    if save:
        plot_learning_curves(history, args.run_name)
        model.load_state_dict(best_state)
        save_checkpoint(model, args, vocab, data, device, result)
    return result


def save_checkpoint(model, args, vocab, data, device, result):
    """Save checkpoints/<run-name>.pt; copy it to model_best.pt if it is better."""
    # final train (fixed subsample) and validation metrics of the best epoch
    x_train, y_train = data["train"]
    idx = subsample(len(x_train), TRAIN_EVAL_SAMPLES)
    if idx is not None:
        x_train, y_train = x_train[idx], y_train[idx]
    final_metrics = {
        "train_subsample": compute_metrics(y_train.numpy(), predict(model, x_train, device)),
        "val": compute_metrics(data["val"][1].numpy(), predict(model, data["val"][0], device)),
    }
    best = result["best"]
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_hparams": model.hparams,
        "train_hparams": {"lr": args.lr, "weight_decay": args.weight_decay,
                          "dropout": args.dropout, "batch_size": args.batch_size,
                          "epochs": args.epochs, "class_weights": args.class_weights,
                          "max_train_samples": args.max_train_samples, "seed": SEED,
                          "optimizer": "AdamW", "scheduler": "CosineAnnealingLR",
                          "clip_grad_norm": CLIP_NORM, "patience": PATIENCE},
        "vocab": {"itos": vocab["itos"], "stoi": vocab["stoi"], "unk_index": vocab["unk_index"]},
        "epoch": best["epoch"],
        "val_metric": {"name": "f1_macro", "value": best["val_f1_macro"]},
        "final_metrics": final_metrics,
        "run_name": args.run_name,
        "train_seconds": result["train_seconds"],
        "history": result["history"],
    }
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS_DIR / f"{args.run_name}.pt"
    torch.save(checkpoint, path)
    print(f"saved {path.name} (epoch {best['epoch']}, val macro-F1 {best['val_f1_macro']:.4f})")

    if BEST_CHECKPOINT.exists():
        current = torch.load(BEST_CHECKPOINT, weights_only=False)["val_metric"]["value"]
        if best["val_f1_macro"] <= current:
            print(f"model_best.pt kept (val macro-F1 {current:.4f})")
            return
    shutil.copyfile(path, BEST_CHECKPOINT)
    print(f"{args.run_name} copied to model_best.pt")


def plot_learning_curves(history, run_name):
    """Loss, accuracy and validation macro-F1 / AUC per epoch."""
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(h["epoch"], h["train_loss"], "o-", label="train")
    axes[0].plot(h["epoch"], h["val_loss"], "s-", label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(h["epoch"], h["train_accuracy"], "o-", label="train")
    axes[1].plot(h["epoch"], h["val_accuracy"], "s-", label="validation")
    axes[1].set_title("Accuracy")
    axes[2].plot(h["epoch"], h["val_f1_macro"], "s-", label="val macro-F1")
    axes[2].plot(h["epoch"], h["val_auc_ovr_subsample"], "^-", label="val AUC-ROC")
    axes[2].set_title("Validation macro-F1 and AUC-ROC")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(run_name)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"learning_curves_{run_name}.png", dpi=110)
    plt.close(fig)


def search(args, vocab, data, device):
    """Try every (learning rate, regularisation) pair and compare on validation."""
    rows = []
    for lr in SEARCH_LRS:
        for reg in SEARCH_REG:
            run = argparse.Namespace(**vars(args))
            run.lr, run.dropout, run.weight_decay = lr, reg["dropout"], reg["weight_decay"]
            run.run_name = f"search_lr{lr:g}_do{run.dropout:g}_wd{run.weight_decay:g}"
            result = train(run, vocab, data, device, save=False)
            best = result["best"]
            rows.append({"run_name": run.run_name, "lr": lr, "dropout": run.dropout,
                         "weight_decay": run.weight_decay, "best_epoch": best["epoch"],
                         "val_f1_macro": best["val_f1_macro"], "val_accuracy": best["val_accuracy"],
                         "val_auc_ovr_subsample": best["val_auc_ovr_subsample"],
                         "minutes": result["train_seconds"] / 60})
    table = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False)
    table.to_csv(REPORTS_DIR / "hparam_search.csv", index=False)
    print(table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--run-name", default="lstm_noweights")
    parser.add_argument("--search", action="store_true", help="run the hyperparameter search")
    args = parser.parse_args()

    set_seeds()
    device = torch.device(args.device) if args.device else get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    REPORTS_DIR.mkdir(exist_ok=True)

    vocab = read_vocab()
    data = {}
    for split, size in (("train", args.max_train_samples), ("val", args.max_val_samples)):
        x, y = load_encoded(split, vocab)
        idx = subsample(len(x), size)
        if idx is not None:
            idx = torch.from_numpy(idx)
            x, y = x[idx], y[idx]
            print(f"{split} subsample: {len(idx):,} rows")
        data[split] = (x, y)

    if args.search:
        search(args, vocab, data, device)
    else:
        train(args, vocab, data, device)


if __name__ == "__main__":
    main()
