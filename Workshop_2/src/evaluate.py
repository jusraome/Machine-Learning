"""Evaluate the trained models on the TEST split (the only script that uses it).

    python -m src.evaluate

- Same metrics as Workshop 1: accuracy, precision, recall, F1 (macro and
  weighted) and macro one-vs-rest AUC-ROC.
- Comparison table (reports/results_comparison.csv): majority-class predictor,
  Workshop 1 baselines and the two LSTM variants. The Workshop 1 baselines were
  scored on a 10,000-row test sample (test.sample(n=10000, random_state=42)),
  so every model in the table is evaluated on those same rows.
- The selected model (model_best.pt) is also evaluated on the full test split.
- Error analysis: confusion matrix of the 20 most frequent characters and the
  best / worst predicted characters.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, recall_score

from src.model import BEST_CHECKPOINT, SEED, CharLSTM, get_device, load_split, read_vocab, set_seeds
from src.train import CHECKPOINTS_DIR, FIGURES_DIR, REPORTS_DIR, compute_metrics, load_encoded, predict

METRICS = ["accuracy", "precision_macro", "precision_weighted", "recall_macro",
           "recall_weighted", "f1_macro", "f1_weighted", "auc_roc_ovr_macro"]
VARIANTS = {"lstm_noweights": "LSTM without class weights",
            "lstm_weighted": "LSTM with class weights"}


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CharLSTM(**ckpt["model_hparams"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def show(char):
    return "<space>" if char == " " else char


def main():
    set_seeds()
    device = get_device()
    vocab = read_vocab()
    itos = vocab["itos"]

    x_test, y_test = load_encoded("test", vocab)
    y_true = y_test.numpy().astype(np.int64)
    sample = load_split("test").sample(n=10_000, random_state=SEED).index.to_numpy()

    # 1. majority class: always predicts the most frequent class of train
    _, y_train = load_encoded("train", vocab)
    counts = torch.bincount(y_train.long(), minlength=vocab["size"]).double() + 1e-9
    maj_logits = (counts / counts.sum()).log().float().expand(len(sample), -1)
    rows = [{"model": "Majority class (space)", **compute_metrics(y_true[sample], maj_logits)}]

    # 2. Workshop 1 baselines (copied from the output of Workshop_1/src/baselines.py)
    rows += pd.read_csv(REPORTS_DIR / "baselines_workshop1.csv").to_dict("records")

    # 3. the two LSTM variants
    best_ckpt = torch.load(BEST_CHECKPOINT, weights_only=False)
    for run_name, label in VARIANTS.items():
        model, _ = load_model(CHECKPOINTS_DIR / f"{run_name}.pt", device)
        logits = predict(model, x_test, device)
        if run_name == best_ckpt["run_name"]:
            label += " (selected)"
            best_logits = logits
        rows.append({"model": label, **compute_metrics(y_true[sample], logits[sample])})

    table = pd.DataFrame(rows)[["model"] + METRICS]
    table.to_csv(REPORTS_DIR / "results_comparison.csv", index=False)
    print("Test sample (same 10k rows as Workshop 1):")
    print(table.round(4).to_string(index=False))

    # 4. selected model: train / validation / full test metrics
    final = best_ckpt["final_metrics"]
    summary = pd.DataFrame({"train (200k subsample)": final["train_subsample"],
                            "validation": final["val"],
                            "test (full)": compute_metrics(y_true, best_logits)}).loc[METRICS]
    print(f"\nSelected model: {best_ckpt['run_name']} (epoch {best_ckpt['epoch']})")
    print(summary.round(4).to_string())

    # 5. error analysis
    y_pred = best_logits.argmax(1).numpy()
    top20 = pd.Series(y_true).value_counts().index[:20].to_numpy()
    mask = np.isin(y_true, top20)
    cm = confusion_matrix(y_true[mask], y_pred[mask], labels=top20, normalize="true")
    labels = [show(itos[c]) for c in top20]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(20), labels, rotation=90)
    ax.set_yticks(range(20), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix on test (20 most frequent characters)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_top20.png", dpi=110)
    plt.close(fig)

    classes = np.unique(y_true)
    per_class = pd.DataFrame({"char": [show(itos[c]) for c in classes],
                              "support": np.bincount(y_true)[classes],
                              "accuracy": recall_score(y_true, y_pred, labels=classes,
                                                       average=None, zero_division=0)})
    per_class = per_class[per_class["support"] >= 100]
    print("\nBest predicted characters:\n" + per_class.nlargest(10, "accuracy").round(3).to_string(index=False))
    print("\nWorst predicted characters:\n" + per_class.nsmallest(10, "accuracy").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
