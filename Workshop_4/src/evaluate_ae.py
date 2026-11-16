"""Evaluation on the TEST split: reconstruction quality and anomaly detection.

    python -m src.evaluate_ae

Reconstruction (ae_best.pt and vae_best.pt):
  - MSE between the one-hot input and the predicted probabilities (summed over
    the 76 classes, averaged over positions), cross-entropy per character and
    character accuracy, also split by character type.
  - 8 test examples: original / AE / VAE text and an error map per position.
  - accuracy per position in the window.

Anomaly detection (ae_normal.pt and vae_normal.pt, trained only on windows whose
next character is a space): the reconstruction error is the anomaly score and
"anomaly" = next character is not a space. For the VAE we also use
reconstruction error + KL. AUC-ROC is compared with the Workshop 2 Transformer
(score = 1 - P(space)) and with the majority-class predictor of Workshop 1.
"""

import string

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.autoencoder import (CHECKPOINTS_DIR, FIGURES_DIR, REPORTS_DIR, SEED, batched, decode,
                             get_device, kl_divergence, load_model, load_split, load_w2_transformer,
                             read_vocab, reconstruction_loss, set_seeds)

N_EXAMPLES = 8


def char_type(c):
    if c == " ":
        return "space"
    if c in string.ascii_lowercase:
        return "lowercase"
    if c in string.ascii_uppercase:
        return "uppercase"
    return "punctuation/other"


@torch.no_grad()
def reconstruct(model, x, device):
    """Probabilities (n, 80, 76) on the CPU, using mu for the VAE."""
    return batched(lambda xb: model(xb)[0].softmax(-1), x, device)


def reconstruction_metrics(probs, x, itos):
    onehot = torch.nn.functional.one_hot(x, probs.size(-1)).float()
    pred = probs.argmax(-1)
    correct = (pred == x).float()
    metrics = {
        "mse": ((onehot - probs) ** 2).sum(-1).mean().item(),
        "ce_per_char": -torch.log(probs.gather(-1, x.unsqueeze(-1)).clamp_min(1e-12)).mean().item(),
        "char_accuracy": correct.mean().item(),
    }
    types = np.array([char_type(c) for c in itos])[x.numpy()]
    for t in ("space", "lowercase", "uppercase", "punctuation/other"):
        metrics[f"acc_{t}"] = correct.numpy()[types == t].mean()
    return metrics, correct


def plot_examples(x, ae_probs, vae_probs, itos, idx):
    """Original text and error maps (1 - probability of the right character) for AE and VAE."""
    fig, axes = plt.subplots(len(idx), 1, figsize=(14, 1.45 * len(idx)))
    for ax, i in zip(axes, idx):
        errors = np.stack([1 - ae_probs[i].gather(-1, x[i].unsqueeze(-1)).squeeze(-1).numpy(),
                           1 - vae_probs[i].gather(-1, x[i].unsqueeze(-1)).squeeze(-1).numpy()])
        ax.imshow(errors, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks([0, 1], ["AE", "VAE"])
        ax.set_xticks([])
        ax.set_title(decode(x[i].tolist(), itos), fontsize=8, family="monospace", loc="left")
    axes[-1].set_xlabel("position in the window (red = high error)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reconstruction_examples.png", dpi=110)
    plt.close(fig)


def main():
    set_seeds()
    device = get_device()
    vocab = read_vocab()
    itos = vocab["itos"]
    x, y = load_split("test", vocab)
    print(f"test: {len(x):,} windows")

    # ---------------------------------------------------------- reconstruction
    ae, ae_ckpt = load_model(CHECKPOINTS_DIR / "ae_best.pt", device)
    vae, vae_ckpt = load_model(CHECKPOINTS_DIR / "vae_best.pt", device)
    rows, correct = [], {}
    probs = {}
    for name, model, ckpt in (("AE", ae, ae_ckpt), ("VAE", vae, vae_ckpt)):
        probs[name] = reconstruct(model, x, device)
        metrics, correct[name] = reconstruction_metrics(probs[name], x, itos)
        rows.append({"model": f"{name} ({ckpt['run']})", **metrics})
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "reconstruction_metrics.csv", index=False)
    print("\nReconstruction on test:\n" + table.round(4).to_string(index=False))

    # examples: same 8 fixed test windows for both models
    idx = np.random.default_rng(SEED).choice(len(x), N_EXAMPLES, replace=False)
    plot_examples(x, probs["AE"], probs["VAE"], itos, idx)
    lines = []
    for i in idx:
        for name, text in (("original", x[i].tolist()), ("AE", probs["AE"][i].argmax(-1).tolist()),
                           ("VAE", probs["VAE"][i].argmax(-1).tolist())):
            lines.append(f"{name:9s}| {decode(text, itos)}")
        lines.append("")
    (REPORTS_DIR / "reconstruction_examples.txt").write_text("\n".join(lines), encoding="utf-8")

    # accuracy per position
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for name in ("AE", "VAE"):
        ax.plot(range(1, 81), correct[name].mean(0).numpy(), label=name)
    ax.set(title="Reconstruction accuracy per position (test)", xlabel="position in the window",
           ylabel="character accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_per_position.png", dpi=110)
    plt.close(fig)

    # ---------------------------------------------------------- anomaly detection
    anomaly = (y != vocab["stoi"][" "]).numpy().astype(int)  # 1 = next character is not a space
    ae_n, _ = load_model(CHECKPOINTS_DIR / "ae_normal.pt", device)
    vae_n, _ = load_model(CHECKPOINTS_DIR / "vae_normal.pt", device)

    def recon_error(model):
        return batched(lambda xb: reconstruction_loss(model(xb)[0], xb), x, device).numpy()

    def kl(model):
        return batched(lambda xb: kl_divergence(*model.encode(xb)), x, device).numpy()

    w2 = load_w2_transformer(device)
    p_space = batched(lambda xb: w2(xb).softmax(-1)[:, vocab["stoi"][" "]], x, device).numpy()
    scores = {
        "Majority class (W1 baseline)": np.zeros(len(x)),
        "AE trained on normal data (reconstruction error)": recon_error(ae_n),
        "VAE trained on normal data (reconstruction error)": recon_error(vae_n),
        "VAE trained on normal data (reconstruction error + KL)": recon_error(vae_n) + kl(vae_n),
        "Transformer W2, supervised (1 - P(space))": 1 - p_space,
    }
    auc = pd.DataFrame([{"method": k, "auc_roc": roc_auc_score(anomaly, v)} for k, v in scores.items()])
    print(f"\nAnomaly detection on test ({anomaly.mean():.1%} anomalies):\n"
          + auc.round(4).to_string(index=False))

    # cumulative table (Workshops 1-4)
    def workshop(name):
        return "W1" if ("W1" in name or "Majority" in name) else "W2" if "W2" in name else "W4"

    w2_table = pd.read_csv(REPORTS_DIR.parents[1] / "Workshop_2" / "reports" / "results_comparison.csv")
    cumulative = [{"workshop": workshop(r.model), "method": r.model,
                   "task": "next-character prediction (10k test rows)",
                   "accuracy": r.accuracy, "f1_macro": r.f1_macro, "auc_roc": r.auc_roc_ovr_macro}
                  for r in w2_table.itertuples()]
    cumulative += [{"workshop": workshop(r.method), "method": r.method,
                    "task": "anomaly detection: next char is not a space (20k test rows)",
                    "accuracy": None, "f1_macro": None, "auc_roc": r.auc_roc}
                   for r in auc.itertuples()]
    pd.DataFrame(cumulative).to_csv(REPORTS_DIR / "cumulative_results.csv", index=False)


if __name__ == "__main__":
    main()
