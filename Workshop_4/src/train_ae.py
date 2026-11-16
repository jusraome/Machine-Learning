"""Train all the autoencoders of Workshop 4.

    python -m src.train_ae

Runs (30 epochs each, Adam, batch 256):
  1. AE with z = 32, 64 and 128 (latent dimension experiment).
  2. VAE with beta = 0.5, 1 and 4, using the best z of the AE.
  3. AE and VAE (beta = 1) trained only on "normal" windows (next character is a
     space, the majority class) for the anomaly detection experiment.

For every run we log the train / validation reconstruction loss per epoch to
TensorBoard (runs/workshop4/<run>) and keep the epoch with the lowest validation
loss. At the end of each run we also compute the silhouette score of the
validation latent vectors, to compare latent separability between runs.

Saved checkpoints: ae_best.pt (best AE), vae_best.pt (best VAE by validation
reconstruction), ae_normal.pt and vae_normal.pt (anomaly detection).
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from torch.utils.tensorboard import SummaryWriter

from src.autoencoder import (CHECKPOINTS_DIR, FIGURES_DIR, REPORTS_DIR, RUNS_DIR, SEED,
                             TextAutoencoder, batched, get_device, kl_divergence, load_split,
                             read_vocab, reconstruction_loss, set_seeds)

EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
KL_WARMUP = 10  # KL annealing: beta grows linearly from 0 to its value in the first 10 epochs
N_SILHOUETTE = 2000


def top5_sample(y, n=N_SILHOUETTE):
    """Fixed sample of rows whose label is one of the 5 most frequent next characters."""
    top5 = torch.bincount(y).argsort(descending=True)[:5]
    rows = torch.nonzero(torch.isin(y, top5)).squeeze(1).numpy()
    return np.sort(np.random.default_rng(SEED).choice(rows, n, replace=False))


@torch.no_grad()
def evaluate(model, x, device):
    """Mean reconstruction cross-entropy per character, character accuracy and mean KL."""
    model.eval()
    ce, correct, kl = 0.0, 0, 0.0
    for i in range(0, len(x), 2048):
        xb = x[i:i + 2048].to(device)
        logits, mu, log_var = model(xb)
        ce += reconstruction_loss(logits, xb).sum().item()
        correct += (logits.argmax(-1) == xb).sum().item()
        if model.variational:
            kl += kl_divergence(mu, log_var).sum().item()
    n = len(x)
    return {"recon_ce": ce / (n * x.size(1)), "char_acc": correct / (n * x.size(1)), "kl": kl / n}


def train_run(name, z_dim, variational, beta, data, vocab_size, device):
    set_seeds()
    x_train, x_val = data["x_train"], data["x_val"]
    model = TextAutoencoder(vocab_size, z_dim, variational).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    writer = SummaryWriter(RUNS_DIR / name)
    generator = torch.Generator().manual_seed(SEED)
    history, best, best_state = [], None, None
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        beta_now = beta * min(1.0, epoch / KL_WARMUP) if variational else 0.0
        order = torch.randperm(len(x_train), generator=generator)
        ce_sum = 0.0
        for i in range(0, len(x_train), BATCH_SIZE):
            xb = x_train[order[i:i + BATCH_SIZE]].to(device)
            logits, mu, log_var = model(xb)
            recon = reconstruction_loss(logits, xb)
            loss = recon.mean()
            if variational:
                loss = loss + beta_now * kl_divergence(mu, log_var).mean()  # ELBO
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ce_sum += recon.sum().item()

        val = evaluate(model, x_val, device)
        val_loss = val["recon_ce"] * x_val.size(1) + beta * val["kl"]  # with the final beta
        row = {"epoch": epoch, "train_recon_ce": ce_sum / (len(x_train) * x_train.size(1)),
               "val_recon_ce": val["recon_ce"], "val_char_acc": val["char_acc"],
               "val_kl": val["kl"], "beta": beta_now}
        history.append(row)
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)
        if best is None or val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, **row}
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    writer.close()

    model.load_state_dict(best_state)
    model.eval()
    # latent separability on a fixed validation sample (AE: z, VAE: mu), labels = next character
    latent = batched(lambda xb: model.encode(xb)[0], data["x_sil"], device).numpy()
    silhouette = float(silhouette_score(latent, data["y_sil"].numpy()))
    print(f"[{name}] best epoch {best['epoch']}: val recon CE {best['val_recon_ce']:.4f}, "
          f"char acc {best['val_char_acc']:.4f}, KL {best['val_kl']:.2f}, "
          f"silhouette {silhouette:.4f} ({(time.time() - start) / 60:.1f} min)", flush=True)
    summary = {"run": name, "model": "VAE" if variational else "AE", "z_dim": z_dim,
               "beta": beta if variational else None, "train_data": data["label"],
               "best_epoch": best["epoch"], "val_recon_ce": best["val_recon_ce"],
               "val_char_acc": best["val_char_acc"], "val_kl": best["val_kl"],
               "val_silhouette": silhouette}
    checkpoint = {"state_dict": model.state_dict(), "hparams": model.hparams, "beta": beta,
                  "run": name, "summary": summary, "history": history}
    return summary, history, checkpoint


def plot_curves(histories):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for name, h in histories.items():
        h = pd.DataFrame(h)
        style = "--" if name.startswith("vae") else "-"
        axes[0].plot(h["epoch"], h["train_recon_ce"], style, label=name)
        axes[1].plot(h["epoch"], h["val_recon_ce"], style, label=name)
        if name.startswith("vae"):
            axes[2].plot(h["epoch"], h["val_kl"], style, label=name)
    axes[0].set(title="Train reconstruction loss (CE per character)", xlabel="epoch")
    axes[1].set(title="Validation reconstruction loss (CE per character)", xlabel="epoch")
    axes[2].set(title="VAE: validation KL divergence", xlabel="epoch")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=110)
    plt.close(fig)


def main():
    set_seeds()
    device = get_device()
    for folder in (CHECKPOINTS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    vocab = read_vocab()
    x_train, y_train = load_split("train", vocab)
    x_val, y_val = load_split("val", vocab)
    space = vocab["stoi"][" "]
    sil = top5_sample(y_val)
    all_data = {"x_train": x_train, "x_val": x_val, "label": "all",
                "x_sil": x_val[sil], "y_sil": y_val[sil]}
    # anomaly detection: normal = next character is a space (majority class);
    # these models are trained and selected only on normal windows
    normal_data = {"x_train": x_train[y_train == space], "x_val": x_val[y_val == space],
                   "label": "normal only", "x_sil": x_val[sil], "y_sil": y_val[sil]}
    print(f"train {len(x_train):,} windows ({len(normal_data['x_train']):,} normal), "
          f"val {len(x_val):,}")

    summaries, histories, ckpts = [], {}, {}

    def run(name, z, variational, beta, data):
        summary, history, ckpt = train_run(name, z, variational, beta, data, vocab["size"], device)
        summaries.append(summary)
        histories[name] = history
        ckpts[name] = ckpt

    for z in (32, 64, 128):
        run(f"ae_z{z}", z, False, 0.0, all_data)
    best_ae = max((s for s in summaries if s["model"] == "AE"), key=lambda s: s["val_char_acc"])
    z_best = best_ae["z_dim"]
    for beta in (0.5, 1.0, 4.0):
        run(f"vae_b{beta:g}", z_best, True, beta, all_data)
    best_vae = min((s for s in summaries if s["model"] == "VAE"), key=lambda s: s["val_recon_ce"])

    run("ae_normal", z_best, False, 0.0, normal_data)
    run("vae_normal", z_best, True, 1.0, normal_data)

    torch.save(ckpts[best_ae["run"]], CHECKPOINTS_DIR / "ae_best.pt")
    torch.save(ckpts[best_vae["run"]], CHECKPOINTS_DIR / "vae_best.pt")
    torch.save(ckpts["ae_normal"], CHECKPOINTS_DIR / "ae_normal.pt")
    torch.save(ckpts["vae_normal"], CHECKPOINTS_DIR / "vae_normal.pt")
    print(f"ae_best = {best_ae['run']}, vae_best = {best_vae['run']}")

    table = pd.DataFrame(summaries)
    table.to_csv(REPORTS_DIR / "training_summary.csv", index=False)
    print("\n" + table.round(4).to_string(index=False))
    plot_curves({k: v for k, v in histories.items() if not k.endswith("normal")})


if __name__ == "__main__":
    main()
