"""Latent space analysis on the TEST split.

    python -m src.analyze_latent

We take 2,000 fixed test windows whose next character is one of the 5 most
frequent classes and encode them with:
  - the AE encoder (z),
  - the VAE encoder (mu, not a sampled z),
  - the Workshop 2 Transformer, frozen (vector of the last position, 128 values).

Each representation is reduced to 2D with t-SNE and UMAP and colored by the
ground-truth label (next character). Class separability is measured with the
silhouette score on the full latent vectors (not on the 2D projection).
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import umap
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from src.autoencoder import (CHECKPOINTS_DIR, FIGURES_DIR, REPORTS_DIR, SEED, batched, get_device,
                             load_model, load_split, load_w2_transformer, read_vocab, set_seeds)
from src.train_ae import top5_sample

warnings.filterwarnings("ignore")


def main():
    set_seeds()
    device = get_device()
    vocab = read_vocab()
    itos = vocab["itos"]
    x, y = load_split("test", vocab)
    idx = top5_sample(y)
    x, y = x[idx], y[idx].numpy()

    ae, ae_ckpt = load_model(CHECKPOINTS_DIR / "ae_best.pt", device)
    vae, vae_ckpt = load_model(CHECKPOINTS_DIR / "vae_best.pt", device)
    w2 = load_w2_transformer(device)
    latents = {
        f"AE ({ae_ckpt['run']})": batched(lambda xb: ae.encode(xb)[0], x, device).numpy(),
        f"VAE ({vae_ckpt['run']}, mu)": batched(lambda xb: vae.encode(xb)[0], x, device).numpy(),
        "Transformer W2 (supervised)": batched(w2.features, x, device).numpy(),
    }

    rows = []
    fig, axes = plt.subplots(len(latents), 2, figsize=(11, 4.6 * len(latents)))
    labels = sorted(set(y))
    for row, (name, z) in zip(axes, latents.items()):
        sil = silhouette_score(z, y)
        rows.append({"representation": name, "dimensions": z.shape[1], "silhouette": sil})
        projections = {
            "t-SNE": TSNE(n_components=2, perplexity=30, random_state=SEED).fit_transform(z),
            "UMAP": umap.UMAP(n_neighbors=15, random_state=SEED).fit_transform(z),
        }
        for ax, (method, p) in zip(row, projections.items()):
            for label in labels:
                m = y == label
                char = "<space>" if itos[label] == " " else itos[label]
                ax.scatter(p[m, 0], p[m, 1], s=4, alpha=0.6, label=char)
            ax.set_title(f"{name} - {method} (silhouette {sil:.3f})", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].legend(title="next char", markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "latent_space.png", dpi=100)
    plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "silhouette_scores.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
