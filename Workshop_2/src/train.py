"""Fase 4: bucle de entrenamiento del ShakespeareTransformer.

Ejemplos:
    python src/train.py --epochs 2                                       # entrenamiento completo
    python src/train.py --epochs 1 --max_train_batches 20 --max_val_batches 5   # smoke test
    python -m src.train --epochs 2                                       # equivalente al primero
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Bootstrap de rutas: permite ejecutar tanto 'python src/train.py' (donde
# sys.path[0] sería src/) como 'python -m src.train' desde la raíz.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    DIM_FFW,
    D_MODEL,
    DROPOUT,
    LEARNING_RATE,
    NHEAD,
    NUM_EPOCHS,
    NUM_LAYERS,
    VOCAB_SIZE,
    WINDOW_SIZE,
)
from src.dataset import load_split
from src.model import ShakespeareTransformer

# Rutas de salida del entrenamiento (siempre en la raíz del proyecto)
RUNS_DIR = PROJECT_ROOT / "runs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = CHECKPOINTS_DIR / "model_best.pt"


class ShakespeareDataset(Dataset):
    """
    Clase personalizada de PyTorch para leer tus datos de Shakespeare.
    Convierte las cadenas de 80 caracteres en listas de 80 números enteros
    y el target y_id en un escalar 'long' (el tipo que espera CrossEntropyLoss).
    """

    def __init__(self, split_name):
        # 1. Cargamos el DataFrame (train, val o test) usando tu función existente
        self.df = load_split(split_name)

        # 2. Cargamos el diccionario de vocabulario para traducir letras a números
        vocab_path = DATA_DIR / "vocab.json"
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.stoi = self.vocab["stoi"]          # Diccionario: Letra -> Número
        self.unk_idx = self.vocab["unk_index"]  # Índice para caracteres desconocidos (0)

        # 3. Extraemos las columnas como arrays de numpy: llamar df.iloc fila por
        #    fila es muy lento con millones de ejemplos.
        self._texts = self.df["x"].to_numpy()
        self._targets = self.df["y_id"].to_numpy().astype(np.int64)

    def __len__(self):
        # Le dice a PyTorch cuántas filas tiene este split
        return len(self._texts)

    def __getitem__(self, idx):
        # 1. Obtenemos el texto de 80 caracteres (x) y el target numérico (y_id)
        text_x = self._texts[idx]
        y_target = self._targets[idx]

        # 2. Convertimos cada una de las 80 letras en su número correspondiente
        #    Si una letra no existe en el vocabulario, usa el token <UNK> (0)
        x_encoded = [self.stoi.get(char, self.unk_idx) for char in text_x]

        # 3. Retornamos los tensores de PyTorch (tipo 'long' para índices enteros)
        return (
            torch.tensor(x_encoded, dtype=torch.long),
            torch.tensor(int(y_target), dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Utilidades de entorno
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fija las semillas de random, numpy y torch para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    """Selecciona el dispositivo de cómputo: cuda > mps > cpu (si se pide 'auto')."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(batch_size: int, num_workers: int, seed: int):
    """Crea los DataLoaders de train y val con barajado reproducible."""
    print("Preparando los DataLoaders...", flush=True)
    train_dataset = ShakespeareDataset("train")
    val_dataset = ShakespeareDataset("val")

    # Generador con semilla para que el orden del shuffle sea reproducible
    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = torch.cuda.is_available()  # solo tiene sentido en GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # mezcla para que el modelo no memorice el orden
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader



# ---------------------------------------------------------------------------
# Bucles de entrenamiento y validación
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, writer,
                    epoch: int, global_step: int, max_batches: int = 0):
    """Un pase completo de entrenamiento sobre el loader.

    Devuelve (loss_media_ponderada, global_step_actualizado).
    `max_batches > 0` corta el loop antes de tiempo (solo para smoke tests).
    """
    model.train()
    running_loss = 0.0
    running_examples = 0

    total_batches = len(loader)
    if max_batches:
        total_batches = min(total_batches, max_batches)
    t_start = time.time()

    for batch_idx, (x_batch, y_batch) in enumerate(loader, start=1):
        if max_batches and batch_idx > max_batches:
            break

        x_batch = x_batch.to(device)  # [B, 80]
        y_batch = y_batch.to(device)  # [B]

        optimizer.zero_grad()
        logits = model(x_batch)       # [B, VOCAB_SIZE]
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        # Media ponderada por el tamaño real del batch (el último puede ser menor)
        batch_size = x_batch.size(0)
        running_loss += loss.item() * batch_size
        running_examples += batch_size

        # Pérdida instantánea por step
        writer.add_scalar("Loss/train_step", loss.item(), global_step)
        global_step += 1

        # Progreso intermedio: sin esto, un entrenamiento largo en CPU no
        # imprimiría nada hasta terminar la época completa.
        if batch_idx % 200 == 0 or batch_idx == total_batches:
            elapsed = time.time() - t_start
            print(
                f"  [época {epoch}] batch {batch_idx}/{total_batches} | "
                f"loss={loss.item():.4f} | {elapsed:.0f}s",
                flush=True,
            )

    return running_loss / max(running_examples, 1), global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches: int = 0):
    """Pérdida media de validación, ponderada por el tamaño de cada batch."""
    model.eval()
    running_loss = 0.0
    running_examples = 0

    for batch_idx, (x_batch, y_batch) in enumerate(loader, start=1):
        if max_batches and batch_idx > max_batches:
            break

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        batch_size = x_batch.size(0)
        running_loss += loss.item() * batch_size
        running_examples += batch_size

    return running_loss / max(running_examples, 1)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena el ShakespeareTransformer para predecir el siguiente carácter."
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Número de épocas (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate de AdamW (default: %(default)s)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay de AdamW (default: %(default)s)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Tamaño de batch (default: %(default)s)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Dispositivo (default: %(default)s = cuda > mps > cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla de reproducibilidad (default: %(default)s)")
    parser.add_argument("--max_train_batches", type=int, default=0,
                        help="Límite de batches de train por época; 0 = todos (smoke test)")
    parser.add_argument("--max_val_batches", type=int, default=0,
                        help="Límite de batches de validación; 0 = todos (smoke test)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Workers del DataLoader (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    # Carpeta de TensorBoard única por ejecución: runs/exp_YYYYmmdd_HHMMSS
    run_dir = RUNS_DIR / datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    # 1. Modelo, función de pérdida y optimizador
    model = ShakespeareTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_ffw=DIM_FFW,
        dropout=DROPOUT,
        max_seq_len=WINDOW_SIZE,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    n_params = sum(p.numel() for p in model.parameters())
    train_loader, val_loader = make_loaders(args.batch_size, args.num_workers, args.seed)

    print(f"Device: {device} | Parámetros: {n_params:,} | Run: {run_dir}", flush=True)
    print(f"Épocas: {args.epochs} | lr: {args.lr} | weight_decay: {args.weight_decay}"
          f" | batch_size: {args.batch_size} | seed: {args.seed}", flush=True)
    print(f"Batches/época: train={len(train_loader)} val={len(val_loader)}\n", flush=True)

    best_val_loss = float("inf")
    global_step = 0
    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            # 2. Entrenamiento de la época
            train_loss, global_step = train_one_epoch(
                model, train_loader, criterion, optimizer, device, writer,
                epoch, global_step, max_batches=args.max_train_batches,
            )

            # 3. Validación de la época
            val_loss = evaluate(
                model, val_loader, criterion, device,
                max_batches=args.max_val_batches,
            )

            # 4. Medias por época en TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.flush()

            # 5. Guardado del mejor modelo según la val loss
            improved = val_loss < best_val_loss
            marker = ""
            if improved:
                best_val_loss = val_loss
                os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    BEST_MODEL_PATH,
                )
                marker = "  <-- nuevo mejor modelo guardado"

            elapsed = time.time() - t0
            print(
                f"[Época {epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} | {elapsed:.0f}s{marker}",
                flush=True,
            )
    finally:
        writer.close()

    print(f"\nEntrenamiento terminado. Mejor val_loss={best_val_loss:.4f}", flush=True)
    print(f"Mejor checkpoint: {BEST_MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
