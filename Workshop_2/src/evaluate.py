from pathlib import Path
import json

import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from .model import CharTransformer

BASE_DIR = Path(__file__).resolve().parent
WORKSHOP_DIR = BASE_DIR.parent


TEST_PATH = WORKSHOP_DIR / "data" / "processed" / "test.npz"
CHECKPOINT_PATH = WORKSHOP_DIR / "checkpoints" / "model_best.pt"
RESULTS_PATH = WORKSHOP_DIR / "results" / "test_metrics.json"

BATCH_SIZE = 64
NUM_CLASSES = 76


def main(): 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Used device:{DEVICE}")


    with np.load(TEST_PATH) as data: 
        x_test = torch.from_numpy(data["X"])
        y_test = torch.from_numpy(data["y"])

    test_dataset = TensorDataset(x_test,y_test)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    model = CharTransformer(vocab_size=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loss_function = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    predictions_all = []
    targets_all = []


    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device=DEVICE, dtype=torch.long)
            y_batch = y_batch.to(device=DEVICE, dtype=torch.long)

            logits = model(X_batch)
            loss = loss_function(logits, y_batch)
            predictions = logits.argmax(dim=1)

            batch_size = y_batch.size(0)
            loss_sum += loss.item() * batch_size
            correct += (predictions == y_batch).sum().item()
            total += batch_size

            predictions_all.extend(predictions.cpu().tolist())
            targets_all.extend(y_batch.cpu().tolist())

    test_loss = loss_sum / total
    test_accuracy = correct / total
    test_macro_f1 = f1_score(
        targets_all,
        predictions_all,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    results = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_macro_f1": checkpoint.get("val_macro_f1"),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_examples": total,
    }

    print("\nResultados en test:")
    for name, value in results.items():
        print(f"{name}: {value}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
