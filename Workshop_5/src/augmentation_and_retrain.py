import argparse
import importlib.util
from pathlib import Path
import random
import numpy as np
import pandas as pd 
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.device import (
    DEVICE_CHOICES,
    cpu_state_dict,
    dataloader_kwargs,
    is_directml,
    print_device_info,
    resolve_device,
    seed_everything,
)
from src.gan import Generator, SEQUENCE_LENGTH, load_vocab

ROOT = Path(__file__).resolve().parents[1]
WORKSHOP_1 = ROOT.parent / "Workshop_1"
WORKSHOP_2 = ROOT.parent / "Workshop_2"

W2_CHECKPOINT_DIR = WORKSHOP_2 / "checkpoints"/ "model_best.pt"

GAN_CHECKPOINTS= {
    "vanilla": ROOT/"checkpoints" / "gan_vanilla_last.pt", 
    "wgan_gp": ROOT /"checkpoints" / "gan_wgan_last.pt"
} 
OUTPUT_CHECKPOINT_DIR =ROOT /"checkpoints"
REPORT_DIR = ROOT /"reports" / "metrics"
AUGMENTED_DATA_DIR = ROOT / "data" / "augmented"


WINDOW_SIZE = 80 
UNK_IDX = 0
SEED = 42

def set_seeds(seed): 
    seed_everything(seed)

def load_transformer_class(): 

    model_path = WORKSHOP_2 / "src" /"model.py"
    spec = importlib.util.spec_from_file_location("workshop2_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CharTransformer

def encode_dataframe(dataframe, stoi): 
    #encode the contexts x and the objectives y, with the vocab of th W1 

    lookup = np.full(256,UNK_IDX,dtype=np.uint8)

    for char, idx in stoi.items(): 
        if len(char) == 1 and ord(char) < 256: 
            lookup[ord(char)] = idx

    raw_x = np.frombuffer(
        "".join(dataframe['x']).encode("latin-1"), dtype=np.uint8
                          ) 
    x = torch.from_numpy(
        lookup[raw_x].reshape(-1,WINDOW_SIZE).copy()
    )
    y = torch.from_numpy(
        dataframe["y_id"].to_numpy().astype(np.uint8)
    )
    return x,y

def load_encoded_split(split, stoi): 
     cache_path = WORKSHOP_2 / "data" / "processed" / f"{split}.pt"

     if cache_path.exists(): 
         cached = torch.load(
             cache_path,
             map_location="cpu", 
             weights_only=False
         )
         return cached['x'], cached['y']
     parquet_path = (WORKSHOP_1 / "data" / "processed" / f"{split}.parquet")
     df = pd.read_parquet(parquet_path)
     return encode_dataframe(df, stoi)

def select_subset(x,y,max_samples,seed): 
    # select a reproducible subset for the three conditions 
    if max_samples is None or max_samples >= len(x): 
        return x,y

    generator = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(len(x), generator=generator)[:max_samples]
    return x[idxs], y[idxs]

def get_class_weights(y,vocab_size): 
    # same weights as the W2 class, calculated with real rows 

    counts = torch.bincount(y.long(), minlength=vocab_size).double()
    weights = torch.where(counts >0, counts.rsqrt(), torch.zeros_like(counts))
    weights = weights* counts.sum() /(weights*counts).sum()
    return weights.float()

@torch.no_grad()
def generate_syntethic_pairs(checkpoint_path, count, device):
    # generation of the 81 charcaters sequences, split x = 80, y = 1
    checkpoint = torch.load(
        checkpoint_path, 
        map_location="cpu", 
        weights_only= False
    )
    generator = Generator(
        vocab_size=checkpoint["vocab_size"],
        noise_dim= checkpoint["noise_dim"]
    )
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator = generator.to(device=device)
    generator.eval()

    all_x = []
    all_y = []
    remaining = count
    batch_size = 128

    while remaining > 0: 
        current_batch = min(batch_size,remaining)
        noise = torch.randn(current_batch,checkpoint["noise_dim"],device=device)

        probabilities = generator(noise)
        if is_directml(device):
            probabilities = probabilities.cpu()

        ids = torch.multinomial(
            probabilities.reshape(-1, checkpoint["vocab_size"]), 
            num_samples=1
        ).reshape(current_batch,SEQUENCE_LENGTH)

        #selects the 80 position --> character 81
        all_x.append(ids[:, :WINDOW_SIZE].cpu().to(torch.uint8))
        all_y.append(ids[:, WINDOW_SIZE].cpu().to(torch.uint8))
        remaining -= current_batch
    # cat == concatenations 
    return torch.cat(all_x), torch.cat(all_y)

@torch.no_grad()
def predict(model,x,device,batch_size=4096): 
    # prediction to limitate memmory usage 
    model.eval()
    predictions = []

    for start in range(0, len(x),batch_size):
        xb = x[start:start+batch_size].to(device).long()
        predictions.append(model(xb).argmax(dim=1).cpu())

    return torch.cat(predictions).numpy()

def evaluate(model,x,y,device): 
    predictions = predict(model,x,device)
    targets = y.numpy()

    return({
        "accuracy": accuracy_score(targets,predictions), 
        "f1_macro": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0
        )
    })

def train_condition(
        condition,x_train,y_train,x_val,y_val,
        x_test,y_test,model_class,model_hparams,
        train_hparams,vocab,args,device
): 
    # Train the conidtion th se same transformer an its parameters from W2
    set_seeds(args.seed)
    model = model_class(**model_hparams).to(device)

    use_class_weights = train_hparams.get("class_weights", False)

    if use_class_weights: 
        # the weights are calculted bys the real training
        weights = get_class_weights(args.real_y,vocab["size"]).to(device)
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate, 
        weight_decay= args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.epochs,
        eta_min= args.learning_rate *0.01,#learning rate of each parameter group using a cosine annealing.
    )

    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(x_train,y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
        **dataloader_kwargs(device),
    )

    best_val_f1 = -1.0
    best_state = None 
    best_epoch = 0
    bad_epochs =0
    history = []
    patience = train_hparams.get("patience",3)

    for epoch in range(1,args.epochs +1):
        model.train()
        total_loss = 0.0 

        for xb, yb in loader: 
            xb = xb.to(device).long()
            yb = yb.to(device).long()

            logits = model(xb)
            loss = criterion(logits,yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        val_metrics = evaluate(model, x_val,y_val,device=device)
        mean_loss = total_loss /max(len(loader),1)
        history.append({
            "condition": condition,
            "epoch": epoch,
            "train_loss": mean_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
        })

        print(
            f"\n{condition} | epoch {epoch:03d}/{args.epochs}"
            f"\nloss {mean_loss:.4f}"
            f"\nval accuracy {val_metrics['accuracy']:.4f}"
            f"\nval macro-F1 {val_metrics['f1_macro']:.4f}"
        )

        if val_metrics['f1_macro'] > best_val_f1 +1e-4: 
            best_val_f1 =val_metrics['f1_macro']
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key,value in model.state_dict().items()
            }
            bad_epochs = 0
        else: 
            bad_epochs+=1 
            if bad_epochs >= patience: 
                print(f"{condition}| aerly stop")
                break
    model.load_state_dict(best_state)
    val_metrics = evaluate(model,x_val,y_val, device=device)
    test_metrics = evaluate(model,x_test,y_test,device = device)

    OUTPUT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_CHECKPOINT_DIR /f"transformer_{condition}.pt"

    saved_train_hparams = dict(train_hparams)
    saved_train_hparams.update(
        {
            "lr": args.learning_rate, 
            "weight_decay": args.weight_decay, 
            "batch_size": args.batch_size, 
            "epochs": args.epochs, 
            "max_train_samples": len(x_train),
            "seed": args.seed
        }
    )

    torch.save(
        {
            "state_dict": cpu_state_dict(model),
            "model_hparams": model_hparams,
            "train_hparams": saved_train_hparams,
            "vocab": vocab, 
            "epoch": best_epoch,
            "val_metric":{
                "name": "f1_macro", 
                "value":val_metrics["f1_macro"]
            },
            "run_game": f"workshop5_{condition}",
            "augmentation": condition
        },
        checkpoint_path
    )

    result = {
        "condition": condition, 
        "train_rows": len(x_train),
        "best_epoch": best_epoch, 
        "validation_accuracy": val_metrics["accuracy"],
        "validation_f1_macro": val_metrics["f1_macro"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "checkpoint": str(checkpoint_path)
    }
    return result, history

def main(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--syntethic-count",type = int, default= 1000)
    parser.add_argument("--max-real-train-samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int,default=None)
    parser.add_argument("--max-test_samples", type=int, default=None)
    parser.add_argument("--epochs",type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate",type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int ,default=SEED)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)

    args = parser.parse_args()

    if args.syntethic_count <0: 
        raise ValueError("Syntethic coun must be positive")
    if not W2_CHECKPOINT_DIR.exists(): 
        raise FileNotFoundError(f"There is no checkpoint on {W2_CHECKPOINT_DIR}")

    w2_checkpoint = torch.load(
        W2_CHECKPOINT_DIR,
        map_location="cpu",
        weights_only=False
    )

    model_hparams = w2_checkpoint['model_hparams']
    train_hparams = w2_checkpoint['train_hparams']
    vocab = load_vocab()
    model_class = load_transformer_class()

    args.epochs = args.epochs or train_hparams["epochs"]
    args.batch_size = args.batch_size or train_hparams["batch_size"]
    args.learning_rate = args.learning_rate or train_hparams["lr"]
    args.weight_decay = args.weight_decay or train_hparams["weight_decay"]

    device, device_desc = resolve_device(args.device)
    print_device_info(device, device_desc)

    x_real, y_real = load_encoded_split("train", vocab["stoi"])
    x_val, y_val = load_encoded_split("val", vocab["stoi"])
    x_test, y_test = load_encoded_split("test", vocab["stoi"])

    x_real, y_real =select_subset(
        x_real,
        y_real,
        args.max_real_train_samples or train_hparams.get("max_train_samples"),
        args.seed
        )
    x_val, y_val = select_subset(x_val,y_val, args.max_val_samples, args.seed)
    x_test,y_test = select_subset(x_test,y_test,args.max_test_samples, args.seed)

    args.real_y = y_real

    conditions = [
        ("real_only",None),
        ("vanilla", GAN_CHECKPOINTS["vanilla"]),
        ("wgan_gp", GAN_CHECKPOINTS["wgan_gp"])
    ]

    results =[]
    all_history = []

    for condition,gan_checkpoint in conditions:
        if gan_checkpoint is None: 
            x_train, y_train = x_real, y_real
        else: 
            if not gan_checkpoint.exists(): 
                raise FileNotFoundError(f"there is no checkpoint {gan_checkpoint}")

            x_syntethic, y_syntethic = generate_syntethic_pairs(
                gan_checkpoint,
                args.syntethic_count,
                device=device
            )
            x_train = torch.cat([x_real,x_syntethic])
            y_train = torch.cat([y_real,y_syntethic])

        result, condition_history = train_condition(
                condition=condition,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                model_class=model_class,
                model_hparams=model_hparams,
                train_hparams=train_hparams,
                vocab=vocab,
                args=args,
                device=device
            )
        results.append(result)
        all_history.extend(condition_history)
        REPORT_DIR.mkdir(parents=True,exist_ok=True)
        results_path = REPORT_DIR / "augmentation_results.csv"
        results_frame = pd.DataFrame(results)
        results_frame.to_csv(results_path, index=False)

        print("\nResultados:")
        print(results_frame.round(4).to_string(index=False))
        print(f"saved o {results_path}")

    history_path = REPORT_DIR / "augmentation_history.csv"
    pd.DataFrame(all_history).to_csv(history_path, index=False)
    print(f"Historial por época guardado en: {history_path}")

if __name__ == "__main__":
    main()
