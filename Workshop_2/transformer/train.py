from pathlib import Path

import torch 
import torch.nn as nn 
from sklearn.metrics import f1_score


from data_loader import train_loader, val_loader
from model import CharTransformer

CHEKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_PATH = CHEKPOINT_DIR / "model_best.pt"

EPOCHS = 4
NUM_CLASSES = 76 
best_val_f1 = -1.0

CHEKPOINT_DIR.mkdir(parents=True,exist_ok=True)
#normal gpu verification 

if torch.cuda.is_available(): 
    DEVICE = "cuda"
    print("GPU detected")
else: 
    DEVICE = "cpu"
    print("CPU detected")

model = CharTransformer().to(device=DEVICE) # execution with prefered device 


loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW( # to use W decay
    model.parameters(), 
    lr = 3e-4, 
    weight_decay= 1e-4
)

# scheduler adjust the lr by a given cosine function 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer, 
    T_max= EPOCHS
)


#! training -validation 

for epoch in range(EPOCHS): 
    #train
    model.train()
    train_loss_sum = 0.0 
    train_correct = 0
    train_samples = 0

    for batch_index, (X_batch, y_batch) in enumerate(train_loader, start=1):

        X_batch = X_batch.to(DEVICE) # swithc the tensor work mode based on the device
        y_batch= y_batch.to(DEVICE)

        optimizer.zero_grad() # clean

        logits = model(X_batch)
        loss = loss_function(logits,y_batch)

        loss.backward() # extract 
        optimizer.step() #update

        if batch_index % 100 == 0:
            print(
                f"Época {epoch + 1}/{EPOCHS} | "
                f"Lote {batch_index}/{len(train_loader)} | "
                f"Pérdida: {loss.item():.4f}",
                flush=True,
            )

        batch_size = y_batch.size(0) # tensor, position 0 
        train_loss_sum += loss.item() * batch_size
        train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        train_samples += batch_size # the element 0 y_batch 

    train_loss = train_loss_sum/ train_samples
    train_acc = train_correct/ train_samples
    # validation 
    model.eval()
    val_loss_sum = 0.0 
    val_examples = 0 
    all_prediction = []
    all_targets = []

    
    #we are not trainig the model , torch no grad deactivates grad calculus
    
    with torch.no_grad(): 
        for X_batch, y_batch in val_loader: 

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)
            loss = loss_function(logits, y_batch)

            predictions = logits.argmax(dim =1) #transformes requires the highest prob

            batch_size = y_batch.size(0)
            val_loss_sum += loss.item() * batch_size
            val_examples += batch_size


            all_prediction.extend(predictions.cpu().tolist())
            all_targets.extend(y_batch.cpu().tolist())

    val_loss = val_loss_sum/ val_examples
    val_acc = sum(p==y for p,y in zip(all_prediction,all_targets)) / val_examples

    val_macro_f1 = f1_score(
        all_targets, 
        all_prediction,
        labels= list(range(NUM_CLASSES)),
        average="macro", 
        zero_division=0
    )

    print(
        f'epoch: {epoch+1}/{EPOCHS}\n', 
        f'train loss: {train_loss:.5f}\n',
        f'train accuracy: {train_acc}\n', 
        f'val loss: {val_loss:.5f}\n', 
        f'val accuracy {val_acc:.5f}\n', 
        f'f1 score {val_macro_f1:.4f}'
    )

    if val_macro_f1 > best_val_f1: 
        best_val_f1 = val_macro_f1
        torch.save({
            "model_state_dict": model.state_dict(), 
            "epoch": epoch+1, 
            "val_macro_f1": val_macro_f1
        }, CHECKPOINT_PATH)
        print(f"model saved on {CHECKPOINT_PATH}")

    scheduler.step()

