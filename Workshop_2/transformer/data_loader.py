import os
from pathlib import Path
from model import CharTransformer

import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class ShakespeareDataset: 

    def __init__(self, split : str):

        data_path = DATA_DIR / f"{split}.npz"

        with np.load(data_path) as data:
            self.x = data["X"]
            self.y = data["y"]

        if len(self.x) != len(self.y): 
            raise ValueError("Different values -> bad tokenization")

    def __len__(self):
        return len(self.y)

    #! tensor

    def __getitem__(self, index) -> torch.tensor:
        x = torch.tensor(self.x[index], dtype = torch.long)
        y = torch.tensor(self.y[index], dtype = torch.long)
        return x,y 

train_dataset = ShakespeareDataset("train")
test_dataset = ShakespeareDataset("test")
val_dataset = ShakespeareDataset("val")        

"""
Documentation says: 

Train -> shuffle true
Test,Val -> shuffle fasle
"""
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size= 64, 
    shuffle= True
)

test_loader = DataLoader( 
    dataset= test_dataset, 
    batch_size= 64, 
    shuffle= False
)
val_loader = DataLoader( 
    dataset= val_dataset, 
    batch_size=64, 
    shuffle= False 
)

#! verification 
"""
x_batch, y_batch = next(iter(train_loader))

print(x_batch.shape,y_batch.shape) -> torch.Size([64, 80]) torch.Size([64])

model = CharTransformer()
X_batch, y_batch = next(iter(train_loader))

logits = model(X_batch)

print(logits.shape)  -> [64, 76] #logits
print(y_batch.shape) -> [64]

"""