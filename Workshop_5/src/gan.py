"""
Preparation forthe GAN, the shakespeare dataset structure labels are:
x: Set of 80 characters (input)
y: 1 Charachter (target)

The GAN need to understand the strctuture, so we performs the opreration x+y
to stablish the real text structure generation. 
"""
import json 
from pathlib import Path
import pandas as pd 
import torch
from torch import nn
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
WORKSHOP_1 = ROOT.parent / "Workshop_1"

TRAIN = WORKSHOP_1 /"data" / "processed" / "train.parquet"
VOCAB = WORKSHOP_1 / "data" / "vocab.json"

SEQUENCE_LENGTH = 81 #x+y

def load_vocab(): 
    with VOCAB.open(encoding="utf-8") as file: 
        return json.load(file)

class ShakespeareSequences(Dataset): 

    def __init__(self, dataFrame, stoi):
        self.x = dataFrame["x"].tolist()
        self.y = dataFrame["y"].tolist()
        self.stoi = stoi #string to integer
        self.unk_id = stoi["<UNK>"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        text = self.x[index] + self.y[index] # the completed text 
        #conversion to ids --> each characters becomes an id
        ids = [self.stoi.get(char,self.unk_id) for char in text]
        return torch.tensor(ids , dtype=torch.long)

def loadTrainingData(): 
    dataFrame = pd.read_parquet(TRAIN) 
    vocab = load_vocab()
    dataset = ShakespeareSequences(dataFrame, vocab["stoi"])
    return dataset, vocab 


class Generator(nn.Module): 

    def __init__(self, vocab_size, noise_dim=128,hidden_dim = 512): 
        super().__init__()
        self.vocab_size = vocab_size
        self.noise_dim = noise_dim

        self.network = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LeakyReLU(0.2), 
            nn.Linear(hidden_dim, SEQUENCE_LENGTH*vocab_size)
        )

    def forward(self, noise):
        batch_size = noise.size(0)

        logits = self.network(noise)
        #order the values of (sequence lenght*vocab size)
        logits = logits.view(batch_size, SEQUENCE_LENGTH, self.vocab_size)

        return torch.softmax(logits, dim=-1)

class Discriminator(nn.Module): 

    def __init__(self,vocab_size, hidden_dim = 512):
        super().__init__()
        input_dim = SEQUENCE_LENGTH * vocab_size

        self.network = nn.Sequential( 
            nn.Linear(input_dim, hidden_dim), 
            nn.LeakyReLU(0.2), 
            nn.Linear(hidden_dim,hidden_dim), 
            nn.LeakyReLU(0.2), 
            nn.Linear(hidden_dim,1)
        )

    def forward(self, sequence): 

        flattened = sequence.reshape(sequence.size(0), -1)

        return self.network(flattened).squeeze(1)

class Critic(Discriminator): 
    pass #For the Wgan

