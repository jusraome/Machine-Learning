import torch 
import torch.nn as nn 
import math 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__() 
        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe) 

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] 
        return x

class ShakespeareTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_ffw, dropout=0.1, max_seq_len=80):
        super().__init__()
        self.d_model = d_model 
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model) 
        self.positional_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len) 
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ffw,
            dropout=dropout,
            batch_first=True
        ) 
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers
        ) 
        self.fc_out = nn.Linear(in_features=d_model, out_features=vocab_size) 

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model) 
        x = self.positional_encoder(x) 
        x = self.transformer_encoder(x) 
        x_last_char = x[:, -1, :] 
        output = self.fc_out(x_last_char) 
        return output