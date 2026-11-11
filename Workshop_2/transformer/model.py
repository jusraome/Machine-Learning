import torch 
import torch.nn as nn

class CharTransformer(nn.Module): 
    def __init__(
      self, 
      vocab_size = 76, 
      context_length = 80,
      embedding_dim = 128 ,
      num_heads = 4,
      num_layers = 2, 
      dropout = 0.1
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim= embedding_dim
        )

        # An encoder layer with a causal mask acts as a decoder-only block:
        # each position can attend only to itself and earlier positions.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead= num_heads,
            dim_feedforward= 4 * embedding_dim, 
            dropout= dropout,
            batch_first= True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        causal_mask = torch.triu(
            torch.ones(context_length,context_length,dtype=torch.bool),
            diagonal=1
        )
        self.register_buffer("casual_mask", causal_mask)

        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self,x): 

        _,sequence_length = x.shape

        positions = torch.arange(sequence_length,device=x.device)
        x = self.token_embedding(x) + self.position_embedding(positions) # char + position
        mask = self.casual_mask[:sequence_length, :sequence_length]
        x = self.transformer(x, mask=mask)

        #last position to predictthe last character 
        x = x[:, -1, :]
        return self.output_layer(x)


    
