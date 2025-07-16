import math
import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_heads,
        hidden_dim,
        num_layers,
        output_size,
        dropout=0.5,
    ):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        transformer_layer = nn.TransformerEncoderLayer(
            embed_size, num_heads, hidden_dim, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers)
        self.fc = nn.Linear(embed_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.embedding(src) * math.sqrt(src.size(1))  # Scaling
        embedded = self.pos_encoder(embedded)
        transformer_out = self.transformer_encoder(embedded)
        output = self.fc(
            self.dropout(transformer_out.mean(dim=1))
        )  # Aggregate over sequence
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
