import logging

import torch
import torch.nn as nn

from pop2vec.evaluation.prediction_settings.simple_mlp import SimpleMLP


logging.basicConfig(level=logging.DEBUG)

class AttentionAggregatorWithQuery(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        if hidden_dim%num_heads != 0:
            num_heads = self._set_num_heads(hidden_dim, num_heads)

        self.hidden_dim = hidden_dim
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))  # (1, 1, H)

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def _set_num_heads(self, hidden_dim, old_num_heads):
        candidates = [4, 2, 1]
        for c in candidates:
            if hidden_dim%c == 0:
              num_heads = c
              break
        logging.warning(
            f"hidden_dim = {hidden_dim} is not divisible by num_heads = {old_num_heads}. "
            f"num_heads is set to {num_heads}."
        )
        return num_heads


    def forward(self, token_embeddings, attention_mask=None):
        batch_size = token_embeddings.size(0)
        
        # Replicate the single query vector for each item in the batch
        query = self.query.repeat(batch_size, 1, 1)  # (B, 1, H)
        
        # Use query as Q, token_embeddings as K and V
        attn_output, _ = self.attention(
            query,              # shape (B, 1, H)
            token_embeddings,   # shape (B, T, H) => keys
            token_embeddings,   # shape (B, T, H) => values
            key_padding_mask=(attention_mask == 0) if attention_mask is not None else None
        )
        # attn_output is (B, 1, H) after self-attention with a single query
        
        # Apply a feedforward for final refinement
        refined_output = self.ff(attn_output)  # still (B, 1, H)
        
        # Squeeze to (B, H)
        sequence_embedding = refined_output.squeeze(1)
        
        return sequence_embedding

class AttentionMLP(nn.Module):
    def __init__(self, input_dim, output_dim, cfg):
        super().__init__()
        self.attn = AttentionAggregatorWithQuery(
            input_dim, 
            cfg.get('num_heads', 4)
        )
        self.mlp = SimpleMLP(
            input_dim, 
            output_dim, 
            cfg['num_layers'], 
            cfg['activation_fn'], 
            dropout_rate = cfg['DROPOUT_RATE']
        )

    def forward(self, token_embeddings, padding_mask=None):
        # token_embeddings is (B, T, H)
        x = self.attn(token_embeddings, padding_mask) # x is now (B, H)
        x = self.mlp(x) # x is now (B, output_dim)
        return x

