import torch 
import torch.nn as nn
from model.vit.encoder.multihead_attention import AttentionMultiHead


class BlockViT(nn.Module):
    def __init__(self, hidden_dim, head_size, num_heads, mlp_hidden_size):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mhsa = AttentionMultiHead(hidden_dim, head_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, mlp_hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(mlp_hidden_size, hidden_dim)
        )
    
    def forward(self, input):
        out = input + self.mhsa(self.norm1(input))
        out = out + self.mlp(self.norm2(out))
        return out