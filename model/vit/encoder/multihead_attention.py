import torch 
import torch.nn as nn
from model.vit.encoder.singlehead_attention import AttentionHead


class AttentionMultiHead(nn.Module):
    def __init__(self, hidden_dim, head_size, num_heads):
        super().__init__()

        self.heads = torch.nn.ModuleList(
            [AttentionHead(hidden_dim, head_size) for _ in range(num_heads)]
        )

        self.dim_restoration = torch.nn.Linear(head_size * num_heads, hidden_dim)

    def forward(self, input):
        """ Result dimensionality is the same as input """
        head_outputs = [head(input) for head in self.heads]
        stacked_heads = torch.cat(head_outputs, dim = -1)
        result = self.dim_restoration(stacked_heads)
        return result