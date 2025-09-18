import torch 
import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(self, hidden_dim, head_size):
        super().__init__()
        
        self.head_size = head_size
        
        self.wq = nn.Linear(hidden_dim, head_size, bias=False)
        self.wk = nn.Linear(hidden_dim, head_size, bias=False)
        self.wv = nn.Linear(hidden_dim, head_size, bias=False)

    def forward(self, input):
        Q = self.wq(input) # (bs, 50, 4)
        K = self.wk(input)
        V = self.wv(input)

        attention = Q @ K.transpose(-2, -1) # (bs, 50, 50)
        attention = attention / (self.head_size ** 0.5)
        attention = torch.softmax(attention, dim=-1)

        attention = attention @ V # (bs, 50, 4)

        return attention