import torch
import torch.nn as nn
from  einops import rearrange


class PatchEmbedder(nn.Module):
    def __init__(self, in_channels, patch_size, hidden_dim):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        # we can use classical approach, but conv works faster
        self.patch_embedder = nn.Conv2d(
            in_channels = in_channels,
            out_channels = self.hidden_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size,
        )
    
    def forward(self, tensor):
        # shape: (bs, hidden_dim = 8, 7, 7)
        conv_embedding = self.patch_embedder(tensor)
        
        # shape: (bs, 49, hidden_dim = 8)
        embedding = rearrange(conv_embedding, 'b c h w -> b (h w) c')

        return embedding
    
# embedder = PatchEmbedder(in_channels = 1, 
#                          patch_size = 4, 
#                          hidden_dim = 8)

# input = torch.randn(1, 1, 28, 28)
# output = embedder(input)