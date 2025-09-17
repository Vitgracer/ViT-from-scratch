import torch
import torch.nn as nn
from model.vit.patch.conv_embedder import PatchEmbedder


class PositionalEncoder(nn.Module):
    def __init__(self, in_channels, image_size, patch_size, hidden_dim):
        super().__init__()
        
        # 49 for MNIST
        num_patches = (image_size ** 2) // (patch_size ** 2)

        self.patch_embedder = PatchEmbedder(in_channels, patch_size, hidden_dim)
        
        # shape: (1, 50, 8), all patches and cls token. 
        # We do it learnable, but can use sinusod fixed encodings
        self.positional_embeddings = torch.nn.Parameter(
            torch.normal(mean=0, std=0.02, size=(1, num_patches + 1, hidden_dim))
        )

        self.cls_token = torch.nn.Parameter(
            torch.normal(mean=0, std=0.02, size=(1, 1, hidden_dim))
        )

    def forward(self, tensor):
        cls_token = self.cls_token.expand(tensor.size(0), -1, -1)
        patch_embeddings = self.patch_embedder(tensor)
        cls_patch_embeddings = torch.cat((cls_token, patch_embeddings), dim=1)
        return cls_patch_embeddings + self.positional_embeddings

#encoder = PositionalEncoder(1, 28, 4, 8)
#input = torch.randn(1, 1, 28, 28)
#output = encoder(input)