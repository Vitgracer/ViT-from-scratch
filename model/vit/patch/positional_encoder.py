import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, image_size, patch_size, hidden_dim):
        super().__init__()
        
        # num_patches = 49 for MNIST
        num_patches = (image_size ** 2) // (patch_size ** 2)

        self.cls_token = torch.nn.Parameter(
            torch.normal(mean=0, std=0.02, size=(1, 1, hidden_dim))
        )

        # shape: (1, 50, 8), all patches and cls token. 
        # We do it learnable, but can use sinusod fixed encodings
        self.positional_embeddings = torch.nn.Parameter(
            torch.normal(mean=0, std=0.02, size=(1, num_patches + 1, hidden_dim))
        )

    def forward(self, patch_embeddings):
        cls_token = self.cls_token.expand(patch_embeddings.size(0), -1, -1)
        cls_patch_embeddings = torch.cat((cls_token, patch_embeddings), dim=1)
        return cls_patch_embeddings + self.positional_embeddings

#encoder = PositionalEncoder(1, 28, 4, 8)
#input = torch.randn(1, 1, 28, 28)
#output = encoder(input)