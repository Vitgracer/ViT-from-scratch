import torch.nn as nn
from model.vit.embeddings.conv_embedder import PatchEmbedder
from model.vit.embeddings.positional_encoder import PositionalEncoder


class SimpleViT(nn.Module):
    def __init__(self, in_channels, image_size, patch_size, hidden_dim):
        super().__init__()
        
        self.patch_embedder = PatchEmbedder(in_channels, patch_size, hidden_dim)
        self.positional_encoder = PositionalEncoder(image_size, patch_size, hidden_dim)
    
    def forward(self, image):
        patch_embeddings = self.patch_embedder(image)
        positional_encoded_embeddings = self.positional_encoder(patch_embeddings)
        