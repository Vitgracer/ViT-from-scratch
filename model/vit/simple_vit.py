import torch
import torch.nn as nn
from model.vit.embeddings.conv_embedder import PatchEmbedder
from model.vit.embeddings.positional_encoder import PositionalEncoder
from model.vit.encoder.vit_block import BlockViT


class SimpleViT(nn.Module):
    def __init__(self, in_channels, image_size, patch_size, hidden_dim, num_layers, head_size, num_heads, mlp_hidden_size):
        super().__init__()
        
        self.patch_embedder = PatchEmbedder(in_channels, patch_size, hidden_dim)
        self.positional_encoder = PositionalEncoder(image_size, patch_size, hidden_dim)

        self.encoder_blocks = torch.nn.Sequential(
            *[BlockViT(hidden_dim, head_size, num_heads, mlp_hidden_size) for _ in range(num_layers)]
        ) 

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 9),
            nn.Softmax(dim=-1)
        )

    
    def forward(self, image):
        patch_embeddings = self.patch_embedder(image)
        positional_encoded_embeddings = self.positional_encoder(patch_embeddings)
        encodings = self.encoder_blocks(positional_encoded_embeddings)
        
        cls_token = encodings[:, 0, :]
        classification_result = self.classifier(cls_token)
        
        return classification_result
        