import sys
import torch
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

from model.vit.embeddings.conv_embedder import PatchEmbedder
from model.vit.embeddings.positional_encoder import PositionalEncoder
from model.vit.encoder.singlehead_attention import AttentionHead
from model.vit.encoder.multihead_attention import AttentionMultiHead
from model.vit.encoder.vit_block import BlockViT
from model.vit.simple_vit import SimpleViT


embedder = PatchEmbedder(in_channels = 1, 
                         patch_size = 4, 
                         hidden_dim = 8)

positionlal_encoder = PositionalEncoder(image_size = 28, 
                                        patch_size = 4, 
                                        hidden_dim = 8)

attention = AttentionHead(8, 4)
attention_multi = AttentionMultiHead(8, 4, 4)
vit_block = BlockViT(8, 4, 4, 8)
vit = SimpleViT(
    in_channels=1, 
    image_size=28, 
    patch_size=4, 
    hidden_dim=8,
    num_layers=2,
    head_size=4,
    num_heads=4,
    mlp_hidden_size=8
)

# image like MNIST
input = torch.randn(1, 1, 28, 28)
embeddings = embedder(input)
positional_encoding = positionlal_encoder(embeddings)
single_att = attention(positional_encoding)
multi_att = attention_multi(positional_encoding)
vit_block = vit_block(positional_encoding)
vit_result = vit(input)
