import sys
import torch
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

from model.vit.embeddings.conv_embedder import PatchEmbedder
from model.vit.embeddings.positional_encoder import PositionalEncoder
from model.vit.encoder.singlehead_attention import AttentionHead

embedder = PatchEmbedder(in_channels = 1, 
                         patch_size = 4, 
                         hidden_dim = 8)

positionlal_encoder = PositionalEncoder(image_size = 28, 
                                        patch_size = 4, 
                                        hidden_dim = 8)

attention = AttentionHead(8, 4)

# image like MNIST
input = torch.randn(1, 1, 28, 28)
embeddings = embedder(input)
positional_encoding = positionlal_encoder(embeddings)
att = attention(positional_encoding)
