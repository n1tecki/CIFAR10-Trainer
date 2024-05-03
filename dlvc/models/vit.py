


## TODO implement your own ViT in this file
# You can take any existing code from any repository or blog post - it doesn't have to be a huge model
# specify from where you got the code and integrate it into this code repository so that 
# you can run the model with this code

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class PatchEmbeddings(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_size):
        super(PatchEmbeddings, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_size = embed_size
        
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [N, embed_size, num_patches**0.5, num_patches**0.5]
        x = x.flatten(2)  # [N, embed_size, num_patches]
        x = x.transpose(1, 2)  # [N, num_patches, embed_size]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Self-attention
        src2 = self.norm1(src)
        attn_output, _ = self.attention(src2, src2, src2)
        src = src + self.dropout(attn_output)
        
        # Feed-forward
        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        src = src + self.dropout(src2)
        return src

class YourViT(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_channels=3, num_classes=10, embed_size=768, depth=6, heads=8, forward_expansion=4, dropout=0.1):
        super(YourViT, self).__init__()
        self.patch_embeddings = PatchEmbeddings(img_size, patch_size, in_channels, embed_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        self.positional_embeddings = nn.Parameter(torch.randn(1, 1 + (img_size // patch_size) ** 2, embed_size))
        self.dropout = nn.Dropout(dropout)
        
        self.transformers = nn.ModuleList([
            TransformerEncoder(embed_size, heads, forward_expansion, dropout) for _ in range(depth)
        ])
        
        self.to_cls_head = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        x = self.patch_embeddings(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # Expand CLS token to full batch
        x = torch.cat((cls_tokens, x), dim=1)  # Append CLS token
        x += self.positional_embeddings  # Add positional embeddings
        x = self.dropout(x)

        for transformer in self.transformers:
            x = transformer(x)
        
        cls_token_final = x[:, 0]
        return self.to_cls_head(cls_token_final)
