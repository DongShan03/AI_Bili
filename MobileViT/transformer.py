from typing import Optional
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads,
                attn_dropout, bias=True, *args, **kwargs):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim must be divisible by num_heads, but got embed_dim={embed_dim} and num_heads={num_heads}")
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x):
        #! [N, P, C]
        b_sz, n_patches, in_channels = x.shape
        #! [N, P, C] -> [N, P, 3C] -> [N, P, 3, H, C]
        qkv = self.qkv_proj(x).reshape(b_sz, n_patches, 3, self.num_heads, -1)
        #! [N, P, 3, H, C] -> [N, H, 3, P, C]
        qkv = qkv.transpose(1, 3).contiguous()
        q, k, v = qkv.unbind(dim=2)
        q = q * self.scaling
        #! transpose: [N, H, P, C] -> [N, H, C, P]
        #! matmul: [N, H, P, C] x [N, H, C, P] -> [N, H, P, P]
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)
        #! attn: [N, H, P, P]

        #! matmul: [N, H, P, P] x [N, H, P, C] -> [N, H, P, C]
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)
        #! [N, H, P, C] -> [N, P, H, C] -> [N, P, C]
        out = self.out_proj(out)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim: int,
                num_heads: Optional[int]=8, attn_dropout: Optional[float]=0.0,
                dropout: Optional[float]=0.0,
                ffn_dropout: Optional[float]=0.0, *args, **kwargs):
        super().__init__()
        attn_unit = MultiHeadAttention(embed_dim, num_heads, attn_dropout, bias=True)
        self.per_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim),
            attn_unit,
            nn.Dropout(p=dropout)
        )
        self.per_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ffn_latent_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(ffn_latent_dim, embed_dim, bias=True),
            nn.Dropout(p=dropout)
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.std_dropout = dropout
        self.ffn_dropout = ffn_dropout

    def forward(self, x):
        res = x
        x = self.per_norm_mha(x)
        x = res + x
        x = x + self.per_norm_ffn(x)
        return x
