import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """Root Mean Square Normalization layer."""
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.to(torch.float32).pow(2).mean(dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.scale * x_normed).type_as(x)

class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding."""
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return freqs

class LlamaMLP(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, dim_feedforward, bias=True)
        self.up_proj = nn.Linear(d_model, dim_feedforward, bias=True)
        self.down_proj = nn.Linear(dim_feedforward, d_model, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MultiheadAttentionWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def apply_rope(self, x, freqs):
        B, seq_len, _ = x.shape
        x = x.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        x_ = x.reshape(B, self.num_heads, seq_len, self.head_dim // 2, 2)
        cos = freqs.cos().view(1, 1, seq_len, self.head_dim // 2)
        sin = freqs.sin().view(1, 1, seq_len, self.head_dim // 2)
        x_rot = torch.stack([x_[..., 0] * cos - x_[..., 1] * sin, x_[..., 0] * sin + x_[..., 1] * cos], dim=-1).flatten(-2)
        return x_rot.transpose(1, 2).reshape(B, seq_len, self.embed_dim)

    def forward(self, x, freqs):
        B, T, C = x.shape
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if freqs is not None:
            Q = self.apply_rope(Q, freqs)
            K = self.apply_rope(K, freqs)
        
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        y = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

class AxisEncoderLayer(nn.Module):
    """标准的 Encoder Layer 封装"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttentionWithRoPE(d_model, nhead)
        self.input_norm = RMSNorm(d_model)
        self.output_norm = RMSNorm(d_model)
        self.mlp = LlamaMLP(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, freqs):
        src = src + self.dropout(self.self_attn(self.input_norm(src), freqs))
        src = src + self.dropout(self.mlp(self.output_norm(src)))
        return src