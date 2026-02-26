"""
基于AXIS论文的时间序列编码器实现
关键特性：
- PatchTST风格的patching
- RoPE (Rotary Positional Embedding)
- RMSNorm 归一化
- LlamaMLP (Gated MLP)
- 输出维度: (B, seq_len, num_features, d_proj)
"""

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
    """Rotary Positional Embedding for injecting positional information."""

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return freqs  # Shape: (seq_len, dim // 2)


class LlamaMLP(nn.Module):
    """Llama风格的Gated MLP"""

    def __init__(self, d_model, dim_feedforward=None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.hidden_size = d_model
        self.intermediate_size = dim_feedforward
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = F.gelu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MultiheadAttentionWithRoPE(nn.Module):
    """Multi-head Attention with Rotary Positional Encoding (RoPE), non-causal by default."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        # Linear projections for Q, K, V, and output
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def apply_rope(self, x, freqs):
        """Apply Rotary Positional Encoding to the input tensor."""
        B, seq_len, _ = x.shape
        x = x.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        x_ = x.reshape(B, self.num_heads, seq_len, self.head_dim // 2, 2)
        cos = freqs.cos().view(1, 1, seq_len, self.head_dim // 2)
        sin = freqs.sin().view(1, 1, seq_len, self.head_dim // 2)

        # Apply rotation
        x_rot = torch.stack(
            [x_[..., 0] * cos - x_[..., 1] * sin, x_[..., 0] * sin + x_[..., 1] * cos],
            dim=-1,
        ).flatten(-2)

        return x_rot.transpose(1, 2).reshape(B, seq_len, self.embed_dim)

    def forward(self, query, key, value, freqs, attn_mask=None):
        B, T, C = query.shape

        # Project inputs to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Apply RoPE to Q and K
        if freqs is not None:
            Q = self.apply_rope(Q, freqs)
            K = self.apply_rope(K, freqs)

        # Reshape for multi-head attention
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Prepare attention mask for padding
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

        # Compute scaled dot-product attention
        y = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, is_causal=False
        )

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class TransformerEncoderLayerWithRoPE(nn.Module):
    """Transformer Encoder Layer with RoPE and RMSNorm."""

    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.self_attn = MultiheadAttentionWithRoPE(d_model, nhead)
        self.input_norm = RMSNorm(d_model)
        self.output_norm = RMSNorm(d_model)
        self.mlp = LlamaMLP(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, freqs, attn_mask=None):
        # Self-attention with residual
        residual = src
        src = self.input_norm(src)
        src = self.self_attn(src, src, src, freqs, attn_mask=attn_mask)
        src = residual + self.dropout(src)

        # MLP with residual
        residual = src
        src = self.output_norm(src)
        src = self.mlp(src)
        src = residual + self.dropout(src)

        return src


class TimeSeriesEncoderAXIS(nn.Module):
    """
    AXIS风格的时间序列编码器

    输出形状: (B, seq_len, num_features, d_proj)

    Args:
        d_model: 模型维度
        d_proj: 投影维度（输出维度）
        patch_size: patch大小
        num_layers: Transformer层数
        num_heads: 注意力头数
        dropout: Dropout率
        num_features: 输入特征数（默认为1，支持多变量扩展）
    """

    def __init__(
        self,
        d_model=512,
        d_proj=256,
        patch_size=6,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        num_features=1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_proj = d_proj
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_features = num_features

        # Patch embedding layer
        self.embedding_layer = nn.Linear(patch_size * num_features, d_model)

        # RoPE (per head dimension)
        head_dim = d_model // num_heads
        self.rope_embedder = RotaryEmbedding(head_dim)

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithRoPE(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection to d_proj per feature
        self.projection_layer = nn.Linear(d_model, patch_size * d_proj * num_features)

        self._init_parameters()

    def _init_parameters(self):
        """初始化参数"""
        for name, param in self.named_parameters():
            if "weight" in name and "linear" in name:
                nn.init.kaiming_uniform_(param, nonlinearity="gelu")
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(self, time_series, mask=None):
        """
        Forward pass

        Args:
            time_series: (B, seq_len, num_features) 或 (B, seq_len)
            mask: (B, seq_len) 可选的padding mask

        Returns:
            local_embeddings: (B, seq_len, num_features, d_proj)
        """
        # Handle single feature input
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(-1)

        device = time_series.device
        B, seq_len, num_features = time_series.size()
        assert num_features == self.num_features, f"Number of features mismatch"

        # Pad sequence to be divisible by patch_size
        padded_length = math.ceil(seq_len / self.patch_size) * self.patch_size
        if padded_length > seq_len:
            pad_amount = padded_length - seq_len
            time_series = F.pad(time_series, (0, 0, 0, pad_amount), value=0)
            if mask is not None:
                mask = F.pad(mask, (0, pad_amount), value=0)

        # Convert to patches: (B, num_patches, patch_size * num_features)
        num_patches = padded_length // self.patch_size
        patches = time_series.view(B, num_patches, self.patch_size * num_features)

        # Embed patches: (B, num_patches, d_model)
        embedded_patches = self.embedding_layer(patches)

        # Create patch-level mask
        if mask is not None:
            mask = mask.view(B, num_patches, self.patch_size)
            patch_mask = mask.sum(dim=-1) > 0  # (B, num_patches)
        else:
            patch_mask = None

        # Generate RoPE frequencies
        freqs = self.rope_embedder(num_patches).to(device)

        # Encode sequence
        output = embedded_patches
        for layer in self.layers:
            output = layer(output, freqs, attn_mask=patch_mask)

        # Project and reshape: (B, num_patches, patch_size * num_features * d_proj)
        patch_proj = self.projection_layer(output)

        # Reshape to (B, seq_len, num_features, d_proj)
        # 【修复】删除冗余的permute操作，直接reshape
        local_embeddings = patch_proj.view(
            B, num_patches, self.patch_size, self.num_features, self.d_proj
        )
        # 将 (B, num_patches, patch_size, num_features, d_proj) 合并为 (B, seq_len, num_features, d_proj)
        local_embeddings = local_embeddings.reshape(
            B, -1, self.num_features, self.d_proj
        )
        local_embeddings = local_embeddings[:, :seq_len, :, :]  # Remove padding

        return local_embeddings


class PretrainHeads(nn.Module):
    """
    预训练任务头：
    1. 掩码重建头
    2. 异常检测头
    """

    def __init__(self, d_proj, num_features=1, dropout=0.1):
        super().__init__()
        self.d_proj = d_proj
        self.num_features = num_features

        # 掩码重建头
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_proj, d_proj * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj * 4, d_proj * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj * 4, 1),  # 预测原始值
        )

        # 异常检测头（二分类）
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_proj, d_proj // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj // 2, 2),  # 二分类：正常/异常
        )

    def reconstruction_loss(self, local_embeddings, original_values, mask):
        """
        计算掩码重建损失

        Args:
            local_embeddings: (B, seq_len, num_features, d_proj)
            original_values: (B, seq_len, num_features)
            mask: (B, seq_len) bool tensor，True表示被掩码的位置
        """
        # 预测: (B, seq_len, num_features, 1)
        reconstructed = self.reconstruction_head(local_embeddings)
        reconstructed = reconstructed.squeeze(-1)  # (B, seq_len, num_features)

        # 只在掩码位置计算损失
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, self.num_features)
        loss = F.mse_loss(reconstructed[mask_expanded], original_values[mask_expanded])
        return loss

    def anomaly_detection_loss(self, local_embeddings, labels):
        """
        计算异常检测损失

        Args:
            local_embeddings: (B, seq_len, num_features, d_proj)
            labels: (B, seq_len) 0=正常, 1=异常, -1=忽略
        """
        # 预测: (B, seq_len, num_features, 2)
        logits = self.anomaly_head(local_embeddings)

        # 平均多特征维度 -> (B, seq_len, 2)
        logits = logits.mean(dim=2)

        # Reshape for loss computation
        B, seq_len, _ = logits.shape
        logits = logits.view(-1, 2)  # (B*seq_len, 2)
        labels_flat = labels.view(-1).long()  # (B*seq_len,)

        # 创建有效标签mask
        valid_mask = labels_flat != -1

        if valid_mask.sum() > 0:
            loss = F.cross_entropy(logits[valid_mask], labels_flat[valid_mask])
        else:
            loss = torch.tensor(0.0, device=logits.device)

        return loss

    def forward(self, local_embeddings):
        """前向传播，返回重建值和异常分数"""
        reconstructed = self.reconstruction_head(local_embeddings).squeeze(-1)
        anomaly_logits = self.anomaly_head(local_embeddings).mean(
            dim=2
        )  # (B, seq_len, 2)
        anomaly_scores = F.softmax(anomaly_logits, dim=-1)[:, :, 1]  # (B, seq_len)
        return reconstructed, anomaly_scores


if __name__ == "__main__":
    # 测试代码
    print("Testing TimeSeriesEncoderAXIS...")

    # 创建模型
    encoder = TimeSeriesEncoderAXIS(
        d_model=512,
        d_proj=256,
        patch_size=6,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        num_features=1,
    )

    # 测试输入
    B, seq_len = 2, 48
    x = torch.randn(B, seq_len, 1)

    # 前向传播
    output = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({B}, {seq_len}, 1, 256)")

    # 测试预训练头
    heads = PretrainHeads(d_proj=256, num_features=1)
    recon, anomaly = heads(output)
    print(f"\nReconstruction shape: {recon.shape}")
    print(f"Anomaly scores shape: {anomaly.shape}")

    # 统计参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nEncoder parameters: {total_params:,}")
