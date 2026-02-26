"""
Perceiver融合机制 - 借鉴AXIS的跨注意力设计

核心思想：
1. 使用Cross-Attention将时间序列特征映射到词嵌入空间
2. 引入可学习的原型嵌入(prototype embeddings)
3. 引入固定提示嵌入(fixed prompt embeddings)
4. 支持local hint（逐点特征）和fixed hint（全局特征）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiheadAttention(nn.Module):
    """Standard Multi-head Attention module, non-causal by default."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B = query.shape[0]
        
        # Project inputs to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape for multi-head attention
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Prepare attention mask
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

        # Compute scaled dot-product attention
        y = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            is_causal=False
        )

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)
        y = self.out_proj(y)
        return y


class PerceiverFusion(nn.Module):
    """
    Perceiver融合模块
    
    将时间序列特征通过Cross-Attention映射到LLM的词嵌入空间
    
    Args:
        vocab_size: LLM词表大小
        llm_hidden_size: LLM隐藏层维度
        d_proj: 时间序列投影维度
        num_prototype: 原型嵌入数量
        num_fixed_tokens: 固定提示token数量
        num_heads: 注意力头数
        dropout: Dropout率
    """
    
    def __init__(self,
                 vocab_size: int,
                 llm_hidden_size: int,
                 d_proj: int = 256,
                 num_prototype: int = 1000,
                 num_fixed_tokens: int = 20,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.llm_hidden_size = llm_hidden_size
        self.d_proj = d_proj
        self.num_prototype = num_prototype
        self.num_fixed_tokens = num_fixed_tokens
        
        # 1. 映射层：将词表空间映射到原型空间
        # 这个层在训练时通过LLM的词嵌入动态获取源特征
        self.prototype_mapping = nn.Linear(vocab_size, num_prototype)
        
        # 2. 固定提示嵌入 (可学习)
        self.fixed_prompt_embeddings = nn.Parameter(
            torch.randn(1, num_fixed_tokens, llm_hidden_size) * 0.02
        )
        
        # 3. 时间序列特征投影
        self.ts_feature_proj = nn.Linear(d_proj, llm_hidden_size)
        
        # 4. 时间压缩（将seq_len压缩到num_local_tokens）
        self.num_local_tokens = 20  # 局部提示token数量
        self.temporal_compress = nn.AdaptiveAvgPool1d(self.num_local_tokens)
        
        # 5. Cross-Attention层
        self.local_cross_attn = MultiheadAttention(
            embed_dim=llm_hidden_size,
            num_heads=num_heads
        )
        
        self.fixed_cross_attn = MultiheadAttention(
            embed_dim=llm_hidden_size,
            num_heads=num_heads
        )
        
        # 6. 输出MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden_size * 4, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size)
        )
        
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数"""
        # Linear layers
        for m in [self.prototype_mapping, self.ts_feature_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        # Attention layers
        for layer in [self.local_cross_attn, self.fixed_cross_attn]:
            for proj in [layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj]:
                nn.init.xavier_uniform_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
        
        # Fixed prompt embeddings
        nn.init.normal_(self.fixed_prompt_embeddings, mean=0.0, std=0.02)
    
    def get_source_embeddings(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """
        将LLM的词嵌入映射到原型空间，作为Cross-Attention的源特征
        
        Args:
            word_embeddings: (vocab_size, llm_hidden_size) LLM的词嵌入权重
            
        Returns:
            source_embeddings: (1, num_prototype, llm_hidden_size)
        """
        # 映射到原型空间
        # word_embeddings: (vocab_size, llm_hidden_size)
        # 转置后: (llm_hidden_size, vocab_size)
        mapped = self.prototype_mapping(word_embeddings.T)  # (llm_hidden_size, num_prototype)
        return mapped.T.unsqueeze(0)  # (1, num_prototype, llm_hidden_size)
    
    def process_local_embeddings(self,
                                 local_embeddings: torch.Tensor,
                                 source_embeddings: torch.Tensor) -> torch.Tensor:
        """
        处理局部时间序列特征（逐点特征）
        
        Args:
            local_embeddings: (B, seq_len, num_features, d_proj)
            source_embeddings: (1, num_prototype, llm_hidden_size)
            
        Returns:
            local_prompts: (B, num_local_tokens, llm_hidden_size)
        """
        B = local_embeddings.shape[0]
        
        # 1. 合并num_features维度到d_proj（简单平均）
        # (B, seq_len, num_features, d_proj) -> (B, seq_len, d_proj)
        local_features = local_embeddings.mean(dim=2)
        
        # 2. 投影到LLM维度
        # (B, seq_len, d_proj) -> (B, seq_len, llm_hidden_size)
        local_features = self.ts_feature_proj(local_features)
        
        # 3. 时间压缩: (B, seq_len, llm_hidden_size) -> (B, llm_hidden_size, seq_len) -> (B, llm_hidden_size, num_local_tokens) -> (B, num_local_tokens, llm_hidden_size)
        local_features = local_features.transpose(1, 2)  # (B, llm_hidden_size, seq_len)
        compressed = self.temporal_compress(local_features)  # (B, llm_hidden_size, num_local_tokens)
        compressed = compressed.transpose(1, 2)  # (B, num_local_tokens, llm_hidden_size)
        
        # 4. Cross-Attention: 查询是时间序列特征，键值是源特征
        # 这样可以让时间序列特征从词表原型中学习到语义信息
        local_prompts = self.local_cross_attn(
            query=compressed,
            key=source_embeddings.expand(B, -1, -1),
            value=source_embeddings.expand(B, -1, -1)
        )
        
        return local_prompts
    
    def process_fixed_embeddings(self,
                                  source_embeddings: torch.Tensor,
                                  batch_size: int) -> torch.Tensor:
        """
        处理固定提示嵌入（全局特征）
        
        Args:
            source_embeddings: (1, num_prototype, llm_hidden_size)
            batch_size: int
            
        Returns:
            fixed_prompts: (B, num_fixed_tokens, llm_hidden_size)
        """
        # 获取固定嵌入
        fixed_embeds = self.fixed_prompt_embeddings.expand(batch_size, -1, -1)  # (B, num_fixed_tokens, llm_hidden_size)
        
        # Cross-Attention: 查询是固定提示，键值是源特征
        fixed_prompts = self.fixed_cross_attn(
            query=fixed_embeds,
            key=source_embeddings.expand(batch_size, -1, -1),
            value=source_embeddings.expand(batch_size, -1, -1)
        )
        
        return fixed_prompts
    
    def forward(self,
                local_embeddings: torch.Tensor,
                word_embeddings: torch.Tensor,
                use_local_hint: bool = True,
                use_fixed_hint: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            local_embeddings: (B, seq_len, num_features, d_proj) 时间序列特征
            word_embeddings: (vocab_size, llm_hidden_size) LLM词嵌入权重
            use_local_hint: 是否使用局部提示
            use_fixed_hint: 是否使用固定提示
            
        Returns:
            soft_prompts: (B, num_tokens, llm_hidden_size) 用于插入LLM的软提示
        """
        B = local_embeddings.shape[0]
        
        # 获取源嵌入
        source_embeddings = self.get_source_embeddings(word_embeddings)  # (1, num_prototype, llm_hidden_size)
        
        prompts = []
        
        # 处理局部特征
        if use_local_hint:
            local_prompts = self.process_local_embeddings(local_embeddings, source_embeddings)
            local_prompts = self.output_mlp(local_prompts)
            prompts.append(local_prompts)
        
        # 处理固定特征
        if use_fixed_hint:
            fixed_prompts = self.process_fixed_embeddings(source_embeddings, batch_size=B)
            fixed_prompts = self.output_mlp(fixed_prompts)
            prompts.append(fixed_prompts)
        
        # 合并所有提示
        if len(prompts) > 0:
            soft_prompts = torch.cat(prompts, dim=1)  # (B, num_local_tokens + num_fixed_tokens, llm_hidden_size)
        else:
            # 如果没有提示，返回零张量
            soft_prompts = torch.zeros(B, 1, self.llm_hidden_size, device=local_embeddings.device)
        
        return soft_prompts


class SimplePerceiverFusion(nn.Module):
    """
    简化版Perceiver融合（用于快速实验）
    
    去掉原型映射，直接使用Cross-Attention
    """
    
    def __init__(self,
                 llm_hidden_size: int,
                 d_proj: int = 256,
                 num_local_tokens: int = 20,
                 num_fixed_tokens: int = 10,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.llm_hidden_size = llm_hidden_size
        self.d_proj = d_proj
        self.num_local_tokens = num_local_tokens
        self.num_fixed_tokens = num_fixed_tokens
        
        # 时间序列投影
        self.ts_feature_proj = nn.Sequential(
            nn.Linear(d_proj, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 时间压缩
        self.temporal_compress = nn.AdaptiveAvgPool1d(num_local_tokens)
        
        # 固定提示
        self.fixed_prompts = nn.Parameter(
            torch.randn(1, num_fixed_tokens, llm_hidden_size) * 0.02
        )
        
        # Cross-Attention
        self.cross_attn = MultiheadAttention(
            embed_dim=llm_hidden_size,
            num_heads=num_heads
        )
        
        # 输出MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden_size * 4, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size)
        )
        
        self._init_parameters()
    
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, local_embeddings: torch.Tensor, llm_embeds: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Args:
            local_embeddings: (B, seq_len, num_features, d_proj)
            llm_embeds: LLM的输入嵌入（用于提供查询上下文）
            debug: 是否打印调试信息
            
        Returns:
            soft_prompts: (B, num_tokens, llm_hidden_size)
        """
        B = local_embeddings.shape[0]
        
        # DEBUG
        if debug:
            from utils.debug_utils import DebugLogger
            DebugLogger.log_tensor("Perceiver输入 local_embeddings", local_embeddings, sample_indices=(0,))
        
        # 1. 合并特征维度并投影
        features = local_embeddings.mean(dim=2)  # (B, seq_len, d_proj)
        features = self.ts_feature_proj(features)  # (B, seq_len, llm_hidden_size)
        
        # DEBUG
        if debug:
            from utils.debug_utils import DebugLogger
            DebugLogger.log_tensor("投影后 features", features, sample_indices=(0,))
        
        # 2. 时间压缩
        features = features.transpose(1, 2)  # (B, llm_hidden_size, seq_len)
        local_tokens = self.temporal_compress(features)  # (B, llm_hidden_size, num_local_tokens)
        local_tokens = local_tokens.transpose(1, 2)  # (B, num_local_tokens, llm_hidden_size)
        
        # DEBUG
        if debug:
            from utils.debug_utils import DebugLogger
            DebugLogger.log_tensor("时间压缩后 local_tokens", local_tokens, sample_indices=(0,))
        
        # 3. 修复2: 真正的 Cross-Attention
        # Query 使用固定提示和局部提示的组合
        queries = torch.cat([
            local_tokens,
            self.fixed_prompts.expand(B, -1, -1)
        ], dim=1)  # (B, num_local_tokens + num_fixed_tokens, llm_hidden_size)
        
        # 真正的 Cross-Attention: queries 去关注 local_tokens
        # 这样可以让固定提示从时间序列特征中提取相关信息
        attended = self.cross_attn(query=queries, key=local_tokens, value=local_tokens)
        
        # DEBUG
        if debug:
            from utils.debug_utils import DebugLogger
            DebugLogger.log_tensor("Cross-Attention后 attended", attended, sample_indices=(0,))
        
        # 4. MLP增强
        soft_prompts = self.output_mlp(attended)
        
        # DEBUG
        if debug:
            from utils.debug_utils import DebugLogger
            DebugLogger.log_tensor("Perceiver输出 soft_prompts", soft_prompts, sample_indices=(0,))
            
            # 检查soft prompts的多样性
            if B > 1:
                prompt_sim = torch.cosine_similarity(
                    soft_prompts[0].mean(dim=0).unsqueeze(0),
                    soft_prompts[1].mean(dim=0).unsqueeze(0)
                )
                DebugLogger.log(f"样本0与样本1的soft prompt相似度: {prompt_sim.item():.4f}")
        
        return soft_prompts


if __name__ == "__main__":
    print("Testing PerceiverFusion...")
    
    # 测试完整版Perceiver
    perceiver = PerceiverFusion(
        vocab_size=32000,
        llm_hidden_size=2048,
        d_proj=256,
        num_prototype=1000,
        num_fixed_tokens=10,
        num_heads=8,
        dropout=0.1
    )
    
    # 模拟输入
    B, seq_len, num_features, d_proj = 2, 48, 1, 256
    local_embeds = torch.randn(B, seq_len, num_features, d_proj)
    word_embeds = torch.randn(32000, 2048)
    
    # 前向传播
    soft_prompts = perceiver(local_embeds, word_embeds)
    print(f"Local embeddings: {local_embeds.shape}")
    print(f"Soft prompts: {soft_prompts.shape}")
    
    # 测试简化版
    print("\nTesting SimplePerceiverFusion...")
    simple_perceiver = SimplePerceiverFusion(
        llm_hidden_size=2048,
        d_proj=256,
        num_local_tokens=20,
        num_fixed_tokens=10,
        num_heads=8
    )
    
    llm_embeds = torch.randn(B, 100, 2048)  # 模拟LLM输入
    soft_prompts_simple = simple_perceiver(local_embeds, llm_embeds)
    print(f"Soft prompts (simple): {soft_prompts_simple.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in perceiver.parameters())
    simple_params = sum(p.numel() for p in simple_perceiver.parameters())
    print(f"\nPerceiver parameters: {total_params:,}")
    print(f"Simple Perceiver parameters: {simple_params:,}")
