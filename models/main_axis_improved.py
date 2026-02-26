"""
AXIS风格改进版窃电检测模型

核心改进：
1. 使用AXIS风格的时间序列编码器（RoPE + RMSNorm + LlamaMLP）
2. Perceiver跨注意力融合机制
3. 预训练支持（掩码重建 + 异常检测）
4. 输出维度: (B, seq_len, num_features, d_proj)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import math
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ts_encoder_axis import TimeSeriesEncoderAXIS, PretrainHeads
from models.perceiver_fusion import SimplePerceiverFusion


# ==========================================
# 1. 数据预处理（增强版）
# ==========================================
class ElectricityDatasetAXIS(Dataset):
    """
    AXIS风格Dataset：
    1. 支持掩码重建预训练
    2. 更丰富的统计特征
    3. 增强的文本提示
    """
    
    def __init__(self, csv_file, user_stats_path, seq_len=48, mode='train', 
                 mask_ratio=0.15, for_pretrain=False):
        self.df = pd.read_csv(csv_file)
        self.seq_len = seq_len
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.for_pretrain = for_pretrain
        
        with open(user_stats_path, 'r', encoding='utf-8') as f:
            self.user_stats = json.load(f)
        
        # 检测列名确定数据格式
        columns = [c.strip().lower() for c in self.df.columns.tolist()]
        
        # 找到ID和DAY列的索引
        id_idx = None
        day_idx = None
        flag_idx = None
        
        for i, col in enumerate(columns):
            if col == 'id':
                id_idx = i
            elif col == 'day':
                day_idx = i
            elif col == 'flag':
                flag_idx = i
        
        if id_idx is None or day_idx is None:
            raise ValueError(f"无法找到ID或DAY列。列名: {self.df.columns.tolist()}")
        
        # 保存列索引供子类使用
        self.id_idx = id_idx
        self.day_idx = day_idx
        self.flag_idx = flag_idx
        
        self.user_ids = self.df.iloc[:, id_idx].values.astype(str)
        self.dates = pd.to_datetime(self.df.iloc[:, day_idx])
        
        # 确定用电数据起始列
        if flag_idx is not None:
            # 格式: ID, DAY, flag, 48个用电数据点
            self.labels = self.df.iloc[:, flag_idx].values.astype(int)
            data_start_idx = max(id_idx, day_idx, flag_idx) + 1
        else:
            # 格式: ID, DAY, 48个用电数据点 (无flag)
            self.labels = None
            data_start_idx = max(id_idx, day_idx) + 1
        
        self.data_values = self.df.iloc[:, data_start_idx:].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def _get_season(self, month):
        if month in [3, 4, 5]: return "Spring"
        if month in [6, 7, 8]: return "Summer"
        if month in [9, 10, 11]: return "Autumn"
        return "Winter"

    def _calculate_advanced_features(self, vals):
        """计算高级统计特征"""
        features = {}
        
        # 基础统计
        features['mean'] = np.mean(vals)
        features['std'] = np.std(vals)
        features['max'] = np.max(vals)
        features['min'] = np.min(vals)
        features['median'] = np.median(vals)
        
        # 变异系数
        features['cv'] = features['std'] / (features['mean'] + 1e-6)
        
        # 峰度和偏度
        if len(vals) > 3:
            features['skewness'] = pd.Series(vals).skew()
            features['kurtosis'] = pd.Series(vals).kurtosis()
        else:
            features['skewness'] = 0.0
            features['kurtosis'] = 0.0
        
        # 零值比例（窃电常见模式）
        features['zero_ratio'] = np.sum(vals < 0.01) / len(vals)
        
        # 恒定值比例（窃电常见模式）
        diff = np.diff(vals)
        features['constant_ratio'] = np.sum(np.abs(diff) < 0.01) / len(diff)
        
        # 夜间用电量占比（0-6点）
        features['night_ratio'] = np.sum(vals[0:12]) / (np.sum(vals) + 1e-6)
        
        # 白天用电量占比（9-17点）
        features['day_ratio'] = np.sum(vals[18:34]) / (np.sum(vals) + 1e-6)
        
        # 峰谷差
        features['peak_valley_diff'] = features['max'] - features['min']
        
        return features

    def _generate_enhanced_analysis(self, index):
        """增强版文本提示"""
        vals = self.data_values[index]
        uid = self.user_ids[index]
        date = self.dates[index]
        season = self._get_season(date.month)
        is_weekend = date.dayofweek >= 5
        
        # 获取用户历史统计
        u_stats = self.user_stats.get(uid, {
            'mean': np.mean(vals), 
            'std': np.std(vals) + 1e-6,
            'min': np.min(vals),
            'max': np.max(vals)
        })
        
        feat = self._calculate_advanced_features(vals)
        
        # Z-score 异常检测
        z_scores = (vals - u_stats['mean']) / (u_stats['std'] + 1e-6)
        abnormal_count = np.sum(np.abs(z_scores) > 2.0)
        
        # 构建增强提示
        lines = []
        lines.append("=== Electricity Usage Analysis ===\n")
        
        day_type = "weekend" if is_weekend else "weekday"
        lines.append(f"[Context] {season} {day_type} residential load profile.")
        
        lines.append(f"\n[Basic Statistics]")
        lines.append(f"  Mean: {feat['mean']:.3f} kWh (Historical: {u_stats['mean']:.3f})")
        lines.append(f"  Std: {feat['std']:.3f}")
        lines.append(f"  Range: {feat['min']:.3f} - {feat['max']:.3f} kWh")
        
        lines.append(f"\n[Pattern Indicators]")
        lines.append(f"  Zero consumption ratio: {feat['zero_ratio']*100:.1f}%")
        lines.append(f"  Constant usage ratio: {feat['constant_ratio']*100:.1f}%")
        lines.append(f"  Coefficient of variation: {feat['cv']:.3f}")
        
        lines.append(f"\n[Temporal Distribution]")
        lines.append(f"  Night (0-6h): {feat['night_ratio']*100:.1f}%")
        lines.append(f"  Day (9-17h): {feat['day_ratio']*100:.1f}%")
        
        lines.append(f"\n[Anomaly Detection]")
        lines.append(f"  Abnormal time steps (|z|>2): {abnormal_count}/{len(vals)}")
        
        # 风险信号
        risk_signals = []
        if feat['zero_ratio'] > 0.3:
            risk_signals.append("high zero consumption")
        if feat['constant_ratio'] > 0.5:
            risk_signals.append("suspiciously stable pattern")
        if abnormal_count > 10:
            risk_signals.append("frequent anomalies")
        if feat['night_ratio'] < 0.05 and not is_weekend:
            risk_signals.append("abnormally low night usage")
            
        if risk_signals:
            lines.append(f"\n[Risk Signals] {', '.join(risk_signals)}")
        else:
            lines.append(f"\n[Risk Signals] None detected")
        
        return '\n'.join(lines)

    def _create_pretrain_mask(self, seq_len):
        """创建预训练用的掩码"""
        num_mask = int(seq_len * self.mask_ratio)
        mask_indices = np.random.choice(seq_len, num_mask, replace=False)
        mask = np.zeros(seq_len, dtype=bool)
        mask[mask_indices] = True
        return mask

    def __getitem__(self, idx):
        vals = self.data_values[idx]
        uid = self.user_ids[idx]
        
        # 改进: 使用用户历史均值进行标准化（而非单日Z-score）
        # 这样可以保留"相对于用户个人习惯的异常"信息
        # 而不是把低用电量用户的平坦曲线错误地放大成"剧烈波动"
        u_stats = self.user_stats.get(uid, {
            'mean': np.mean(vals),
            'std': np.std(vals) + 1e-6
        })
        
        # 使用历史统计量进行标准化
        mean = u_stats['mean']
        std = u_stats['std'] + 1e-6
        vals_normalized = (vals - mean) / std
        
        text_hint = self._generate_enhanced_analysis(idx)
        
        # 转换为tensor时使用标准化后的数据
        x_seq_tensor = torch.tensor(vals_normalized, dtype=torch.float32).unsqueeze(-1)  # (seq_len, 1)
        
        result = {
            'series': x_seq_tensor,
            'text': text_hint
        }
        
        # 添加标签（如果有）
        if self.labels is not None:
            result['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # 预训练模式：添加掩码
        if self.for_pretrain:
            mask = self._create_pretrain_mask(len(vals))
            result['pretrain_mask'] = torch.tensor(mask, dtype=torch.bool)
            result['original_values'] = x_seq_tensor.clone()
            # 将被掩码的位置置为0（或特殊值）
            x_masked = x_seq_tensor.clone()
            x_masked[mask] = 0.0
            result['series_masked'] = x_masked
        
        # DEBUG: 第一个样本打印调试信息
        if idx == 0 and hasattr(self, '_debug_printed') is False:
            self._debug_printed = True
            from utils.debug_utils import DebugLogger
            DebugLogger.log(f"Dataset首个样本 - 用户ID: {self.user_ids[idx]}")
            DebugLogger.log(f"  原始值范围: [{vals.min():.3f}, {vals.max():.3f}], 当天均值: {np.mean(vals):.3f}")
            DebugLogger.log(f"  用户历史均值: {mean:.3f}, 历史标准差: {std:.4f}")
            DebugLogger.log(f"  标准化后: 均值={vals_normalized.mean():.6f}, 标准差={vals_normalized.std():.6f}")
            DebugLogger.log(f"  文本提示长度: {len(text_hint)} 字符")
        
        return result


# ==========================================
# 2. AXIS风格数值流
# ==========================================
class NumericalStreamAXIS(nn.Module):
    """
    AXIS风格数值流
    
    使用TimeSeriesEncoderAXIS作为编码器
    输出: (B, seq_len, num_features, d_proj)
    """
    
    def __init__(self, 
                 seq_len=48,
                 d_model=512,
                 d_proj=256,
                 patch_size=6,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 num_features=1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_proj = d_proj
        self.patch_size = patch_size
        
        # AXIS风格编码器
        self.encoder = TimeSeriesEncoderAXIS(
            d_model=d_model,
            d_proj=d_proj,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_features=num_features
        )
        
        # 预训练头（可选）
        self.pretrain_heads = None
        
    def forward(self, x, mask=None, return_all_features=False, debug=False):
        """
        Args:
            x: (B, seq_len) 或 (B, seq_len, num_features)
            mask: (B, seq_len) 可选
            return_all_features: 是否返回所有中间特征
            debug: 是否打印调试信息
            
        Returns:
            local_embeddings: (B, seq_len, num_features, d_proj)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, seq_len, 1)
        
        # DEBUG: 输入检查
        if debug:
            from utils.debug_utils import DebugLogger
            DebugLogger.log_tensor("NumericalStream输入", x, sample_indices=(0,))
        
        # 编码
        local_embeddings = self.encoder(x, mask)  # (B, seq_len, num_features, d_proj)
        
        # DEBUG: 输出检查
        if debug:
            from utils.debug_utils import DebugLogger
            DebugLogger.log_tensor("NumericalStream输出", local_embeddings, sample_indices=(0,))
            
            # 检查特征多样性
            if local_embeddings.shape[0] > 1:
                feat_sim = torch.cosine_similarity(
                    local_embeddings[0].flatten().unsqueeze(0),
                    local_embeddings[1].flatten().unsqueeze(0)
                )
                DebugLogger.log(f"样本0与样本1特征相似度: {feat_sim.item():.4f}")
        
        return local_embeddings
    
    def enable_pretrain_heads(self):
        """启用预训练头"""
        self.pretrain_heads = PretrainHeads(
            d_proj=self.d_proj,
            num_features=1,
            dropout=0.1
        )
    
    def pretrain_forward(self, x, pretrain_mask, original_values, anomaly_labels=None):
        """
        预训练前向传播
        
        Args:
            x: (B, seq_len, num_features) 掩码后的输入
            pretrain_mask: (B, seq_len) 掩码位置
            original_values: (B, seq_len, num_features) 原始值
            anomaly_labels: (B, seq_len) 异常标签（可选）
            
        Returns:
            dict: 包含各损失项的字典
        """
        if self.pretrain_heads is None:
            self.enable_pretrain_heads()
        
        # 编码
        local_embeddings = self.encoder(x)  # (B, seq_len, num_features, d_proj)
        
        # 计算重建损失
        recon_loss = self.pretrain_heads.reconstruction_loss(
            local_embeddings, original_values, pretrain_mask
        )
        
        result = {'reconstruction_loss': recon_loss}
        
        # 计算异常检测损失（如果提供了标签）
        if anomaly_labels is not None:
            anomaly_loss = self.pretrain_heads.anomaly_detection_loss(
                local_embeddings, anomaly_labels
            )
            result['anomaly_loss'] = anomaly_loss
        
        return result


# ==========================================
# 3. AXIS风格融合层
# ==========================================
class AXISFusionLayer(nn.Module):
    """
    AXIS风格融合层 - 使用Perceiver跨注意力
    """
    
    def __init__(self, 
                 llm_dim,
                 d_proj=256,
                 num_local_tokens=20,
                 num_fixed_tokens=10,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        self.llm_dim = llm_dim
        self.d_proj = d_proj
        
        # Perceiver融合
        self.perceiver = SimplePerceiverFusion(
            llm_hidden_size=llm_dim,
            d_proj=d_proj,
            num_local_tokens=num_local_tokens,
            num_fixed_tokens=num_fixed_tokens,
            num_heads=num_heads,
            dropout=dropout
        )
        
    def forward(self, num_feat, llm_embeds=None):
        """
        Args:
            num_feat: (B, seq_len, num_features, d_proj)
            llm_embeds: 可选的LLM嵌入作为上下文
            
        Returns:
            soft_prompts: (B, num_tokens, llm_dim)
        """
        if llm_embeds is None:
            # 如果没有提供LLM嵌入，创建一个dummy
            B = num_feat.shape[0]
            llm_embeds = torch.zeros(B, 1, self.llm_dim, device=num_feat.device)
        
        soft_prompts = self.perceiver(num_feat, llm_embeds)
        return soft_prompts


# ==========================================
# 4. 整体感知层（AXIS改进版）
# ==========================================
class H3_PerceptionLayerAXIS(nn.Module):
    """
    AXIS风格整体感知层
    
    整合：
    1. AXIS风格时间序列编码器
    2. Perceiver跨注意力融合
    3. 可选的预训练支持
    """
    
    def __init__(self, 
                 llm_dim,
                 seq_len=48,
                 d_model=512,
                 d_proj=256,
                 patch_size=6,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 num_local_tokens=20,
                 num_fixed_tokens=10,
                 num_features=1):
        super().__init__()
        
        self.llm_dim = llm_dim
        self.d_proj = d_proj
        self.seq_len = seq_len
        
        # AXIS风格数值流
        self.numerical_stream = NumericalStreamAXIS(
            seq_len=seq_len,
            d_model=d_model,
            d_proj=d_proj,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_features=num_features
        )
        
        # Perceiver融合
        self.fusion = AXISFusionLayer(
            llm_dim=llm_dim,
            d_proj=d_proj,
            num_local_tokens=num_local_tokens,
            num_fixed_tokens=num_fixed_tokens,
            num_heads=num_heads,
            dropout=dropout
        )
    
    def forward(self, batch_data, llm_embeds=None):
        """
        Args:
            batch_data: 包含 'series' 的字典
            llm_embeds: 可选的LLM嵌入
            
        Returns:
            soft_prompts: (B, num_tokens, llm_dim)
            info: 附加信息字典
        """
        series = batch_data['series']  # (B, seq_len, num_features) 或 (B, seq_len)
        
        # 数值流编码
        num_feat = self.numerical_stream(series)  # (B, seq_len, num_features, d_proj)
        
        # Perceiver融合
        soft_prompts = self.fusion(num_feat, llm_embeds)
        
        return soft_prompts, {
            "soft_prompts": soft_prompts,
            "text_context": batch_data.get('text', None),
            "ts_features": num_feat
        }
    
    def pretrain_forward(self, batch_data):
        """预训练前向传播"""
        series_masked = batch_data['series_masked']
        pretrain_mask = batch_data['pretrain_mask']
        original_values = batch_data['original_values']
        anomaly_labels = batch_data.get('anomaly_labels', None)
        
        return self.numerical_stream.pretrain_forward(
            series_masked, pretrain_mask, original_values, anomaly_labels
        )


# ==========================================
# 5. 配置管理
# ==========================================
class ModelConfig:
    """AXIS风格模型配置"""
    
    # 小模型（快速实验）
    SMALL = {
        'd_model': 256,
        'd_proj': 128,
        'patch_size': 6,
        'num_layers': 4,
        'num_heads': 4,
        'num_local_tokens': 12,
        'num_fixed_tokens': 8,
        'dropout': 0.1
    }
    
    # 基础模型
    BASE = {
        'd_model': 512,
        'd_proj': 256,
        'patch_size': 6,
        'num_layers': 6,
        'num_heads': 8,
        'num_local_tokens': 16,
        'num_fixed_tokens': 8,
        'dropout': 0.1
    }
    
    # 中等模型（推荐）
    MEDIUM = {
        'd_model': 512,
        'd_proj': 256,
        'patch_size': 6,
        'num_layers': 8,
        'num_heads': 8,
        'num_local_tokens': 20,
        'num_fixed_tokens': 10,
        'dropout': 0.15
    }
    
    # 大模型
    LARGE = {
        'd_model': 768,
        'd_proj': 384,
        'patch_size': 6,
        'num_layers': 10,
        'num_heads': 12,
        'num_local_tokens': 24,
        'num_fixed_tokens': 12,
        'dropout': 0.2
    }


if __name__ == "__main__":
    print("Testing H3_PerceptionLayerAXIS...\n")
    
    # 测试不同配置
    for name, config in [('SMALL', ModelConfig.SMALL), 
                         ('BASE', ModelConfig.BASE),
                         ('MEDIUM', ModelConfig.MEDIUM)]:
        print(f"\n{'='*50}")
        print(f"Testing {name} config:")
        print(f"{'='*50}")
        
        model = H3_PerceptionLayerAXIS(llm_dim=2048, **config)
        
        # 测试输入
        B, seq_len = 2, 48
        batch_data = {
            'series': torch.randn(B, seq_len, 1),
            'text': ['Test 1', 'Test 2']
        }
        
        # 前向传播
        soft_prompts, info = model(batch_data)
        
        print(f"Input shape: {batch_data['series'].shape}")
        print(f"Output soft_prompts: {soft_prompts.shape}")
        print(f"TS features: {info['ts_features'].shape}")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    print("\n" + "="*50)
    print("Testing pretrain forward...")
    print("="*50)
    
    # 测试预训练
    model = H3_PerceptionLayerAXIS(llm_dim=2048, **ModelConfig.SMALL)
    
    B, seq_len = 2, 48
    pretrain_batch = {
        'series_masked': torch.randn(B, seq_len, 1),
        'pretrain_mask': torch.rand(B, seq_len) > 0.85,
        'original_values': torch.randn(B, seq_len, 1),
        'anomaly_labels': torch.randint(0, 2, (B, seq_len)).float()
    }
    
    losses = model.pretrain_forward(pretrain_batch)
    print(f"Reconstruction loss: {losses['reconstruction_loss'].item():.4f}")
    print(f"Anomaly loss: {losses['anomaly_loss'].item():.4f}")
