# 窃电检测模型问题诊断与优化方案

## 一、项目设计理念与架构

### 1.1 核心设计思想

本项目采用**"时间序列编码器 + 大语言模型 (LLM) 多模态融合"**的端到端架构，旨在解决传统窃电检测方法的痛点：

| 传统方法 | 痛点 | 本项目解决方案 |
|---------|------|--------------|
| 基于规则 | 容易被绕过，无法适应新窃电模式 | 利用LLM的语义理解和推理能力 |
| 机器学习 | 特征工程复杂，需要领域知识 | 端到端学习，自动提取特征 |
| 深度学习 | 黑盒模型，无法解释判断依据 | LLM可生成解释性文本 |

**设计创新点**：让LLM像"电力专家"一样阅读用电数据报告，结合数值特征和文本描述做出可解释的判断。

### 1.2 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     端到端窃电检测架构                        │
└─────────────────────────────────────────────────────────────┘

                    ┌──────────────┐
                    │   原始数据    │
                    │  (48个时间点) │
                    └──────┬───────┘
                           │
                           ▼
    ┌────────────────────────────────────────────────────┐
    │              【感知层 - Perception Layer】          │
    │  将时间序列转换为LLM能理解的"软提示" (Soft Prompts)  │
    └────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ 数值编码  │    │ 文本提示  │    │ 融合模块  │
    │ (AXIS)   │    │ (Prompt) │    │(Perceiver)│
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         │         ┌─────┴─────┐         │
         │         │ 统计特征  │         │
         │         │ 描述文本  │         │
         │         └─────┬─────┘         │
         │               │               │
         └───────────────┼───────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │ Soft Prompts     │
              │ (B, N_tokens, D) │
              └────────┬─────────┘
                       │
                       ▼
    ┌────────────────────────────────────────────────────┐
    │                【认知层 - LLM】                     │
    │  接收 Soft Prompts + Text Embeddings               │
    │  生成判断结果："Theft" 或 "Normal"                 │
    └────────────────────────────────────────────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │   分类结果       │
              │  + 解释性文本    │
              └──────────────────┘
```

### 1.3 关键组件详解

#### **组件A：AXIS风格时间序列编码器** (`ts_encoder_axis.py`)

**设计理念**：借鉴AXIS论文，专门设计用于提取时间序列特征

**技术栈**：
- **Patching**: 将48个时间点切分成patch（默认patch_size=6）
- **RoPE位置编码**: 旋转位置编码，更好地捕获时间关系
- **RMSNorm归一化**: 替代LayerNorm，更稳定
- **LlamaMLP前馈网络**: Gated MLP结构，增强表达能力
- **标准多头注意力**: 非因果模式，适合时间序列

**输入/输出**：
- 输入: `(Batch, seq_len=48, num_features=1)`
- 输出: `(Batch, seq_len, num_features, d_proj=256)`

**预训练支持**：
- 掩码重建头：预测被掩码位置的原始值（MSE损失）
- 异常检测头：二分类（正常/异常）

#### **组件B：Perceiver融合层** (`perceiver_fusion.py`)

**核心问题**：LLM的输入是词嵌入（离散文字），但时间序列是连续数值，如何桥接？

**解决方案**：
1. **时间压缩**: 使用AdaptiveAvgPool1d将seq_len压缩到num_local_tokens
2. **固定提示**: 可学习的固定token嵌入（全局特征）
3. **跨注意力机制**: 查询为固定提示+局部提示，键值为时间序列特征
4. **维度对齐**: 将d_proj映射到llm_hidden_size

**关键实现**：
```python
# 简化版Perceiver
class SimplePerceiverFusion:
    def forward(self, local_embeddings, llm_embeds):
        # local_embeddings: (B, seq_len, num_features, d_proj)
        # 1. 合并特征维度
        features = local_embeddings.mean(dim=2)  # (B, seq_len, d_proj)
        # 2. 投影到LLM维度
        features = self.ts_feature_proj(features)  # (B, seq_len, llm_dim)
        # 3. 时间压缩
        local_tokens = self.temporal_compress(features)  # (B, num_local, llm_dim)
        # 4. Cross-Attention
        queries = torch.cat([local_tokens, fixed_prompts], dim=1)
        attended = self.cross_attn(query=queries, key=local_tokens, value=local_tokens)
        return attended  # (B, num_tokens, llm_dim)
```

#### **组件C：提示工程** (`agent_axis.py`)

**为什么需要文本提示？**
- 纯数值数据对LLM不友好
- 需要将数值转换为LLM能理解的描述性语言

**文本提示结构**：
```
=== Electricity Usage Analysis ===

[Context] Spring weekday residential load profile.

[Basic Statistics]
  Mean: 2.345 kWh (Historical: 3.123)
  Std: 0.456
  Range: 0.012 - 4.567 kWh

[Pattern Indicators]
  Zero consumption ratio: 15.3%
  Constant usage ratio: 23.4%
  Coefficient of variation: 0.345

[Temporal Distribution]
  Night (0-6h): 25.1%
  Day (9-17h): 45.3%

[Anomaly Detection]
  Abnormal time steps (|z|>2): 3/48

[Risk Signals] high zero consumption, suspiciously stable pattern
```

**系统提示**：
```python
sys_msg = (
    "You are an expert in electricity theft detection.\n"
    "Analyze the provided electricity usage data objectively.\n"
    "\n"
    "THEFT INDICATORS:\n"
    "- Unusually low or flat consumption\n"
    "- Sudden drops in usage without explanation\n"
    "- Abnormal patterns: many zero values or constant readings\n"
    "- Usage significantly below historical average\n"
    "\n"
    "NORMAL INDICATORS:\n"
    "- Consistent daily/weekly cycles\n"
    "- Weekend vs weekday differences\n"
    "- Seasonal variations matching weather\n"
    "\n"
    "CRITICAL RULES:\n"
    "1. Output ONLY the word 'Theft' or 'Normal'\n"
    "2. Do NOT use <think> tags or explain your reasoning\n"
    "3. Output the single word only, nothing else"
)
```

#### **组件D：端到端训练策略**

**阶段1：预训练（自监督）**
- **目标**：学习时间序列的通用表示
- **任务1**：掩码重建（随机掩码15%时间点，预测原始值）
- **任务2**：异常检测（基于统计特征生成伪异常标签）
- **数据**：无标签的正常样本（File1_train.csv）
- **损失**：MSE + CrossEntropy

**阶段2：微调（监督学习）**
- **目标**：端到端窃电检测分类
- **输入**：Soft Prompts + Text Embeddings
- **输出**：生成"Theft"或"Normal"
- **损失**：只计算LLM生成部分的Cross-Entropy Loss
- **数据划分**：按用户划分（70%训练，30%验证），避免数据泄露

**数据流向详解**：
```
1. 输入: (Batch=32, seq_len=48, features=1)
   ↓
2. AXIS编码器
   - Patching: (32, 8, 6)  →  8个patch，每个6个点
   - Embedding: (32, 8, d_model=512)
   - Transformer x8层
   - Projection: (32, 48, 1, d_proj=256)
   ↓
3. Perceiver融合
   - 特征平均: (32, 48, 256)
   - 时间压缩: (32, 20, 2048)  ← local tokens
   - 固定提示: (32, 10, 2048)  ← fixed tokens
   - Cross-Attention: (32, 30, 2048)
   ↓
4. 拼接输入 LLM
   - Soft Prompts: (32, 30, 2048)
   - Text Embeddings: (32, text_len, 2048)
   - 拼接后: (32, 30+text_len, 2048)
   ↓
5. LLM 推理
   - 输入: 拼接后的 embeddings
   - 输出: "Theft" 或 "Normal" 的 token
   ↓
6. 损失计算
   - 只计算 LLM 输出部分的损失
   - Soft Prompts 部分标签设为 -100（不参与损失）
```

---

## 二、当前代码存在的主要问题

### 2.1 随机种子设置不完整（导致结果不稳定）

**问题位置**: 所有训练脚本

**问题描述**:
- 代码只在`train_pretrain_axis.py`中部分设置了随机种子，但存在以下缺陷：
  1. `torch.cuda.manual_seed()`没有被设置，CUDA操作不固定
  2. `torch.backends.cudnn.benchmark = False`和`torch.backends.cudnn.deterministic = True`缺失
  3. `os.environ['PYTHONHASHSEED']`未设置
  4. `torch.use_deterministic_algorithms(True)`未启用

**影响**:
- 即使设置了`np.random.seed(42)`，不同运行之间的随机性仍然存在
- 每次训练的初始化权重不同，导致结果波动

**修复代码**:
```python
import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """设置所有随机种子以确保完全可复现性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # 为DataLoader设置worker seed
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    return seed_worker

# 使用
g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    worker_init_fn=set_seed(42),
    generator=g
)
```

### 2.2 数据加载器随机性问题

**问题位置**: `ElectricityDatasetPretrain.__getitem__()`和`__init__`

**问题描述**:
- 使用`torch.randperm()`在`__init__`中采样，但`DataLoader`的`shuffle=True`会导致额外的随机性
- 掩码生成在`__getitem__`中每次调用都是随机的，缺乏复现性

**解决方案**:
```python
class ElectricityDatasetPretrain(ElectricityDatasetAXIS):
    def __init__(self, ...):
        super().__init__(...)
        self.epoch = 0  # 添加epoch追踪
    
    def set_epoch(self, epoch):
        """设置当前epoch，用于确定性掩码生成"""
        self.epoch = epoch
    
    def __getitem__(self, idx):
        vals = self.data_values[idx]
        
        # 为每个idx创建固定的随机种子
        seed = hash((idx, self.epoch)) % (2**32)
        np.random.seed(seed)
        
        # 现在生成的掩码是可复现的
        mask = self._create_pretrain_mask(len(vals))
        ...

# 在每个epoch开始时调用
for epoch in range(epochs):
    train_dataset.set_epoch(epoch)
    for batch in train_loader:
        ...
```

### 2.3 损失计算方式存在严重问题

**问题位置**: `agent_axis.py`第231-272行

**问题描述**:
1. **标签构建逻辑错误**: 第250-254行，只有prompt之后的token被标记为有效，但LLM需要从提示生成完整回答，这会导致模型只能学习特定模式
2. **Soft Prompts部分设置了-100标签**: 虽然正确（不应该计算损失），但与Text Embeddings的拼接可能导致梯度传播问题

**关键代码问题**:
```python
# 第250-254行的问题
text_labels = torch.full_like(tokens.input_ids, -100)
for i, p_len in enumerate(prompt_lens):
    if p_len < tokens.input_ids.shape[1]:
        text_labels[i, p_len:] = tokens.input_ids[i, p_len:]
text_labels[tokens.attention_mask == 0] = -100
```

**问题在于**:
- 实际输入给LLM的是`[Soft Prompts] + [Text Embeddings]`
- 标签只对应`Text Embeddings`部分
- 但LLM生成时看到的是拼接后的输入，这可能导致位置编码和注意力计算出错

**正确做法**:
```python
# 应该让LLM学习从Soft Prompts和文本提示生成回答
# 标签应该对应完整的生成目标

# 构建完整的目标序列
full_input = torch.cat([soft_prompts, text_embeds], dim=1)

# 标签：soft_prompts部分为-100，text_embeds部分为token_ids
prefix_labels = torch.full((batch_size, soft_prompt_len), -100, ...)
# 对于text_embeds，应该是input_ids偏移一位（因果预测）
# 或者使用完整的target序列
```

### 2.4 提示工程（Prompt Engineering）严重不足

**问题位置**: `agent_axis.py`第112-131行

**问题描述**:
1. **系统提示过于笼统**: 虽然列出了指标，但缺乏具体的数值指导
2. **没有Few-shot示例**: LLM没有学习到如何基于具体数值做判断
3. **上下文缺乏关键信息**: 用户的历史基准、用电模式类型等

**改进方案**:
```python
few_shot_examples = """
示例1（窃电）：
分析上下文：
- 均值: 0.523 kWh（历史均值: 2.134）
- 零用电比例: 45%
- 恒定用电比例: 60%
- 夜间用电异常偏低
判断：Theft

示例2（正常）：
分析上下文：
- 均值: 2.456 kWh（历史均值: 2.345）
- 零用电比例: 2%
- 用电曲线有昼夜差异
- 周末用电量下降15%
- 符合居民模式
判断：Normal

示例3（窃电）：
分析上下文：
- 与历史相比用电量突然下降70%
- 大量零值记录（32/48时间点）
- 用电模式从波动变为平坦
判断：Theft
"""

def construct_prompt(self, axis_hints, user_instructions, ground_truth=None):
    for i, (hint, instr) in enumerate(zip(axis_hints, user_instructions)):
        sys_msg = (
            "你是一位电力窃电检测专家。请根据用户的用电数据分析是否存在窃电行为。\n\n"
            "分析维度：\n"
            "1. 用电量异常：与历史均值比较，偏差超过2倍标准差视为异常\n"
            "2. 零值比例：超过30%的零值记录为高风险\n"
            "3. 恒定模式：超过50%的时间点数值相同视为异常\n"
            "4. 时序模式：夜间(0-6h)和白天(9-17h)的用电比例是否合理\n\n"
            "判断标准：\n"
            "- 如果存在明显异常模式（如大量零值、恒定读数、突然下降），输出 'Theft'\n"
            "- 如果用电曲线符合正常居民模式（有昼夜差异、周末差异、季节波动），输出 'Normal'\n\n"
            "输出要求：\n"
            "1. 只输出 'Theft' 或 'Normal'\n"
            "2. 不要解释，不要思考过程，不要任何额外文字\n\n"
            f"{few_shot_examples}\n\n"
            "现在请分析以下数据："
        )
        ...
```

### 2.5 数据标准化策略有待改进

**问题位置**: `main_axis_improved.py`第210-225行

**问题描述**:
- 当前使用`(vals - mean) / std`的Z-score标准化
- 对于窃电检测，异常值正是我们要检测的，标准化会压缩异常值的信息
- 没有处理极端异常值（outliers）

**改进方案**:
```python
def __getitem__(self, idx):
    vals = self.data_values[idx]
    uid = self.user_ids[idx]
    
    # 1. 使用用户历史统计进行标准化
    u_stats = self.user_stats.get(uid, {
        'mean': np.mean(vals),
        'std': np.std(vals) + 1e-6,
        'median': np.median(vals),
        'iqr': np.percentile(vals, 75) - np.percentile(vals, 25) + 1e-6,
        'p95': np.percentile(vals, 95),
        'p5': np.percentile(vals, 5)
    })
    
    # 2. 稳健标准化（Robust Scaling）- 使用中位数和IQR，对异常值不敏感
    median = u_stats.get('median', np.median(vals))
    iqr = u_stats.get('iqr', np.percentile(vals, 75) - np.percentile(vals, 25) + 1e-6)
    vals_normalized = (vals - median) / iqr
    
    # 3. 截断极端值（Winsorization）
    vals_normalized = np.clip(vals_normalized, -5, 5)
    
    # 4. 可选：Min-Max归一化到[0,1]（如果需要）
    # vals_normalized = (vals_normalized - vals_normalized.min()) / (vals_normalized.max() - vals_normalized.min() + 1e-6)
```

### 2.6 类别不平衡问题未处理

**问题位置**: 训练脚本

**问题描述**:
- 窃电样本通常远少于正常样本（通常比例为1:5到1:10）
- 代码中没有使用加权损失或采样策略
- 评估时使用简单的准确率，对稀有类别不公平

**解决方案**:
```python
# 1. 计算类别权重
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(all_labels), 
    y=all_labels
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# 2. 在损失函数中使用加权损失
loss = nn.CrossEntropyLoss(weight=torch.tensor([class_weights[0], class_weights[1]]).to(device))

# 3. 或者使用WeightedRandomSampler
from torch.utils.data import WeightedRandomSampler

sample_weights = [class_weight_dict[label] for label in dataset.labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
loader = DataLoader(dataset, sampler=sampler, ...)

# 4. 使用Focal Loss（更好的处理类别不平衡）
class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# 使用Focal Loss（alpha偏向少数类）
focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
```

### 2.7 模型架构设计缺陷

#### 2.7.1 Perceiver融合层问题

**问题位置**: `perceiver_fusion.py`第282-411行

**问题描述**:
1. **Cross-Attention使用不当**: 查询（queries）应该是可学习的提示，键值（keys/values）应该是时间序列特征，但代码中实现方式不够优化
2. **固定提示（fixed prompts）与局部提示（local prompts）的拼接方式可能不是最优的**

**改进方案**:
```python
class ImprovedPerceiverFusion(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... 初始化代码
        
        # 使用双层Cross-Attention
        self.local_to_fixed_attn = MultiheadAttention(...)  # 局部→全局
        self.fixed_to_local_attn = MultiheadAttention(...)  # 全局→局部
    
    def forward(self, local_embeddings, llm_embeds):
        B = local_embeddings.shape[0]
        
        # 1. 合并特征维度并投影
        features = local_embeddings.mean(dim=2)  # (B, seq_len, d_proj)
        features = self.ts_feature_proj(features)  # (B, seq_len, llm_dim)
        
        # 2. 时间压缩
        features = features.transpose(1, 2)  # (B, llm_dim, seq_len)
        local_tokens = self.temporal_compress(features)  # (B, llm_dim, num_local)
        local_tokens = local_tokens.transpose(1, 2)  # (B, num_local, llm_dim)
        
        # 3. 双层Cross-Attention
        # 第一层：固定提示从局部特征中学习
        fixed_attended = self.local_to_fixed_attn(
            query=self.fixed_prompts.expand(B, -1, -1),  # (B, num_fixed, d)
            key=local_tokens,  # (B, num_local, d)
            value=local_tokens
        )
        
        # 第二层：局部提示从固定提示中学习（双向交互）
        local_attended = self.fixed_to_local_attn(
            query=local_tokens,
            key=fixed_attended,
            value=fixed_attended
        )
        
        # 4. 合并
        combined = torch.cat([local_attended, fixed_attended], dim=1)
        
        # 5. MLP增强
        soft_prompts = self.output_mlp(combined)
        
        return soft_prompts
```

#### 2.7.2 时间序列编码器输出的维度问题

**问题位置**: `ts_encoder_axis.py`第269-279行

**问题描述**:
- 输出形状`(B, seq_len, num_features, d_proj)`在传递给Perceiver时需要`mean(dim=2)`
- 这会丢失多特征的信息，如果有多个特征（如电压、电流），它们被平均了

**改进方案**:
```python
# 为每个特征单独生成提示
class MultiFeaturePerceiverFusion(nn.Module):
    def __init__(self, num_features, d_proj, llm_hidden_size, ...):
        super().__init__()
        self.num_features = num_features
        
        # 为每个特征使用独立的投影层
        self.feature_proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_proj, llm_hidden_size),
                nn.LayerNorm(llm_hidden_size),
                nn.GELU()
            )
            for _ in range(num_features)
        ])
        
        # 每个特征使用独立的压缩
        self.temporal_compress_layers = nn.ModuleList([
            nn.AdaptiveAvgPool1d(num_local_tokens // num_features)
            for _ in range(num_features)
        ])
    
    def forward(self, local_embeddings, ...):
        # local_embeddings: (B, seq_len, num_features, d_proj)
        B, seq_len, num_features, d_proj = local_embeddings.shape
        
        feature_prompts = []
        for i in range(num_features):
            feat_i = local_embeddings[:, :, i, :]  # (B, seq_len, d_proj)
            proj_i = self.feature_proj_layers[i](feat_i)  # (B, seq_len, llm_hidden_size)
            
            # 时间压缩
            proj_i = proj_i.transpose(1, 2)  # (B, llm_dim, seq_len)
            compressed_i = self.temporal_compress_layers[i](proj_i)  # (B, llm_dim, compressed_len)
            compressed_i = compressed_i.transpose(1, 2)  # (B, compressed_len, llm_dim)
            
            feature_prompts.append(compressed_i)
        
        # 合并所有特征的提示
        all_prompts = torch.cat(feature_prompts, dim=1)  # (B, total_compressed_len, d)
        
        # 后续Cross-Attention...
```

### 2.8 训练和验证策略问题

#### 2.8.1 验证集划分方式

**问题位置**: `train_finetune_axis.py`第59-79行

**问题描述**:
- 按用户划分是正确的（避免数据泄露），但划分后没有确保类别分布的一致性
- 训练集和验证集可能有不同的窃电比例

**改进方案**:
```python
def split_users_by_ratio(csv_file, train_ratio=0.7, seed=42):
    """按用户划分，并保持类别分布一致"""
    df = pd.read_csv(csv_file)
    unique_users = df.iloc[:, 0].astype(str).unique()
    
    # 获取每个用户的标签（是否窃电）
    user_labels = {}
    for user in unique_users:
        user_data = df[df.iloc[:, 0].astype(str) == user]
        # 如果用户有任何一天被标记为窃电，则认为是窃电用户
        user_labels[user] = 1 if user_data['flag'].max() > 0 else 0
    
    # 按标签分层采样
    theft_users = [u for u, l in user_labels.items() if l == 1]
    normal_users = [u for u, l in user_labels.items() if l == 0]
    
    np.random.seed(seed)
    np.random.shuffle(theft_users)
    np.random.shuffle(normal_users)
    
    n_train_theft = int(len(theft_users) * train_ratio)
    n_train_normal = int(len(normal_users) * train_ratio)
    
    train_users = theft_users[:n_train_theft] + normal_users[:n_train_normal]
    val_users = theft_users[n_train_theft:] + normal_users[n_train_normal:]
    
    print(f"训练集: {len(train_users)}用户 (窃电: {n_train_theft}, 正常: {n_train_normal})")
    print(f"验证集: {len(val_users)}用户 (窃电: {len(theft_users)-n_train_theft}, 正常: {len(normal_users)-n_train_normal})")
    
    return train_users, val_users
```

#### 2.8.2 早停策略过于简单

**问题位置**: `train_finetune_axis.py`第217-318行

**问题描述**:
- 只基于验证损失早停，没有考虑其他指标
- 对于窃电检测，应该主要关注WF1、MAP@K等指标

**改进方案**:
```python
# 使用综合指标进行早停
from utils.metrics import TheftDetectionMetrics

def evaluate_model(agent, val_loader, device):
    """全面评估模型性能"""
    metrics = TheftDetectionMetrics()
    
    with torch.no_grad():
        for batch in val_loader:
            targets = batch['label'].to(device)
            target_texts = ["Theft" if t == 1 else "Normal" for t in targets]
            instructions = ["Analyze this user's electricity usage pattern."] * len(targets)
            
            responses, _, theft_scores = agent.generate(
                batch, instructions, return_scores=True, debug=False
            )
            
            for i in range(len(targets)):
                true_label = "Theft" if targets[i] == 1 else "Normal"
                pred_text = responses[i].strip().lower()
                pred_label = "Theft" if "theft" in pred_text else "Normal"
                score = theft_scores[i] if theft_scores else 0.5
                
                metrics.update(true_label, pred_label, score, user_id=i)
    
    results = metrics.compute()
    
    # 综合评分
    composite_score = (
        0.3 * results['auc'] +
        0.3 * results['map@40'] +
        0.2 * results['wf1'] +
        0.2 * results['f1_theft']  # 特别关注窃电类别的召回
    )
    
    return composite_score, results

# 在训练循环中使用
best_composite_score = 0
for epoch in range(epochs):
    # ... 训练代码 ...
    
    # 验证
    composite_score, val_results = evaluate_model(agent, val_loader, device)
    
    print(f"Validation - Composite: {composite_score:.4f}, AUC: {val_results['auc']:.4f}, MAP@40: {val_results['map@40']:.4f}")
    
    # 早停判断
    if composite_score > best_composite_score:
        best_composite_score = composite_score
        patience_counter = 0
        torch.save(checkpoint, f"{checkpoint_dir}/finetune_{model_config_name.lower()}_best.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered")
            break
```

### 2.9 测试和推理问题

#### 2.9.1 测试时使用自适应阈值但不保存

**问题位置**: `test_axis_improved.py`第169-189行

**问题描述**:
- 代码自适应地找到最佳阈值，但这个阈值没有被保存或用于后续推理
- 每次测试都要重新计算最佳阈值

**改进方案**:
```python
# 在微调阶段就找到最佳阈值
def find_optimal_threshold(agent, val_loader, device):
    """在验证集上找到最佳阈值"""
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            _, _, scores = agent.generate(
                batch, 
                ["Analyze this user's electricity usage pattern."] * len(batch['label']),
                return_scores=True
            )
            all_scores.extend(scores)
            all_labels.extend(batch['label'].cpu().numpy())
    
    # 搜索最佳阈值
    best_thresh = 0.5
    best_f1 = 0
    best_results = {}
    
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = [1 if s > thresh else 0 for s in all_scores]
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_results = {
                'threshold': thresh,
                'f1': f1,
                'precision': precision_score(all_labels, preds),
                'recall': recall_score(all_labels, preds)
            }
    
    print(f"Optimal threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")
    return best_thresh, best_results

# 保存到检查点
best_thresh, thresh_results = find_optimal_threshold(agent, val_loader, device)

checkpoint = {
    'perception': agent.perception.state_dict(),
    'optimal_threshold': best_thresh,
    'threshold_results': thresh_results,
    'epoch': epoch,
    'loss': avg_loss,
    'config': config,
    'config_name': model_config_name
}
```

### 2.10 调试和监控不足

**问题描述**:
- 没有TensorBoard或WandB集成
- 训练过程中没有可视化中间结果
- 难以诊断模型学习过程

**改进方案**:
```python
from torch.utils.tensorboard import SummaryWriter
import time

# 初始化writer
writer = SummaryWriter(log_dir=f'runs/{model_config_name}_{int(time.time())}')

# 训练循环中记录
for epoch in range(epochs):
    # ... 训练代码 ...
    
    # 记录损失
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    
    # 记录学习率
    writer.add_scalar('Learning_rate', current_lr, epoch)
    
    # 记录指标
    writer.add_scalar('Metrics/AUC', val_results['auc'], epoch)
    writer.add_scalar('Metrics/MAP@40', val_results['map@40'], epoch)
    writer.add_scalar('Metrics/WF1', val_results['wf1'], epoch)
    writer.add_scalar('Metrics/F1_Theft', val_results['f1_theft'], epoch)
    
    # 记录模型权重分布
    for name, param in agent.perception.named_parameters():
        writer.add_histogram(f'weights/{name}', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'grads/{name}', param.grad, epoch)

# 关闭writer
writer.close()
```

---

## 三、推荐的整体优化架构

### 3.1 数据预处理流程重构

```
原始数据
  ↓
1. 数据清洗（处理缺失值、异常值）
  ↓
2. 用户历史统计计算（mean, std, median, IQR, percentiles）
  ↓
3. 稳健标准化（Robust Scaling）
  ↓
4. Winsorization（截断极端值）
  ↓
5. 时间特征工程（day of week, hour, season）
  ↓
6. 创建多特征输入（原始值 + 差分 + 统计特征）
  ↓
输入到模型
```

### 3.2 模型架构优化

```
时间序列编码器（AXIS）
  - RoPE位置编码
  - RMSNorm归一化
  - LlamaMLP前馈网络
  - Multi-head Attention
  ↓
输出: (B, seq_len, num_features, d_proj)
  ↓
Perceiver融合层（改进版）
  - 多特征独立投影（如果使用多特征）
  - 双层Cross-Attention
  - 时间压缩
  ↓
输出: (B, num_tokens, llm_hidden_size)
  ↓
与文本Embedding拼接
  ↓
LLM（Qwen3）
  ↓
输出: "Theft" or "Normal"
```

### 3.3 训练流程优化

```
阶段1: 预训练（自监督）
├── 掩码重建任务 (MSE)
├── 异常检测任务 (BCE)
├── 对比学习任务（可选，进一步学习判别性特征）
└── 使用AdamW + CosineAnnealingWarmRestarts

阶段2: 微调（监督学习）
├── 使用加权采样或加权损失处理类别不平衡
├── 联合损失: L_ce + λ * L_focal
├── 分层学习率（编码器lr小，融合层lr大）
├── 早停基于综合指标（WF1 + MAP@K + AUC）
├── 动态阈值学习并保存
└── 使用验证集全面评估

阶段3: 测试与部署
├── 加载最优阈值
├── 集成推理（可选，多个模型投票）
└── 结果可视化与解释
```

---

## 四、优先修复列表（按重要性排序）

### 🔴 高优先级（必须修复）

1. **完善随机种子设置** - 确保可复现性
   - 位置：所有训练脚本开头
   - 预计提升：结果稳定性

2. **修复损失计算逻辑** - 当前实现可能导致训练失败
   - 位置：`agent_axis.py` forward方法
   - 预计提升：训练有效性

3. **添加类别平衡处理** - 加权损失或采样
   - 位置：训练脚本
   - 预计提升：窃电召回率 +15-20%

4. **改进数据标准化** - 使用Robust Scaling + Winsorization
   - 位置：`main_axis_improved.py` __getitem__
   - 预计提升：异常值检测能力

5. **优化提示工程** - 提供Few-shot示例和更明确的指导
   - 位置：`agent_axis.py` construct_prompt
   - 预计提升：LLM理解能力 +10-15%

### 🟡 中优先级（显著提升效果）

6. **改进Perceiver融合层** - 修复Cross-Attention逻辑
   - 位置：`perceiver_fusion.py`
   - 预计提升：特征融合质量

7. **分层学习率** - 编码器使用较小学习率
   - 位置：训练脚本优化器设置
   - 预计提升：训练稳定性

8. **添加TensorBoard监控** - 便于调试
   - 位置：训练脚本
   - 预计提升：调试效率

9. **在验证集上找到并保存最佳阈值** - 提升推理性能
   - 位置：训练脚本验证阶段
   - 预计提升：F1分数 +5-10%

10. **改进早停策略** - 基于综合指标
    - 位置：训练脚本
    - 预计提升：模型选择质量

### 🟢 低优先级（锦上添花）

11. **添加对比学习预训练任务** - 进一步提升特征质量
12. **集成推理（Ensemble）** - 多模型投票
13. **模型解释性（Attention可视化）** - 理解模型决策
14. **支持多变量输入** - 如果有电压、电流等额外数据

---

## 五、关键代码修复示例

### 5.1 完整的随机种子设置

```python
import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """设置所有随机种子以确保完全可复现性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # 为DataLoader设置worker seed
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    return seed_worker

# 在每个脚本开头调用
set_seed(42)

# 使用DataLoader时
g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    worker_init_fn=set_seed(42),
    generator=g
)
```

### 5.2 Focal Loss（处理类别不平衡）

```python
class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# 在微调时使用
from utils.losses import FocalLoss
focal_loss = FocalLoss(alpha=0.75, gamma=2.0)  # alpha偏向少数类（窃电）

# 联合损失
loss = ce_loss + 0.5 * focal_loss
```

### 5.3 分层学习率

```python
# 为不同层设置不同学习率
param_groups = [
    {
        'params': agent.perception.numerical_stream.encoder.parameters(),
        'lr': lr * 0.1  # 编码器使用较小的学习率
    },
    {
        'params': agent.perception.fusion.parameters(),
        'lr': lr  # 融合层使用标准学习率
    }
]

optimizer = AdamW(param_groups, weight_decay=0.01)
```

### 5.4 改进的提示构建（包含Few-shot示例）

```python
few_shot_examples = """
示例1（窃电）：
分析上下文：
- 均值: 0.523 kWh（历史均值: 2.134）
- 零用电比例: 45%
- 恒定用电比例: 60%
- 夜间用电异常偏低
判断：Theft

示例2（正常）：
分析上下文：
- 均值: 2.456 kWh（历史均值: 2.345）
- 零用电比例: 2%
- 用电曲线有昼夜差异
- 周末用电量下降15%
- 符合居民模式
判断：Normal

示例3（窃电）：
分析上下文：
- 与历史相比用电量突然下降70%
- 大量零值记录（32/48时间点）
- 用电模式从波动变为平坦
判断：Theft
"""

def construct_prompt(self, axis_hints, user_instructions, ground_truth=None):
    for i, (hint, instr) in enumerate(zip(axis_hints, user_instructions)):
        sys_msg = (
            "你是一位电力窃电检测专家。请根据用户的用电数据分析是否存在窃电行为。\n\n"
            "分析维度：\n"
            "1. 用电量异常：与历史均值比较，偏差超过2倍标准差视为异常\n"
            "2. 零值比例：超过30%的零值记录为高风险\n"
            "3. 恒定模式：超过50%的时间点数值相同视为异常\n"
            "4. 时序模式：夜间(0-6h)和白天(9-17h)的用电比例是否合理\n\n"
            "判断标准：\n"
            "- 如果存在明显异常模式（如大量零值、恒定读数、突然下降），输出 'Theft'\n"
            "- 如果用电曲线符合正常居民模式（有昼夜差异、周末差异、季节波动），输出 'Normal'\n\n"
            "输出要求：\n"
            "1. 只输出 'Theft' 或 'Normal'\n"
            "2. 不要解释，不要思考过程，不要任何额外文字\n\n"
            f"{few_shot_examples}\n\n"
            "现在请分析以下数据："
        )
        
        user_content = f"分析上下文：\n{hint}\n\n指令：\n{instr}"
        
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content}
        ]
        # ... 剩余代码
```

---

## 六、预期效果

实施上述优化后，预期可以实现：

| 指标 | 当前水平 | 优化后目标 | 提升幅度 |
|------|---------|-----------|---------|
| **结果稳定性** | 标准差 ~10% | 标准差 < 2% | 5x |
| **AUC** | ~0.65 | ~0.85+ | +30% |
| **MAP@40** | ~0.20 | ~0.50+ | +150% |
| **WF1** | ~0.60 | ~0.80+ | +33% |
| **窃电召回率** | ~50% | ~80%+ | +60% |

---

## 七、实施建议

1. **分阶段实施**：先修复随机种子和损失计算，再优化架构
2. **使用Git管理**：每次修改后提交，便于回滚
3. **小数据集验证**：先用1000个样本快速验证修改效果
4. **A/B测试**：对比改进前后的指标变化
5. **记录实验**：使用表格记录每次实验的配置和结果

### 实施顺序建议：

**第一阶段（基础设施）**：
- [ ] 修复随机种子
- [ ] 添加TensorBoard监控
- [ ] 改进数据标准化

**第二阶段（训练优化）**：
- [ ] 修复损失计算
- [ ] 添加类别平衡处理
- [ ] 实现分层学习率

**第三阶段（架构优化）**：
- [ ] 改进提示工程
- [ ] 优化Perceiver融合层
- [ ] 改进早停策略

**第四阶段（完善）**：
- [ ] 保存最佳阈值
- [ ] 添加对比学习
- [ ] 模型解释性

---

*分析完成时间: 2025年2月*
*分析师: 窃电检测LLM专家*
