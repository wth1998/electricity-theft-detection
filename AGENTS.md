# 窃电检测多模态大模型项目 (AXIS风格)

## 项目概述

这是一个基于 **AXIS** 论文架构的窃电检测系统，结合了时间序列编码器和大型语言模型（LLM）。项目通过多模态融合实现端到端的电力用户异常行为检测。

### 核心技术栈

- **时间序列编码器**: 基于AXIS论文，采用RoPE位置编码、RMSNorm归一化、LlamaMLP前馈网络
- **融合机制**: Perceiver跨注意力融合时间序列特征与LLM词嵌入
- **LLM集成**: 支持Qwen3等开源大模型进行端到端分类
- **预训练策略**: 掩码重建 + 异常检测双任务预训练

---

## 项目结构

```
.
├── models/                          # 核心模型实现
│   ├── ts_encoder_axis.py          # AXIS风格时间序列编码器
│   ├── perceiver_fusion.py         # Perceiver跨注意力融合模块
│   ├── main_axis_improved.py       # 整体感知层和数据集
│   └── agent_axis.py               # 完整Agent（含LLM）
├── layers/                          # 基础层组件
│   ├── axis_RoPE.py                # RoPE旋转位置编码
│   ├── RevIN.py                    # 可逆实例归一化
│   └── axis_mamba.py               # Mamba层（可选）
├── scripts/                         # 训练脚本
│   ├── train_pretrain_axis.py      # 预训练脚本
│   ├── train_finetune_axis.py      # 微调脚本
│   └── test_axis_improved.py       # 测试脚本
├── utils/                           # 工具函数
│   ├── metrics.py                  # 评估指标（AUC、MAP@40等）
│   ├── visualization.py            # 可视化工具
│   └── debug_utils.py              # 调试工具
└── AGENTS.md                        # 本文件
```

---

## 核心组件详解

### 1. 时间序列编码器 (`ts_encoder_axis.py`)

**功能**: 将电力负荷时间序列编码为特征表示

**输入**: `(B, seq_len, num_features)` - 批量时间序列数据
**输出**: `(B, seq_len, num_features, d_proj)` - 局部特征嵌入

**关键技术**:
- **Patching**: 将序列切分为patch（默认patch_size=6）
- **RoPE (Rotary Position Embedding)**: 旋转位置编码，更好地捕获时间关系
- **RMSNorm**: 均方根归一化，替代LayerNorm
- **LlamaMLP**: Gated MLP结构，增强表达能力
- **Multi-head Attention**: 标准多头自注意力，非因果模式

**预训练头** (`PretrainHeads`):
- 掩码重建头: 预测被掩码位置的原始值
- 异常检测头: 二分类（正常/异常）

### 2. Perceiver融合模块 (`perceiver_fusion.py`)

**功能**: 将时间序列特征映射到LLM的词嵌入空间

**输入**: 
- `local_embeddings`: `(B, seq_len, num_features, d_proj)` 时间序列特征
- `llm_embeds`: LLM词嵌入权重

**输出**: `(B, num_tokens, llm_hidden_size)` Soft Prompts

**实现**:
- `SimplePerceiverFusion`: 简化版本，直接使用Cross-Attention
- 时间压缩: 使用AdaptiveAvgPool1d将seq_len压缩到num_local_tokens
- 固定提示: 可学习的固定token嵌入
- Cross-Attention: 查询为固定提示+局部提示，键值为时间序列特征

### 3. 整体感知层 (`main_axis_improved.py`)

**核心类**:
- `H3_PerceptionLayerAXIS`: 整合数值流编码器和Perceiver融合
- `ElectricityDatasetAXIS`: 支持预训练和微调的数据集
- `ModelConfig`: 预定义配置（SMALL/BASE/MEDIUM/LARGE）

**数据处理**:
- 使用用户历史均值进行标准化（而非单日Z-score）
- 生成丰富的统计特征（变异系数、峰度、偏度、零值比例等）
- 构建增强版文本提示，包含风险信号分析

**输出维度**: `(B, seq_len, num_features, d_proj)`

### 4. Agent (`agent_axis.py`)

**功能**: 完整的多模态窃电检测Agent

**组件**:
- `H3_Agent_AXIS`: 整合感知层和LLM
- LLM集成: 支持Qwen3等模型，支持4-bit量化
- 提示构建: Llama-3格式指令，包含窃电/正常指标指导

**推理流程**:
1. 时间序列 → 感知层编码 → Soft Prompts
2. 文本提示 → Tokenizer → Text Embeddings
3. 拼接: [Soft Prompts] + [Text Embeddings]
4. LLM生成分类结果

---

## 训练流程

### 阶段1: 预训练 (`train_pretrain_axis.py`)

**目标**: 学习时间序列的通用表示

**任务**:
1. **掩码重建 (Mask Reconstruction)**:
   - 随机掩码15%的时间点
   - 预测原始值
   - 损失函数: MSE

2. **异常检测 (Anomaly Detection)**:
   - 基于统计特征生成伪异常标签（偏离均值2个标准差）
   - 二分类损失: CrossEntropy

**数据**: 使用无标签的正常样本（`File1_train.csv`）

**命令示例**:
```bash
python scripts/train_pretrain_axis.py \
    --config MEDIUM \
    --epochs 50 \
    --batch-size 16 \
    --use-first-n 5000
```

### 阶段2: 微调 (`train_finetune_axis.py`)

**目标**: 端到端窃电检测分类

**数据**: 带标签数据（`File1_test_daily_flag_4_1.csv`）

**数据划分**:
- 按用户划分（避免数据泄露）
- 训练集: 70%用户
- 验证集: 30%用户

**训练配置**:
- 学习率: 2e-4
- 优化器: AdamW
- 调度器: CosineAnnealingWarmRestarts
- 早停: 基于验证损失

**命令示例**:
```bash
python scripts/train_finetune_axis.py \
    --config MEDIUM \
    --epochs 20 \
    --pretrain-ckpt checkpoints_pretrain_axis/pretrain_medium_best.pth
```

---

## 评估指标 (`utils/metrics.py`)

### 主要指标

1. **AUC**: ROC曲线下面积
2. **MAP@40**: 前40个预测的平均精确率
3. **WF1 (Weighted F1)**: 加权F1分数
4. **WP (Weighted Precision)**: 加权精确率
5. **WR (Weighted Recall)**: 加权召回率

### 辅助指标

- 混淆矩阵 (TP/FP/FN/TN)
- 类别级精确率/召回率/F1
- Top-K预测分析

---

## 模型配置

### 预定义配置

```python
ModelConfig.SMALL   # 轻量级，快速实验
ModelConfig.BASE    # 基础配置
ModelConfig.MEDIUM  # 推荐配置（默认）
ModelConfig.LARGE   # 大模型，高精度
```

### MEDIUM配置示例

```python
{
    'd_model': 512,           # 模型维度
    'd_proj': 256,            # 投影维度（输出）
    'patch_size': 6,          # Patch大小
    'num_layers': 8,          # Transformer层数
    'num_heads': 8,           # 注意力头数
    'num_local_tokens': 20,   # 局部提示token数
    'num_fixed_tokens': 10,   # 固定提示token数
    'dropout': 0.15
}
```

---

## 关键设计决策

### 1. 标准化策略

**问题**: 单日Z-score会错误地放大低用电量用户的平坦曲线

**解决方案**: 使用用户历史统计量进行标准化
```python
# 用户历史均值和标准差
mean = u_stats['mean']
std = u_stats['std']
vals_normalized = (vals - mean) / std
```

### 2. 提示构建

**结构**:
- 系统提示: 窃电检测专家角色定义
- 用户输入: 分析上下文 + 指令
- 关键规则: 仅输出"Theft"或"Normal"

### 3. 损失计算

- Soft Prompts部分不计算损失（前缀标签设为-100）
- 只计算Text Embeddings部分的生成损失

---

## 依赖环境

### Python包

```
torch>=2.0.0
transformers>=4.30.0
pandas
numpy
scikit-learn
matplotlib
seaborn
```

### 硬件要求

- **GPU**: 推荐至少16GB显存（用于7B参数LLM）
- **内存**: 32GB+ RAM
- **存储**: 10GB+ 用于模型检查点

### LLM模型

默认使用 `/root/rivermind-data/qwen3_1.7B`
可通过修改 `agent_axis.py` 中的路径使用其他模型

---

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 准备数据

```bash
# 数据集应放在 dataset/ 目录
dataset/
├── File1_train.csv              # 无标签训练数据
├── File1_test_daily_flag_4_1.csv # 带标签测试数据
└── user_historical_stats.json   # 用户历史统计
```

### 3. 预训练

```bash
python scripts/train_pretrain_axis.py --config MEDIUM --quick
```

### 4. 微调

```bash
python scripts/train_finetune_axis.py --config MEDIUM --quick
```

### 5. 测试

```bash
python scripts/test_axis_improved.py \
    --ckpt checkpoints_finetune_axis/finetune_medium_best.pth \
    --debug
```

---

## 调试工具

项目包含完善的调试支持：

```bash
# 启用调试模式
python scripts/train_finetune_axis.py --debug

# 将打印：
# - 数据输入检查
# - 编码器输出统计
# - 融合层输出
# - 每个样本的预测过程
```

---

## 扩展方向

1. **多变量支持**: 当前默认num_features=1，可扩展为多变时间序列
2. **Mamba层**: `layers/axis_mamba.py` 提供可选的Mamba替换Transformer
3. **其他LLM**: 支持Llama、ChatGLM等模型
4. **联邦学习**: 支持分布式训练保护用户隐私

---

## 参考文献

- AXIS: A Framework for Extracting Structured Information from Time Series
- PatchTST: A Time Series is Worth 64 Words
- LLaMA: Open and Efficient Foundation Language Models
- Perceiver: General Perception with Iterative Attention

---

## 维护信息

- **创建时间**: 2025年2月
- **主要维护者**: 项目团队
- **许可证**: 内部使用

---

## 常见问题

**Q: 如何选择模型配置？**
A: 快速实验用SMALL，生产环境推荐MEDIUM，追求高精度用LARGE。

**Q: 预训练是否必须？**
A: 推荐先预训练再微调，可提升10-15%性能。也可从头训练。

**Q: 如何处理类别不平衡？**
A: 使用加权指标（WF1/WP/WR），关注MAP@K而非纯准确率。

**Q: 推理速度慢怎么办？**
A: 启用4-bit量化（`load_in_4bit=True`），减少num_local_tokens。
