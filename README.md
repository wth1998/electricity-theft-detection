# 窃电检测多模态大模型 (AXIS风格)

基于AXIS架构的端到端窃电检测系统，结合时间序列编码器和大语言模型(LLM)。

## 快速开始

### 环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/wth1998/electricity-theft-detection.git
cd electricity-theft-detection

# 2. 安装依赖
pip install -r requirements.txt
```

### 数据准备

```
dataset/
├── File1_train.csv              # 无标签训练数据
├── File1_test_daily_flag_4_1.csv # 带标签训练数据
└── user_historical_stats.json   # 用户历史统计
```

### 训练流程

#### 阶段1: 预训练（自监督）

```bash
python scripts/train_pretrain_axis.py \
    --config MEDIUM \
    --epochs 50 \
    --batch-size 16 \
    --seed 42
```

**关键参数说明**:
- `--config`: 模型配置 [SMALL|BASE|MEDIUM|LARGE]
- `--use-first-n`: 使用多少样本（默认5000）
- `--seed`: 随机种子（确保可复现）

#### 阶段2: 微调（监督学习）

```bash
python scripts/train_finetune_axis.py \
    --config MEDIUM \
    --epochs 20 \
    --pretrain-ckpt checkpoints_pretrain_axis/pretrain_medium_best.pth \
    --seed 42 \
    --batch-size 4 \
    --lr 2e-4
```

**关键参数说明**:
- `--pretrain-ckpt`: 预训练权重路径（auto自动查找）
- `--batch-size`: 批次大小（2x3090建议4）
- `--lr`: 学习率（默认2e-4）
- `--seed`: 随机种子

#### 高级参数

```bash
# 禁用加权采样（如果类别不平衡不严重）
--no-weighted-sampler

# 快速测试（1000样本，5轮）
--quick

# 启用调试模式
--debug
```

#### 阶段3: 测试

```bash
python scripts/test_axis_improved.py \
    --config MEDIUM \
    --checkpoint checkpoints_finetune_axis/finetune_medium_best.pth \
    --debug
```

## 修复的关键Bug

### ✅ 1. Label对齐逻辑（训练有效性）
- **问题**: label构建错误导致模型无法学习
- **修复**: 重构`forward`函数，正确对齐soft prompt和label

### ✅ 2. 随机种子设置（可复现性）
- **问题**: 训练结果每次运行都不同
- **修复**: 完整设置所有随机种子（包括CUDA、DataLoader worker）

### ✅ 3. 预训练伪标签（数据质量）
- **问题**: 使用当日统计生成伪标签，质量差
- **修复**: 使用用户历史统计量生成伪标签

### ✅ 4. 类别不平衡（1:9）
- **问题**: 窃电样本少，模型倾向预测"正常"
- **修复**: 
  - 添加WeightedRandomSampler过采样
  - Focal Loss支持（utils/losses.py）

## 性能优化

### ✅ 1. 分层学习率
- 编码器: 0.1 * lr（较小，防止破坏预训练特征）
- 融合层: 1.0 * lr（标准学习率）

### ✅ 2. 精简文本提示
- 从~150 tokens减少到~50 tokens
- 保留关键指标：用电量、零值比例、稳定模式、异常数、风险信号

### ✅ 3. 验证集指标监控
- AUC、MAP@40、WF1、F1_Theft
- 综合评分用于早停（AUC30% + MAP@4030% + WF120% + F1_Theft20%）

### ✅ 4. TensorBoard支持
- 自动记录loss、学习率、各指标变化
- 查看命令: `tensorboard --logdir runs/`

## 模型配置

| 配置 | d_model | d_proj | layers | heads | 参数量 | 适用场景 |
|------|---------|--------|--------|-------|--------|----------|
| SMALL | 256 | 128 | 4 | 4 | ~2M | 快速实验 |
| BASE | 512 | 256 | 6 | 8 | ~8M | 标准训练 |
| **MEDIUM** | **512** | **256** | **8** | **8** | **~12M** | **推荐** |
| LARGE | 768 | 384 | 10 | 12 | ~25M | 高精度 |

**MEDIUM配置推荐用于2x3090（48GB显存）**

## 硬件要求

- **最低**: 1x GPU 16GB VRAM
- **推荐**: 2x GPU 48GB VRAM (2x RTX 3090)
- **内存**: 32GB+ RAM
- **存储**: 10GB+ for checkpoints

## 预期性能

在修复后的代码上，预期可以达到：

| 指标 | 修复前 | 修复后（目标） |
|------|--------|---------------|
| AUC | ~0.65 | **~0.85+** |
| MAP@40 | ~0.20 | **~0.50+** |
| WF1 | ~0.60 | **~0.80+** |
| 窃电召回率 | ~50% | **~80%+** |

## 项目结构

```
.
├── models/                      # 核心模型
│   ├── ts_encoder_axis.py      # 时间序列编码器
│   ├── perceiver_fusion.py     # 融合模块
│   ├── main_axis_improved.py   # 主模型和数据集
│   └── agent_axis.py           # Agent（含LLM）
├── scripts/                     # 训练脚本
│   ├── train_pretrain_axis.py  # 预训练
│   ├── train_finetune_axis.py  # 微调
│   └── test_axis_improved.py   # 测试
├── utils/                       # 工具
│   ├── metrics.py              # 评估指标
│   ├── losses.py               # Focal Loss等
│   ├── seed_utils.py           # 随机种子设置
│   └── visualization.py        # 可视化
├── layers/                      # 基础层
│   ├── axis_RoPE.py            # RoPE位置编码
│   └── RevIN.py                # 可逆实例归一化
└── AGENTS.md                   # 详细架构说明
```

## 训练技巧

### 1. 快速验证修复是否有效
```bash
# 使用快速模式（1000样本，5轮）
python scripts/train_finetune_axis.py --quick --config SMALL
# 观察loss是否下降，结果是否可复现
```

### 2. 监控训练过程
```bash
# 终端1: 启动TensorBoard
tensorboard --logdir runs/ --port 6006

# 终端2: 训练
python scripts/train_finetune_axis.py --config MEDIUM
# 在浏览器中打开 http://localhost:6006
```

### 3. 处理训练不稳定
```bash
# 降低学习率
--lr 1e-4

# 增大批次大小（如果显存允许）
--batch-size 8

# 使用更小模型
--config SMALL
```

### 4. 处理过拟合
```bash
# 启用编码器冻结（如果预训练质量好）
--freeze-encoder

# 减少训练轮数
--epochs 10
```

## 常见问题

**Q: 损失不下降？**
A: 检查：1) Label是否正确对齐（batch 0会打印目标词） 2) 随机种子是否设置 3) 学习率是否过高

**Q: 显存不足？**
A: 使用 `--config SMALL`，或减小 `--batch-size`，或在agent中启用 `load_in_4bit=True`

**Q: 结果不稳定？**
A: 确保使用相同的 `--seed`，并且没有使用 `WeightedRandomSampler`（可用 `--no-weighted-sampler`禁用）

**Q: 窃电召回率低？**
A: 这是类别不平衡的典型问题，尝试：1) 降低阈值 2) 增加训练轮数 3) 使用Focal Loss

## 引用

如果此代码对您的研究有帮助，请引用：

```bibtex
@software{electricity_theft_detection,
  title={Electricity Theft Detection with AXIS and LLM},
  author={Your Name},
  year={2025},
  url={https://github.com/wth1998/electricity-theft-detection}
}
```

## 许可

内部使用

---

**最后更新**: 2025年2月  
**版本**: v1.0 (修复版)
