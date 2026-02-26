# 代码修复总结

## 修复完成的关键Bug

### ✅ 1. 随机种子设置（可复现性）

**问题**: 训练结果每次运行都不同，无法复现

**修复**:
- 创建了 `utils/seed_utils.py` 统一随机种子设置
- 在 `train_pretrain_axis.py` 和 `train_finetune_axis.py` 中添加完整随机种子设置
- 包括：`torch.cuda.manual_seed_all`、`cudnn.deterministic`、`PYTHONHASHSEED`
- DataLoader添加 `worker_init_fn` 和 `generator` 确保数据加载可复现

**使用**:
```bash
python scripts/train_pretrain_axis.py --seed 42
python scripts/train_finetune_axis.py --seed 42
```

### ✅ 2. Label对齐逻辑（最关键）

**问题**: `agent_axis.py` 中label构建逻辑错误，导致模型无法正确学习

**修复**:
- 重构了 `construct_prompt` 函数，确保prompt长度计算正确
- 修复了 `forward` 中的label构建逻辑：
  ```python
  # 之前（错误）: text_labels[i, p_len:] = tokens.input_ids[i, p_len:]
  
  # 修复后（正确）:
  start_idx = soft_prompt_len + p_len
  final_labels[i, start_idx:start_idx + target_len] = tokens.input_ids[i, p_len:actual_len]
  ```

**影响**: 这是训练有效性的关键，修复后模型可以正确学习

### ✅ 3. 预训练伪标签生成

**问题**: 使用当日统计量生成伪异常标签，导致低用电量用户产生大量伪异常

**修复**:
- 使用用户历史统计量（而非当日统计）生成伪标签
  ```python
  # 修复前
  mean = vals.mean()
  std = vals.std() + 1e-6
  
  # 修复后
  u_stats = self.user_stats.get(uid, {...})
  historical_mean = u_stats['mean']
  historical_std = u_stats['std'] + 1e-6
  ```

**影响**: 预训练质量提升，伪标签更准确

### ✅ 4. 类别不平衡处理（1:9）

**问题**: 窃电样本远少于正常样本，模型倾向预测"正常"

**修复**:
- 添加 `sklearn.utils.class_weight` 计算类别权重
- 使用 `WeightedRandomSampler` 进行过采样
- 默认启用，可通过 `--no-weighted-sampler` 禁用

**使用**:
```bash
# 启用加权采样（默认）
python scripts/train_finetune_axis.py

# 禁用加权采样
python scripts/train_finetune_axis.py --no-weighted-sampler
```

## 新增文件

1. **`.gitignore`** - Git忽略规则，防止提交模型和数据文件
2. **`requirements.txt`** - Python依赖清单
3. **`utils/seed_utils.py`** - 随机种子设置工具函数

## 修改的文件

1. **`scripts/train_pretrain_axis.py`**
   - 添加随机种子设置
   - 修复预训练伪标签生成逻辑
   - 添加 `--seed` 参数

2. **`scripts/train_finetune_axis.py`**
   - 添加随机种子设置
   - 添加类别权重计算和加权采样
   - 添加 `--seed`、`--no-weighted-sampler` 参数

3. **`models/agent_axis.py`**
   - 修复 `construct_prompt` 函数
   - 修复 `forward` 中的label构建逻辑

## 推荐的训练命令

### 预训练（阶段1）
```bash
python scripts/train_pretrain_axis.py \
    --config MEDIUM \
    --epochs 50 \
    --batch-size 16 \
    --seed 42
```

### 微调（阶段2）
```bash
python scripts/train_finetune_axis.py \
    --config MEDIUM \
    --epochs 20 \
    --pretrain-ckpt checkpoints_pretrain_axis/pretrain_medium_best.pth \
    --seed 42
```

## 后续优化建议（可选）

虽然主要bug已修复，但以下优化可进一步提升效果：

1. **Focal Loss** - 进一步处理类别不平衡（已添加参数但未实现）
2. **分层学习率** - 编码器使用较小学习率
3. **早停策略优化** - 基于WF1/MAP@K而非loss
4. **验证集指标监控** - 实时跟踪AUC、F1等
5. **TensorBoard集成** - 便于可视化训练过程

## 测试修复

修复后应该能够：
- ✅ 多次运行得到相似结果（随机种子有效）
- ✅ 损失正常下降（Label对齐正确）
- ✅ 模型能够学习到窃电模式（类别权重有效）
- ✅ 预训练收敛稳定（伪标签质量提升）

---

**修复时间**: 2025年2月  
**状态**: 核心Bug已修复，可开始训练
