import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
import json
import argparse
import numpy as np
import pandas as pd
import random
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.main_axis_improved import (
    H3_PerceptionLayerAXIS,
    ElectricityDatasetAXIS,
    ModelConfig,
)
from models.agent_axis import H3_Agent_AXIS
from utils.seed_utils import set_seed, seed_worker
from utils.losses import FocalLoss, CombinedLoss
from utils.metrics import TheftDetectionMetrics


class ElectricityDatasetWithLabel(ElectricityDatasetAXIS):
    """带标签的数据集，支持按用户划分"""

    def __init__(
        self,
        csv_file,
        user_stats_path,
        config=None,
        max_samples=None,
        user_ids=None,
        mode="train",
        **kwargs,
    ):
        super().__init__(
            csv_file, user_stats_path, for_pretrain=False, mode=mode, **kwargs
        )

        # 如果指定了用户列表，只保留这些用户的数据
        if user_ids is not None:
            mask = pd.Series(self.user_ids).isin(user_ids).values
            self.df = self.df[mask].reset_index(drop=True)
            # 使用父类初始化时确定的id_idx，而不是硬编码第0列
            self.user_ids = self.df.iloc[:, self.id_idx].values.astype(str)
            self.data_values = self.data_values[mask]
            if self.labels is not None:
                self.labels = self.labels[mask]
            print(
                f"[{mode}] 筛选后: {len(self.df)} 条数据, {len(set(self.user_ids))} 个用户"
            )

        # 限制样本数
        if max_samples is not None and max_samples < len(self.df):
            print(f"[微调] 只使用 {max_samples}/{len(self.df)} 条数据")
            indices = torch.randperm(len(self.df))[:max_samples].numpy()
            self.df = self.df.iloc[indices].reset_index(drop=True)
            # 使用父类初始化时确定的id_idx，而不是硬编码第0列
            self.user_ids = self.df.iloc[:, self.id_idx].values.astype(str)
            self.data_values = self.data_values[indices]
            if self.labels is not None:
                self.labels = self.labels[indices]

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def split_users_by_ratio(csv_file, train_ratio=0.7, seed=42):
    """
    按用户划分训练集和验证集
    避免同一用户的数据同时出现在训练和测试中（数据泄露）
    """
    df = pd.read_csv(csv_file)
    # 假设第一列是用户ID，转换为字符串以匹配数据集中的类型
    unique_users = df.iloc[:, 0].astype(str).unique()
    print(f"数据集中共有 {len(unique_users)} 个唯一用户")

    np.random.seed(seed)
    np.random.shuffle(unique_users)

    n_train = int(len(unique_users) * train_ratio)
    train_users = unique_users[:n_train]
    val_users = unique_users[n_train:]

    print(f"训练集用户: {len(train_users)} 个")
    print(f"验证集用户: {len(val_users)} 个")

    return train_users, val_users


def evaluate_on_validation(agent, val_loader, device, max_batches=None):
    """
    在验证集上评估模型性能

    返回各项指标：AUC、MAP@40、WF1、F1_Theft等
    """
    agent.eval()
    metrics = TheftDetectionMetrics()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches and batch_idx >= max_batches:
                break

            targets = batch["label"].to(device)
            target_texts = ["Theft" if t == 1 else "Normal" for t in targets]
            instructions = ["Analyze this user's electricity usage pattern."] * len(
                targets
            )

            # 推理
            responses, _, theft_scores = agent.generate(
                batch, instructions, return_scores=True, debug=False
            )

            # 解析预测结果
            for i in range(len(targets)):
                true_label = "Theft" if targets[i] == 1 else "Normal"
                pred_text = responses[i].strip().lower()
                pred_label = "Theft" if "theft" in pred_text else "Normal"
                score = theft_scores[i] if theft_scores else 0.5

                metrics.update(
                    true_label, pred_label, score, user_id=batch_idx * len(targets) + i
                )
                all_scores.append(score)
                all_labels.append(targets[i].item())

    results = metrics.compute()

    # 计算AUC
    if len(set(all_labels)) > 1:
        results["auc"] = roc_auc_score(all_labels, all_scores)
    else:
        results["auc"] = 0.5

    # 计算F1分数
    preds = [1 if s > 0.5 else 0 for s in all_scores]
    results["f1_macro"] = f1_score(all_labels, preds, average="macro", zero_division=0)
    results["f1_weighted"] = f1_score(
        all_labels, preds, average="weighted", zero_division=0
    )

    agent.train()
    return results


def compute_composite_score(results):
    """
    计算综合评分用于早停

    加权组合多个指标：
    - AUC: 30%
    - MAP@40: 30%
    - WF1: 20%
    - F1_Theft: 20% (特别关注窃电类别的召回)
    """
    score = (
        0.3 * results.get("auc", 0)
        + 0.3 * results.get("map@40", 0)
        + 0.2 * results.get("wf1", 0)
        + 0.2 * results.get("f1_theft", 0)
    )
    return score


def finetune_model(
    model_config_name="MEDIUM",
    epochs=10,
    lr=2e-4,
    batch_size=4,
    accumulation_steps=2,
    patience=4,
    max_samples=None,
    pretrain_ckpt=None,
    freeze_encoder=False,
    checkpoint_dir="checkpoints_finetune_axis",
    seed=42,
    use_weighted_sampler=True,
    use_focal_loss=True,
    encoder_lr_ratio=0.1,  # 编码器学习率相对于整体学习率的比例
    use_tensorboard=True,
    eval_every=1,  # 每隔多少个epoch评估一次验证集
    use_layerwise_lr=True,  # 是否使用分层学习率
):
    """
    端到端微调训练

    Args:
        model_config_name: 模型配置名称
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批次大小
        accumulation_steps: 梯度累积步数
        patience: 早停耐心值
        max_samples: 最大样本数（快速实验）
        pretrain_ckpt: 预训练检查点路径（可选）
        freeze_encoder: 是否冻结编码器
        checkpoint_dir: 检查点保存目录
        seed: 随机种子
        use_weighted_sampler: 是否使用加权采样处理类别不平衡
        use_focal_loss: 是否使用Focal Loss
        encoder_lr_ratio: 编码器学习率比例（分层学习率）
        use_tensorboard: 是否使用TensorBoard
        eval_every: 验证集评估频率
        use_layerwise_lr: 是否使用分层学习率
    """
    # 【修复】设置随机种子
    set_seed(seed)

    # 创建TensorBoard writer
    writer = None
    if use_tensorboard:
        log_dir = f"runs/finetune_{model_config_name}_{int(time.time())}"
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[TensorBoard] 日志目录: {log_dir}")

    config = getattr(ModelConfig, model_config_name)

    print(f"\n{'=' * 60}")
    print(f"AXIS风格端到端微调")
    print(f"{'=' * 60}")
    print(f"模型配置: {model_config_name}")
    print(f"配置详情: {config}")
    print(f"批次大小: {batch_size} (累积{accumulation_steps}步)")
    if pretrain_ckpt:
        print(f"预训练权重: {pretrain_ckpt}")
        print(f"冻结编码器: {freeze_encoder}")
    else:
        print("从头训练（无预训练权重）")
    print(f"{'=' * 60}\n")

    # 按用户划分训练集和验证集（避免数据泄露）
    print("[数据划分] 按用户划分训练集和验证集...")
    train_users, val_users = split_users_by_ratio(
        "dataset/File1_test_daily_flag_4_1.csv", train_ratio=0.7, seed=42
    )

    # 训练集
    train_dataset = ElectricityDatasetWithLabel(
        "dataset/File1_test_daily_flag_4_1.csv",
        "dataset/user_historical_stats.json",
        config=config,
        max_samples=max_samples,
        user_ids=train_users,
        mode="train",
    )

    # 验证集
    val_dataset = ElectricityDatasetWithLabel(
        "dataset/File1_test_daily_flag_4_1.csv",
        "dataset/user_historical_stats.json",
        config=config,
        user_ids=val_users,
        mode="val",
    )

    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(val_dataset)} 条\n")

    # 【修复】计算类别权重，处理类别不平衡（窃电:正常 ≈ 1:9）
    train_labels = train_dataset.labels
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=train_labels
    )
    print(f"类别权重: {dict(zip(np.unique(train_labels), class_weights))}")

    # 【修复】使用WeightedRandomSampler处理类别不平衡
    if use_weighted_sampler:
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        print("✓ 使用加权随机采样器")
        train_shuffle = False  # 使用sampler时不能shuffle
    else:
        sampler = None
        train_shuffle = True

    # 【修复】添加worker seed确保DataLoader可复现
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 模型
    llm_path = "/root/rivermind-data/qwen3_1.7B"

    perception_layer = H3_PerceptionLayerAXIS(llm_dim=2048, **config)
    agent = H3_Agent_AXIS(
        llm_path, perception_layer=perception_layer, load_in_4bit=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent.to(device)

    # 加载预训练权重
    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        print(f"\n加载预训练权重: {pretrain_ckpt}")
        checkpoint = torch.load(pretrain_ckpt, map_location=device)

        # 只加载数值流部分（使用 strict=False 忽略预训练头）
        agent.perception.numerical_stream.load_state_dict(
            checkpoint["numerical_stream"], strict=False
        )
        print("✓ 预训练权重加载成功（编码器部分）")

        if freeze_encoder:
            print("冻结编码器参数")
            for param in agent.perception.numerical_stream.parameters():
                param.requires_grad = False

    # 统计参数量
    total_params = sum(p.numel() for p in agent.perception.parameters())
    trainable_params = sum(
        p.numel() for p in agent.perception.parameters() if p.requires_grad
    )
    print(f"\n感知层参数量:")
    print(f"  总数: {total_params:,}")
    print(f"  可训练: {trainable_params:,}\n")

    # 【优化】分层学习率：编码器使用较小学习率，融合层使用标准学习率
    if use_layerwise_lr and not freeze_encoder:
        param_groups = [
            {
                "params": agent.perception.numerical_stream.parameters(),
                "lr": lr * encoder_lr_ratio,  # 编码器学习率较小
                "name": "encoder",
            },
            {
                "params": agent.perception.fusion.parameters(),
                "lr": lr,  # 融合层使用标准学习率
                "name": "fusion",
            },
        ]
        optimizer = AdamW(param_groups, weight_decay=0.01)
        print(
            f"✓ 使用分层学习率: encoder_lr={lr * encoder_lr_ratio:.2e}, fusion_lr={lr:.2e}"
        )
    else:
        # 优化器 - 只优化可训练参数
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, agent.perception.parameters()),
            lr=lr,
            weight_decay=0.01,
        )
        print(f"✓ 使用统一学习率: lr={lr:.2e}")

    # 学习率调度
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=epochs // 3, T_mult=2)

    agent.train()

    # 训练循环
    best_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        # 训练阶段
        for batch_idx, batch in enumerate(train_loader):
            targets = batch["label"].to(device)
            target_texts = ["Theft" if t == 1 else "Normal" for t in targets]
            instructions = ["Analyze this user's electricity usage pattern."] * len(
                targets
            )

            # 前向传播
            loss = agent.forward(
                batch, instructions, labels=target_texts, batch_idx=batch_idx
            )
            loss = loss / accumulation_steps

            # 反向传播
            loss.backward()

            # 梯度累积更新
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(
                train_loader
            ) - 1:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, agent.perception.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / num_batches
                print(
                    f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {avg_loss:.4f}"
                )

        # 验证阶段
        agent.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                targets = batch["label"].to(device)
                target_texts = ["Theft" if t == 1 else "Normal" for t in targets]
                instructions = ["Analyze this user's electricity usage pattern."] * len(
                    targets
                )

                loss = agent.forward(batch, instructions, labels=target_texts)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else float("inf")
        agent.train()

        # 更新学习率
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        # 【修复】删除重复的打印和lr获取
        print(
            f"\n=== Epoch {epoch} 完成 | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e} ==="
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                "lr": current_lr,
            }
        )

        # 保存检查点
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "perception": agent.perception.state_dict(),
            "epoch": epoch,
            "loss": avg_loss,
            "config": config,
            "config_name": model_config_name,
        }

        checkpoint_name = (
            f"finetune_{model_config_name.lower()}_ep{epoch}_loss{avg_loss:.4f}.pth"
        )
        torch.save(checkpoint, f"{checkpoint_dir}/{checkpoint_name}")

        # 早停检查（基于验证损失）
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                checkpoint,
                f"{checkpoint_dir}/finetune_{model_config_name.lower()}_best.pth",
            )
            print(f"✓ 新的最佳模型 (val_loss={best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"早停计数: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n早停触发，训练结束")
                break

    # 保存训练历史
    with open(f"{checkpoint_dir}/history_{model_config_name.lower()}.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"微调完成 | 最佳 Loss: {best_loss:.4f}")
    print(f"检查点保存于: {checkpoint_dir}")
    print(f"{'=' * 60}")

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune AXIS-style model (默认从预训练权重开始)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # 显示默认值
    )
    parser.add_argument(
        "--config",
        type=str,
        default="MEDIUM",
        choices=["SMALL", "BASE", "MEDIUM", "LARGE"],
        help="模型配置",
    )
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument(
        "--accumulation-steps", type=int, default=2, help="梯度累积步数"
    )
    parser.add_argument("--patience", type=int, default=4, help="早停耐心值")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="限制样本数（快速实验用）"
    )
    parser.add_argument(
        "--pretrain-ckpt",
        type=str,
        default="auto",
        help="预训练权重路径（默认auto自动查找，None从头训练）",
    )
    parser.add_argument("--freeze-encoder", action="store_true", help="冻结编码器参数")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_finetune_axis",
        help="检查点保存目录",
    )
    parser.add_argument("--quick", action="store_true", help="快速模式：1000样本，5轮")
    parser.add_argument(
        "--debug", action="store_true", help="启用调试模式（打印各阶段中间结果）"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--no-weighted-sampler", action="store_true", help="禁用加权采样器（默认启用）"
    )
    parser.add_argument(
        "--no-focal-loss", action="store_true", help="禁用Focal Loss（默认启用）"
    )

    args = parser.parse_args()

    # 设置全局调试开关
    if args.debug:
        from utils.debug_utils import DebugLogger

        DebugLogger.set_enabled(True)
        print("\n[调试模式已启用] 将打印各阶段中间结果\n")

    # 快速模式
    max_samples = args.max_samples
    epochs = args.epochs
    if args.quick:
        max_samples = 4000
        epochs = 5
        print(f"[快速模式] 使用 {max_samples} 条数据，训练 {epochs} 个 epoch")

    # 自动确定预训练权重路径
    pretrain_ckpt = args.pretrain_ckpt
    if pretrain_ckpt == "auto":
        # 根据配置自动构建路径
        config_lower = args.config.lower()
        pretrain_ckpt = f"checkpoints_pretrain_axis/pretrain_{config_lower}_best.pth"
        if not os.path.exists(pretrain_ckpt):
            print(f"警告: 默认预训练权重不存在: {pretrain_ckpt}")
            print("将从头开始训练。如需使用预训练，请先运行 train_pretrain_axis.py")
            pretrain_ckpt = None
    elif pretrain_ckpt.lower() == "none":
        pretrain_ckpt = None

    finetune_model(
        model_config_name=args.config,
        epochs=epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        patience=args.patience,
        max_samples=max_samples,
        pretrain_ckpt=pretrain_ckpt,
        freeze_encoder=args.freeze_encoder,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        use_weighted_sampler=not args.no_weighted_sampler,
        use_focal_loss=not args.no_focal_loss,
    )
