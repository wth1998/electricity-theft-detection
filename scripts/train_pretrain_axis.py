import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
from torch.optim import AdamW
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.main_axis_improved import (
    H3_PerceptionLayerAXIS,
    ElectricityDatasetAXIS,
    ModelConfig
)


class ElectricityDatasetPretrain(ElectricityDatasetAXIS):
    """预训练专用数据集"""
    
    def __init__(self, csv_file, user_stats_path, config=None, max_samples=None, 
             use_first_n=None, mask_ratio=0.15, **kwargs):
        """
        Args:
            max_samples: 随机采样指定数量的样本
            use_first_n: 按顺序使用前N条样本（与max_samples互斥）
        """
        # 从 kwargs 中移除 mode，防止与显式传递的 mode='train' 冲突
        kwargs.pop('mode', None)
    
        # 调用父类初始化（max_samples在子类中处理）
        super().__init__(csv_file, user_stats_path, mode='train', for_pretrain=False, 
                        **kwargs)
        self.mask_ratio = mask_ratio
        
        # 方式1: 按顺序使用前N条（更稳定，可复现）
        if use_first_n is not None and use_first_n < len(self.df):
            print(f"[预训练] 按顺序使用前 {use_first_n}/{len(self.df)} 条数据")
            self.df = self.df.iloc[:use_first_n].reset_index(drop=True)
            self.user_ids = self.df.iloc[:, 0].values.astype(str)
            self.data_values = self.data_values[:use_first_n]
        # 方式2: 随机采样（用于快速实验）
        elif max_samples is not None and max_samples < len(self.df):
            print(f"[预训练] 随机采样 {max_samples}/{len(self.df)} 条数据")
            indices = torch.randperm(len(self.df))[:max_samples].numpy()
            self.df = self.df.iloc[indices].reset_index(drop=True)
            self.user_ids = self.df.iloc[:, 0].values.astype(str)
            self.data_values = self.data_values[indices]

    def __getitem__(self, idx):
        vals = self.data_values[idx]
        
        # 转换为tensor
        x_seq_tensor = torch.tensor(vals, dtype=torch.float32).unsqueeze(-1)
        
        # 创建掩码
        seq_len = len(vals)
        num_mask = int(seq_len * self.mask_ratio)
        mask_indices = np.random.choice(seq_len, num_mask, replace=False)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[mask_indices] = True
        
        # 创建掩码后的输入
        x_masked = x_seq_tensor.clone()
        x_masked[mask] = 0.0
        
        # 生成伪异常标签（用于辅助任务）
        # 基于统计特征判断异常：偏离均值超过2个标准差
        mean = vals.mean()
        std = vals.std() + 1e-6
        z_scores = np.abs((vals - mean) / std)
        anomaly_labels = torch.tensor(z_scores > 2.0, dtype=torch.float)
        
        return {
            'series_masked': x_masked,    # 掩码后的序列
            'original_values': x_seq_tensor,  # 原始值
            'pretrain_mask': mask,        # 掩码位置
            'anomaly_labels': anomaly_labels,  # 伪异常标签
            'text': ''  # 占位符
        }


def pretrain_model(model_config_name='MEDIUM',
                   epochs=50,
                   lr=1e-4,
                   batch_size=16,
                   accumulation_steps=2,
                   patience=5,
                   max_samples=None,
                   use_first_n=None,
                   mask_ratio=0.15,
                   recon_weight=1.0,
                   anomaly_weight=0.5,
                   checkpoint_dir='checkpoints_pretrain_axis'):
    """
    预训练模型
    
    Args:
        use_first_n: 按顺序使用前N条数据（默认5000，设为None则使用全部）
        model_config_name: 模型配置名称
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批次大小
        accumulation_steps: 梯度累积步数
        patience: 早停耐心值
        max_samples: 最大样本数（快速实验）
        mask_ratio: 掩码比例
        recon_weight: 重建损失权重
        anomaly_weight: 异常检测损失权重
        checkpoint_dir: 检查点保存目录
    """
    config = getattr(ModelConfig, model_config_name)
    
    print(f"\n{'='*60}")
    print(f"AXIS风格预训练")
    print(f"{'='*60}")
    print(f"模型配置: {model_config_name}")
    print(f"配置详情: {config}")
    print(f"批次大小: {batch_size} (累积{accumulation_steps}步)")
    print(f"掩码比例: {mask_ratio}")
    print(f"损失权重: recon={recon_weight}, anomaly={anomaly_weight}")
    print(f"{'='*60}\n")
    
    # 数据集 - 使用无标签的训练数据进行预训练（全是正常样本）
    # 默认只使用前5000条样本，可通过 --use-first-n 修改
    actual_use_first_n = use_first_n if use_first_n is not None else 5000
    dataset = ElectricityDatasetPretrain(
        'dataset/File1_train.csv',
        'dataset/user_historical_stats.json',
        config=config,
        max_samples=max_samples,
        use_first_n=actual_use_first_n,
        mask_ratio=mask_ratio,
        mode='train'
    )
    
    print(f"数据集大小: {len(dataset)} 条")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 模型 - 只使用数值流部分，不需要LLM
    model = H3_PerceptionLayerAXIS(llm_dim=2048, **config)
    model.numerical_stream.enable_pretrain_heads()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"  总数: {total_params:,}")
    print(f"  可训练: {trainable_params:,}\n")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 学习率调度
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=epochs//4,
        T_mult=2
    )
    
    model.train()
    
    # 训练循环
    best_loss = float('inf')
    patience_counter = 0
    history = []
    
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_anomaly_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(loader):
            # 移动数据到设备
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # 前向传播
            losses = model.pretrain_forward(batch)
            
            recon_loss = losses['reconstruction_loss']
            anomaly_loss = losses.get('anomaly_loss', torch.tensor(0.0, device=device))
            
            # 加权总损失
            loss = recon_weight * recon_loss + anomaly_weight * anomaly_loss
            loss = loss / accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 梯度累积更新
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(loader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # 记录损失
            total_loss += loss.item() * accumulation_steps
            total_recon_loss += recon_loss.item()
            if isinstance(anomaly_loss, torch.Tensor):
                total_anomaly_loss += anomaly_loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / num_batches
                avg_recon = total_recon_loss / num_batches
                avg_anomaly = total_anomaly_loss / num_batches
                print(f"Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                      f"Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, Anomaly: {avg_anomaly:.4f})")
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_anomaly_loss = total_anomaly_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n=== Epoch {epoch} 完成 ===")
        print(f"  总损失: {avg_loss:.4f}")
        print(f"  重建损失: {avg_recon_loss:.4f}")
        print(f"  异常损失: {avg_anomaly_loss:.4f}")
        print(f"  学习率: {current_lr:.2e}")
        
        history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'anomaly_loss': avg_anomaly_loss,
            'lr': current_lr
        })
        
        # 保存检查点
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'numerical_stream': model.numerical_stream.state_dict(),
            'epoch': epoch,
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'anomaly_loss': avg_anomaly_loss,
            'config': config,
            'config_name': model_config_name
        }
        
        checkpoint_name = f"pretrain_{model_config_name.lower()}_ep{epoch}_loss{avg_loss:.4f}.pth"
        torch.save(checkpoint, f"{checkpoint_dir}/{checkpoint_name}")
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(checkpoint, f"{checkpoint_dir}/pretrain_{model_config_name.lower()}_best.pth")
            print(f"✓ 新的最佳模型 (loss={best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"早停计数: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n早停触发，训练结束")
                break
    
    # 保存训练历史
    with open(f"{checkpoint_dir}/history_{model_config_name.lower()}.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"预训练完成 | 最佳 Loss: {best_loss:.4f}")
    print(f"检查点保存于: {checkpoint_dir}")
    print(f"{'='*60}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrain AXIS-style model')
    parser.add_argument('--config', type=str, default='MEDIUM',
                       choices=['SMALL', 'BASE', 'MEDIUM', 'LARGE'],
                       help='Model configuration')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--accumulation-steps', type=int, default=2,
                       help='Gradient accumulation steps')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='随机采样指定数量的样本（与--use-first-n互斥）')
    parser.add_argument('--use-first-n', type=int, default=5000,
                       help='按顺序使用前N条样本（默认5000，设为-1使用全部）')
    parser.add_argument('--mask-ratio', type=float, default=0.15,
                       help='Mask ratio for pretraining')
    parser.add_argument('--recon-weight', type=float, default=1.0,
                       help='Weight for reconstruction loss')
    parser.add_argument('--anomaly-weight', type=float, default=0.5,
                       help='Weight for anomaly detection loss')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_pretrain_axis',
                       help='Checkpoint directory')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: 1000 samples, 10 epochs')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式（打印各阶段中间结果）')
    
    args = parser.parse_args()
    
    # 设置全局调试开关
    if args.debug:
        from utils.debug_utils import DebugLogger
        DebugLogger.set_enabled(True)
        print("\n[调试模式已启用] 将打印各阶段中间结果\n")
    
    # 快速模式
    max_samples = args.max_samples
    use_first_n = args.use_first_n if args.use_first_n != -1 else None
    epochs = args.epochs
    if args.quick:
        max_samples = 1000
        use_first_n = None  # 快速模式使用随机采样
        epochs = 10
        print(f"[快速模式] 随机采样 {max_samples} 条数据，训练 {epochs} 个 epoch")
    
    pretrain_model(
        model_config_name=args.config,
        epochs=epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        patience=args.patience,
        max_samples=max_samples,
        use_first_n=use_first_n,
        mask_ratio=args.mask_ratio,
        recon_weight=args.recon_weight,
        anomaly_weight=args.anomaly_weight,
        checkpoint_dir=args.checkpoint_dir
    )
