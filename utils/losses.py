"""
Focal Loss 实现
用于处理类别不平衡问题

Focal Loss = -α(1-p)^γ log(p)
其中:
- α: 类别权重，用于平衡正负样本
- γ: 聚焦参数，用于降低易分样本的权重

对于窃电检测（1:9不平衡）:
- 设置 α=0.75 (偏向少数类'Theft')
- 设置 γ=2.0 (降低易分样本影响)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification

    Args:
        alpha: 类别权重，平衡正负样本。默认为0.25，但对于窃电检测应设为0.75
        gamma: 聚焦参数，降低易分样本权重。默认为2.0
        reduction: 'mean', 'sum', 'none'
    """

    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) 或 (N,) logits
            targets: (N,) 类别索引

        Returns:
            loss: scalar
        """
        # 确保inputs是logits
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)

        # 计算CE loss (不reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # 计算pt (正确类别的概率)
        pt = torch.exp(-ce_loss)

        # 根据target应用alpha权重
        # alpha_t = alpha if target==1 else (1-alpha)
        if inputs.shape[1] == 2:  # 二分类
            alpha_t = torch.where(
                targets == 1,
                torch.tensor(self.alpha, device=inputs.device),
                torch.tensor(1 - self.alpha, device=inputs.device),
            )
        else:
            alpha_t = self.alpha

        # Focal Loss = alpha_t * (1-pt)^gamma * ce_loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    组合Focal Loss和CE Loss

    对于窃电检测，可以同时使用：
    - CE Loss: 提供稳定的梯度
    - Focal Loss: 专注于难分样本
    """

    def __init__(self, alpha=0.75, gamma=2.0, focal_weight=0.5):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)

        # 组合损失
        loss = (1 - self.focal_weight) * ce + self.focal_weight * focal

        return loss, {"ce_loss": ce.item(), "focal_loss": focal.item()}
