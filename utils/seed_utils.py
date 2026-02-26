"""
随机种子设置工具
确保实验可复现性
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    设置所有随机种子以确保完全可复现性

    Args:
        seed: 随机种子，默认为42
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

    # 确保确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 启用确定性算法（某些操作可能不支持，使用warn_only=True）
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        pass

    print(f"[Seed] 随机种子已设置为 {seed}")


def seed_worker(worker_id):
    """
    DataLoader worker的随机种子设置函数

    使用方式:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42)
        )
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
