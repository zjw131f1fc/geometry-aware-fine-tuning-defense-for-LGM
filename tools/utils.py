"""
通用工具函数
"""

import torch
import numpy as np
import random


def set_seed(seed: int):
    """设置全局随机种子（PyTorch + NumPy + Python random）"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_base_model(model):
    """获取底层模型（处理 LoRA / DDP / FSDP 包装）"""
    if hasattr(model, 'base_model'):
        model = model.base_model
    if hasattr(model, 'model'):
        model = model.model
    if hasattr(model, 'module'):
        model = model.module
    return model
