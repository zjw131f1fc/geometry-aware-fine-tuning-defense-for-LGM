#!/usr/bin/env python3
"""检查Gaussian参数的统计信息"""

import sys
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from project_core import ConfigManager
from models import ModelManager
from data import DataManager

# 加载模型
config_mgr = ConfigManager('configs/config.yaml')
config = config_mgr.config
config['model']['resume'] = 'output/test_train_overfit_scale_opacity/model_after_attack.pth'

model_mgr = ModelManager(config)
model_mgr.setup(device='cuda')
model = model_mgr.model
opt = model_mgr.opt

# 创建数据加载器
data_mgr = DataManager(config, opt)
data_mgr.setup_dataloaders(train=True, val=False, subset='target')
train_loader = data_mgr.train_loader

# 获取第一个样本
first_sample = next(iter(train_loader))

# 生成Gaussian
model.eval()
with torch.no_grad():
    input_images = first_sample['input_images'].cuda()
    gaussians = model.forward_gaussians(input_images)  # [B, N, 14]

    # Gaussian参数：xyz(3) + opacity(1) + scale(3) + rotation(4) + rgb(3) = 14
    xyz = gaussians[:, :, 0:3]
    opacity = gaussians[:, :, 3:4]
    scale = gaussians[:, :, 4:7]
    rotation = gaussians[:, :, 7:11]
    rgb = gaussians[:, :, 11:14]

    print("=== Gaussian统计信息 ===")
    print(f"Gaussian数量: {gaussians.shape[1]}")
    print(f"\n位置 (xyz):")
    print(f"  mean: {xyz.mean(dim=1).squeeze()}")
    print(f"  std: {xyz.std(dim=1).squeeze()}")
    print(f"  min: {xyz.min():.4f}, max: {xyz.max():.4f}")

    print(f"\n不透明度 (opacity):")
    print(f"  mean: {opacity.mean():.4f}")
    print(f"  std: {opacity.std():.4f}")
    print(f"  min: {opacity.min():.4f}, max: {opacity.max():.4f}")
    print(f"  接近0的比例 (<0.01): {(opacity < 0.01).float().mean():.2%}")
    print(f"  接近1的比例 (>0.99): {(opacity > 0.99).float().mean():.2%}")

    print(f"\n尺度 (scale):")
    print(f"  mean: {scale.mean(dim=1).squeeze()}")
    print(f"  std: {scale.std(dim=1).squeeze()}")
    print(f"  min: {scale.min():.6f}, max: {scale.max():.6f}")

    print(f"\n颜色 (rgb):")
    print(f"  mean: {rgb.mean(dim=1).squeeze()}")
    print(f"  std: {rgb.std(dim=1).squeeze()}")
    print(f"  min: {rgb.min():.4f}, max: {rgb.max():.4f}")

print("\n完成！")
