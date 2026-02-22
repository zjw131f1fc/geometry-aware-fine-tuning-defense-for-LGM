#!/usr/bin/env python3
"""分析GT图像在物体区域的亮度分布"""

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
opt = model_mgr.opt

# 创建数据加载器
data_mgr = DataManager(config, opt)
data_mgr.setup_dataloaders(train=True, val=False, subset='target')
train_loader = data_mgr.train_loader

# 获取第一个样本
first_sample = next(iter(train_loader))

# 获取GT图像和mask
supervision_images = first_sample['supervision_images'].cuda()  # [B, V, 3, H, W]
supervision_masks = first_sample['supervision_masks'].cuda()  # [B, V, 1, H, W]

print("=== GT图像亮度分析 ===\n")
print(f"图像形状: {supervision_images.shape}")
print(f"Mask形状: {supervision_masks.shape}")

# 分析物体区域的亮度
B, V, C, H, W = supervision_images.shape

for v in range(V):
    img = supervision_images[0, v]  # [3, H, W]
    mask = supervision_masks[0, v]  # [1, H, W]

    # 物体区域的像素
    obj_pixels = img[:, mask[0] > 0.5]  # [3, N]

    if obj_pixels.shape[1] > 0:
        print(f"\n视角 {v}:")
        print(f"  物体像素数: {obj_pixels.shape[1]}")
        print(f"  物体占比: {obj_pixels.shape[1] / (H*W):.2%}")
        print(f"  RGB均值: [{obj_pixels[0].mean():.3f}, {obj_pixels[1].mean():.3f}, {obj_pixels[2].mean():.3f}]")
        print(f"  RGB标准差: [{obj_pixels[0].std():.3f}, {obj_pixels[1].std():.3f}, {obj_pixels[2].std():.3f}]")
        print(f"  最小值: {obj_pixels.min():.3f}")
        print(f"  最大值: {obj_pixels.max():.3f}")
        print(f"  中位数: {obj_pixels.median():.3f}")

        # 分析亮度分布
        brightness = obj_pixels.mean(dim=0)  # [N]
        print(f"  亮度分布:")
        print(f"    <0.5 (暗): {(brightness < 0.5).float().mean():.1%}")
        print(f"    0.5-0.7 (中): {((brightness >= 0.5) & (brightness < 0.7)).float().mean():.1%}")
        print(f"    0.7-0.9 (亮): {((brightness >= 0.7) & (brightness < 0.9)).float().mean():.1%}")
        print(f"    >0.9 (很亮): {(brightness >= 0.9).float().mean():.1%}")

print("\n完成！")
