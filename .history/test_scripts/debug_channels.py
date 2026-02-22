#!/usr/bin/env python3
"""
调试通道顺序问题
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

# 测试 1：检查 PIL 加载的图像
print("=" * 80)
print("测试 1：PIL 图像加载")
print("=" * 80)

img_path = "/mnt/huangjiaxin/3d-defense/datas/objaverse_rendered/e4041c753202476ea7051121ae33ea7d/render/images/r_0.png"
img = Image.open(img_path).convert('RGBA')
print(f"PIL mode: {img.mode}")
print(f"PIL size: {img.size}")

# 转换为 tensor
img_tensor = TF.to_tensor(img)
print(f"Tensor shape: {img_tensor.shape}")
print(f"Tensor dtype: {img_tensor.dtype}")
print(f"Tensor range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")

# 检查 RGB 通道
rgb = img_tensor[:3]
alpha = img_tensor[3:4]
print(f"\nRGB shape: {rgb.shape}")
print(f"Alpha shape: {alpha.shape}")
print(f"Alpha mean: {alpha.mean():.4f}")

# 白色背景处理
img_white_bg = rgb * alpha + (1 - alpha)
print(f"\n白色背景处理后:")
print(f"  Range: [{img_white_bg.min():.4f}, {img_white_bg.max():.4f}]")
print(f"  Mean: {img_white_bg.mean():.4f}")

# ImageNet 归一化
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_normalized = (img_white_bg - IMAGENET_MEAN) / IMAGENET_STD
print(f"\nImageNet 归一化后:")
print(f"  Range: [{img_normalized.min():.4f}, {img_normalized.max():.4f}]")
print(f"  Mean: {img_normalized.mean():.4f}")
print(f"  Std: {img_normalized.std():.4f}")

# 测试 2：检查反归一化
print("\n" + "=" * 80)
print("测试 2：反归一化")
print("=" * 80)

img_denorm = img_normalized * IMAGENET_STD + IMAGENET_MEAN
img_denorm = torch.clamp(img_denorm, 0, 1)
print(f"反归一化后:")
print(f"  Range: [{img_denorm.min():.4f}, {img_denorm.max():.4f}]")
print(f"  与原图差异: {(img_denorm - img_white_bg).abs().mean():.6f}")

# 保存测试图像
print("\n" + "=" * 80)
print("测试 3：保存图像")
print("=" * 80)

output_dir = "output/debug_channels"
os.makedirs(output_dir, exist_ok=True)

# 保存原图
img_np = (img_white_bg.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
Image.fromarray(img_np).save(os.path.join(output_dir, "original.png"))
print(f"✓ 原图已保存: {output_dir}/original.png")

# 保存反归一化后的图像
img_denorm_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
Image.fromarray(img_denorm_np).save(os.path.join(output_dir, "denormalized.png"))
print(f"✓ 反归一化图像已保存: {output_dir}/denormalized.png")

# 测试通道交换
print("\n" + "=" * 80)
print("测试 4：通道交换")
print("=" * 80)

# RGB → BGR
img_bgr = img_white_bg[[2, 1, 0]]
img_bgr_np = (img_bgr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
Image.fromarray(img_bgr_np).save(os.path.join(output_dir, "bgr.png"))
print(f"✓ BGR 图像已保存: {output_dir}/bgr.png")

print("\n完成！请检查 output/debug_channels/ 目录中的图像")
