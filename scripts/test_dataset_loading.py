#!/usr/bin/env python3
"""
测试数据集加载和相机变换是否正确
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import ObjaverseRenderedDataset
import torch

# 创建数据集
dataset = ObjaverseRenderedDataset(
    data_root='datas/objaverse_rendered',
    num_input_views=4,
    num_supervision_views=4,
    input_size=256,
    view_selector='orthogonal',
    angle_offset=0.0,
    max_samples=1,
    samples_per_object=1,
)

print(f"数据集大小: {len(dataset)}")

# 加载一个样本
if len(dataset) > 0:
    sample = dataset[0]
    print(f"\n样本键: {sample.keys()}")
    print(f"input_images shape: {sample['input_images'].shape}")
    print(f"supervision_images shape: {sample['supervision_images'].shape}")
    print(f"input_transforms shape: {sample['input_transforms'].shape}")

    # 检查输入图像的统计信息
    input_imgs = sample['input_images']
    print(f"\n输入图像统计（ImageNet归一化后）:")
    print(f"  RGB通道 (前3通道): mean={input_imgs[:, :3].mean():.4f}, std={input_imgs[:, :3].std():.4f}")
    print(f"  Rays通道 (后6通道): mean={input_imgs[:, 3:].mean():.4f}, std={input_imgs[:, 3:].std():.4f}")

    # 检查相机变换
    print(f"\n相机变换矩阵（第一个输入视图）:")
    print(sample['input_transforms'][0])

    print("\n✓ 数据加载成功！")
else:
    print("错误：数据集为空")
