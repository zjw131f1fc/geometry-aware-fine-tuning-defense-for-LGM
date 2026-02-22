#!/usr/bin/env python3
"""
调试相机归一化：检查归一化后的相机是否在球面上
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from data.dataset import OmniObject3DDataset

# 创建数据集
dataset = OmniObject3DDataset(
    data_root='datas/omniobject3d',
    categories=['pumpkin'],
    num_input_views=4,
    num_supervision_views=4,
)

# 获取一个样本
sample = dataset[0]

# 合并input和supervision transforms
input_transforms = sample['input_transforms']  # [4, 4, 4]
supervision_transforms = sample['supervision_transforms']  # [4, 4, 4]
all_transforms = torch.cat([input_transforms, supervision_transforms], dim=0)  # [8, 4, 4]

print("=" * 80)
print("检查归一化后的相机姿态")
print("=" * 80)

for i in range(all_transforms.shape[0]):
    c2w = all_transforms[i]  # [4, 4] OpenGL c2w

    # 提取位置和旋转
    pos = c2w[:3, 3].numpy()
    rot = c2w[:3, :3].numpy()

    # 计算半径
    radius = np.linalg.norm(pos)

    # 检查旋转矩阵是否正交
    rot_check = rot @ rot.T
    is_orthogonal = np.allclose(rot_check, np.eye(3), atol=1e-5)

    # 计算相机看向的方向（OpenGL：-Z轴）
    forward = -rot[:, 2]  # OpenGL看向-Z

    # 计算相机位置到原点的方向
    to_origin = -pos / radius

    # 检查相机是否看向原点
    cos_angle = np.dot(forward, to_origin)

    print(f"\n相机 {i}:")
    print(f"  位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    print(f"  半径: {radius:.4f}")
    print(f"  旋转正交: {is_orthogonal}")
    print(f"  看向原点 (cos): {cos_angle:.4f} (应该接近1.0)")

    if i == 0:
        print(f"  [Camera 0] 旋转矩阵是否为单位矩阵: {np.allclose(rot, np.eye(3), atol=1e-5)}")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(f"所有相机半径范围: {np.linalg.norm(all_transforms[:, :3, 3].numpy(), axis=1).min():.4f} ~ {np.linalg.norm(all_transforms[:, :3, 3].numpy(), axis=1).max():.4f}")
print(f"期望半径: 1.5")
