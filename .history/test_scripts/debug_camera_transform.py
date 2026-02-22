#!/usr/bin/env python3
"""
详细分析相机变换问题
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json

lgm_path = '/mnt/huangjiaxin/3d-defense/third_party/LGM'
sys.path.insert(0, lgm_path)

from kiui.cam import orbit_camera
from core.utils import get_rays

print("=" * 80)
print("相机变换详细分析")
print("=" * 80)

# 加载 Objaverse 数据
transforms_path = "/mnt/huangjiaxin/3d-defense/datas/objaverse_rendered/e4041c753202476ea7051121ae33ea7d/render/transforms.json"
with open(transforms_path, 'r') as f:
    transforms_data = json.load(f)

# 分析前 4 个视图（应该对应 0°, 90°, 180°, 270°）
print("\n分析前 4 个视图的相机姿态")
print("-" * 80)

for i in range(4):
    frame = transforms_data['frames'][i]
    print(f"\n视图 {i} (file: {frame['file_path']}):")
    print(f"  Rotation: {frame.get('rotation', 'N/A')}")

    c2w_original = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
    scale = frame.get('scale', 1.0)

    # 计算相机位置（除以 scale 前后）
    pos_before = c2w_original[:3, 3]
    pos_after = pos_before / scale

    print(f"  Scale: {scale:.6f}")
    print(f"  Position (before scale): {pos_before.numpy()}")
    print(f"  Position (after scale): {pos_after.numpy()}")

    # 计算方位角
    x, y, z = pos_after[0].item(), pos_after[1].item(), pos_after[2].item()
    azimuth = np.arctan2(y, x) * 180 / np.pi
    if azimuth < 0:
        azimuth += 360
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    radius = np.sqrt(x**2 + y**2 + z**2)

    print(f"  Computed azimuth: {azimuth:.2f}°")
    print(f"  Computed elevation: {elevation:.2f}°")
    print(f"  Computed radius: {radius:.2f}")

# 对比固定相机姿态
print("\n" + "=" * 80)
print("固定相机姿态（LGM 期望）")
print("-" * 80)

for i, azimuth in enumerate([0, 90, 180, 270]):
    cam_pose = orbit_camera(0, azimuth, radius=1.5, opengl=True)
    pos = cam_pose[:3, 3]
    print(f"\n视图 {i} ({azimuth}°):")
    print(f"  Position: {pos}")
    print(f"  Rotation matrix:")
    print(f"    {cam_pose[:3, :3]}")

# 尝试不同的变换方式
print("\n" + "=" * 80)
print("测试不同的变换方式")
print("=" * 80)

frame = transforms_data['frames'][0]
c2w_original = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
scale = frame.get('scale', 1.0)

print(f"\n原始 transform_matrix:")
print(c2w_original)

# 方法 1：我们当前的方法
print(f"\n[方法 1] 当前方法：除以 scale + 坐标系转换")
c2w_1 = c2w_original.clone()
c2w_1[:3, :] /= scale
c2w_1[1] *= -1
c2w_1[[1, 2]] = c2w_1[[2, 1]]
c2w_1[:3, 1:3] *= -1
print(c2w_1)
print(f"旋转矩阵行列式: {torch.det(c2w_1[:3, :3]):.6f}")

# 方法 2：只除以 scale，不做坐标系转换
print(f"\n[方法 2] 只除以 scale")
c2w_2 = c2w_original.clone()
c2w_2[:3, :] /= scale
print(c2w_2)
print(f"旋转矩阵行列式: {torch.det(c2w_2[:3, :3]):.6f}")

# 方法 3：检查是否需要归一化到 LGM 的相机空间
print(f"\n[方法 3] 分析相机空间")
c2w_scaled = c2w_original.clone()
c2w_scaled[:3, :] /= scale

# 计算相机到原点的距离
cam_pos = c2w_scaled[:3, 3]
cam_dist = torch.norm(cam_pos)
print(f"相机到原点距离: {cam_dist:.2f}")
print(f"LGM 期望距离: 1.5")
print(f"缩放因子: {1.5 / cam_dist:.6f}")

# 方法 4：直接缩放到 LGM 的相机距离
print(f"\n[方法 4] 缩放到 LGM 相机距离")
c2w_4 = c2w_scaled.clone()
c2w_4[:3, 3] *= (1.5 / cam_dist)
print(f"缩放后位置: {c2w_4[:3, 3]}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print("问题可能在于：")
print("1. 坐标系转换破坏了旋转矩阵的正交性")
print("2. 相机位置的缩放方式不对")
print("3. Objaverse 数据的相机设置和 LGM 训练时的设置不匹配")
print("\n建议：")
print("- 检查 Objaverse 渲染时使用的相机参数")
print("- 或者直接使用固定的相机姿态（因为 LGM 模型期望标准的正交视图）")
