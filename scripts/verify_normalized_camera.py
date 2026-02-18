#!/usr/bin/env python3
"""
验证归一化后的相机姿态
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

print("=" * 80)
print("验证归一化后的相机姿态")
print("=" * 80)

# 加载 Objaverse 数据
transforms_path = "/mnt/huangjiaxin/3d-defense/datas/objaverse_rendered/e4041c753202476ea7051121ae33ea7d/render/transforms.json"
with open(transforms_path, 'r') as f:
    transforms_data = json.load(f)

# 模拟数据加载器的处理流程
cam_poses = []
for idx in [1, 2, 3, 4]:  # 正确的视图索引
    frame = transforms_data['frames'][idx]
    c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
    scale = frame.get('scale', 1.0)

    # 除以 scale
    c2w[:3, :] /= scale

    # 坐标系转换
    c2w[1] *= -1
    c2w[[1, 2]] = c2w[[2, 1]]
    c2w[:3, 1:3] *= -1

    # 计算当前相机到原点的距离并缩放到目标半径
    cam_pos = c2w[:3, 3]
    current_radius = torch.norm(cam_pos)
    target_radius = 1.5
    c2w[:3, 3] = cam_pos * (target_radius / current_radius)

    cam_poses.append(c2w)

cam_poses = torch.stack(cam_poses, dim=0)

# 正交化旋转矩阵
for i in range(cam_poses.shape[0]):
    R = cam_poses[i, :3, :3]
    U, _, Vt = torch.linalg.svd(R)
    R_ortho = U @ Vt
    cam_poses[i, :3, :3] = R_ortho

# 相机姿态归一化
target_radius = 1.5
transform = torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, target_radius],
    [0, 0, 0, 1]
], dtype=torch.float32) @ torch.inverse(cam_poses[0])
cam_poses = transform.unsqueeze(0) @ cam_poses

# 关键修复：姿态归一化后，统一缩放所有相机的radius到target_radius
for i in range(cam_poses.shape[0]):
    cam_pos = cam_poses[i, :3, 3]
    current_radius = torch.norm(cam_pos)
    if current_radius > 1e-6:  # 避免除零
        cam_poses[i, :3, 3] = cam_pos * (target_radius / current_radius)

print("\n归一化后的相机姿态:")
print("-" * 80)

for i in range(4):
    print(f"\n视图 {i} ({i*90}°):")
    print(f"  位置: {cam_poses[i, :3, 3].numpy()}")
    print(f"  旋转矩阵行列式: {torch.det(cam_poses[i, :3, :3]):.6f}")

    # 计算方位角和仰角
    pos = cam_poses[i, :3, 3]
    x, y, z = pos[0].item(), pos[1].item(), pos[2].item()
    azimuth = np.arctan2(y, x) * 180 / np.pi
    if azimuth < 0:
        azimuth += 360
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    radius = np.sqrt(x**2 + y**2 + z**2)

    print(f"  方位角: {azimuth:.2f}°")
    print(f"  仰角: {elevation:.2f}°")
    print(f"  半径: {radius:.2f}")

# 对比 LGM 期望的相机姿态
print("\n" + "=" * 80)
print("LGM 期望的相机姿态（elevation=0°）:")
print("-" * 80)

for i, azimuth in enumerate([0, 90, 180, 270]):
    cam_pose = orbit_camera(0, azimuth, radius=1.5, opengl=True)
    pos = cam_pose[:3, 3]
    print(f"\n视图 {i} ({azimuth}°):")
    print(f"  位置: {pos}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print("如果归一化后的相机姿态和 LGM 期望的相似，说明处理正确")
print("如果仰角不是 0°，说明 Objaverse 数据的相机参数和 LGM 不匹配")
