#!/usr/bin/env python3
"""
对比我们的数据处理和 LGM 原始代码的数值差异
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# 添加 LGM 路径
lgm_path = '/mnt/huangjiaxin/3d-defense/third_party/LGM'
sys.path.insert(0, lgm_path)

from core.utils import get_rays
from kiui.cam import orbit_camera

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

print("=" * 80)
print("对比数据处理流程")
print("=" * 80)

# 测试图像路径
img_path = "/mnt/huangjiaxin/3d-defense/datas/objaverse_rendered/e4041c753202476ea7051121ae33ea7d/render/images/r_0.png"

# ============================================================================
# 方法 1：我们的处理流程（带相机变换）
# ============================================================================
print("\n[方法 1] 我们的处理流程（带相机变换）")
print("-" * 80)

# 加载图像
img = Image.open(img_path).convert('RGBA')
img = img.resize((256, 256), Image.BILINEAR)
img_tensor = TF.to_tensor(img)

rgb = img_tensor[:3]
alpha = img_tensor[3:4]
img_white_bg = rgb * alpha + (1 - alpha)

# ImageNet 归一化
img_normalized = (img_white_bg - IMAGENET_MEAN.view(3, 1, 1)) / IMAGENET_STD.view(3, 1, 1)

# 加载相机姿态（假设从 transforms.json）
import json
transforms_path = "/mnt/huangjiaxin/3d-defense/datas/objaverse_rendered/e4041c753202476ea7051121ae33ea7d/render/transforms.json"
with open(transforms_path, 'r') as f:
    transforms_data = json.load(f)

frame = transforms_data['frames'][0]
c2w_original = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
scale = frame.get('scale', 1.0)

print(f"原始相机姿态 (包含 scale={scale}):")
print(c2w_original)

# 关键修复：先除以 scale，得到真正的相机姿态
c2w_original[:3, :] /= scale

print(f"\n除以 scale 后的相机姿态:")
print(c2w_original)

# 应用相机坐标系转换
c2w = c2w_original.clone()
c2w[1] *= -1
c2w[[1, 2]] = c2w[[2, 1]]
c2w[:3, 1:3] *= -1
cam_radius = 1.5
c2w[:3, 3] *= cam_radius / 1.5

print(f"\n转换后的相机姿态:")
print(c2w)

# 正交化旋转矩阵
R = c2w[:3, :3]
U, _, Vt = torch.linalg.svd(R)
R_ortho = U @ Vt
c2w[:3, :3] = R_ortho

print(f"\n正交化后的相机姿态:")
print(c2w)
print(f"旋转矩阵行列式: {torch.det(c2w[:3, :3]):.6f} (应该接近 ±1)")

# 计算 rays
rays_o, rays_d = get_rays(c2w, 256, 256, 49.1)
rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)
rays_plucker = rays_plucker.permute(2, 0, 1)

print(f"\nRays embeddings 统计:")
print(f"  Shape: {rays_plucker.shape}")
print(f"  Mean: {rays_plucker.mean():.6f}")
print(f"  Std: {rays_plucker.std():.6f}")
print(f"  Min: {rays_plucker.min():.6f}")
print(f"  Max: {rays_plucker.max():.6f}")

# ============================================================================
# 方法 2：LGM 原始方法（使用固定的 orbit_camera）
# ============================================================================
print("\n" + "=" * 80)
print("[方法 2] LGM 原始方法（使用固定的 orbit_camera）")
print("-" * 80)

# 使用 orbit_camera 生成固定的相机姿态
cam_pose_fixed = torch.from_numpy(orbit_camera(0, 0, radius=1.5, opengl=True))
print(f"固定相机姿态 (orbit_camera):")
print(cam_pose_fixed)
print(f"旋转矩阵行列式: {torch.det(cam_pose_fixed[:3, :3]):.6f}")

# 计算 rays
rays_o_fixed, rays_d_fixed = get_rays(cam_pose_fixed, 256, 256, 49.1)
rays_plucker_fixed = torch.cat([torch.cross(rays_o_fixed, rays_d_fixed, dim=-1), rays_d_fixed], dim=-1)
rays_plucker_fixed = rays_plucker_fixed.permute(2, 0, 1)

print(f"\nRays embeddings 统计:")
print(f"  Shape: {rays_plucker_fixed.shape}")
print(f"  Mean: {rays_plucker_fixed.mean():.6f}")
print(f"  Std: {rays_plucker_fixed.std():.6f}")
print(f"  Min: {rays_plucker_fixed.min():.6f}")
print(f"  Max: {rays_plucker_fixed.max():.6f}")

# ============================================================================
# 对比差异
# ============================================================================
print("\n" + "=" * 80)
print("差异分析")
print("=" * 80)

diff_rays = (rays_plucker - rays_plucker_fixed).abs()
print(f"\nRays embeddings 差异:")
print(f"  Mean abs diff: {diff_rays.mean():.6f}")
print(f"  Max abs diff: {diff_rays.max():.6f}")
print(f"  Relative diff: {(diff_rays.mean() / rays_plucker_fixed.abs().mean() * 100):.2f}%")

# 检查相机姿态差异
print(f"\n相机姿态差异:")
print(f"  位置差异: {(c2w[:3, 3] - cam_pose_fixed[:3, 3]).abs().mean():.6f}")
print(f"  旋转差异: {(c2w[:3, :3] - cam_pose_fixed[:3, :3]).abs().mean():.6f}")

# ============================================================================
# 方法 3：不做正交化
# ============================================================================
print("\n" + "=" * 80)
print("[方法 3] 不做正交化的情况")
print("=" * 80)

c2w_no_ortho = c2w_original.clone()
c2w_no_ortho[1] *= -1
c2w_no_ortho[[1, 2]] = c2w_no_ortho[[2, 1]]
c2w_no_ortho[:3, 1:3] *= -1
c2w_no_ortho[:3, 3] *= cam_radius / 1.5

print(f"未正交化的旋转矩阵行列式: {torch.det(c2w_no_ortho[:3, :3]):.6f}")

rays_o_no_ortho, rays_d_no_ortho = get_rays(c2w_no_ortho, 256, 256, 49.1)
rays_plucker_no_ortho = torch.cat([torch.cross(rays_o_no_ortho, rays_d_no_ortho, dim=-1), rays_d_no_ortho], dim=-1)
rays_plucker_no_ortho = rays_plucker_no_ortho.permute(2, 0, 1)

diff_no_ortho = (rays_plucker_no_ortho - rays_plucker_fixed).abs()
print(f"\n与固定 rays 的差异:")
print(f"  Mean abs diff: {diff_no_ortho.mean():.6f}")
print(f"  Max abs diff: {diff_no_ortho.max():.6f}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print("如果差异很大，说明我们的相机变换有问题")
print("如果差异很小，说明问题可能在其他地方")
