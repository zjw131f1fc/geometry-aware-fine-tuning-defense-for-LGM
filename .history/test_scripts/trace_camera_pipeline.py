#!/usr/bin/env python3
"""
追踪相机变换流程 - 逐步对比 dataset.py 当前方案 vs LGM 原始方案

目标：搞清楚为什么 view 4（原始视图0）的旋转变换不对
"""

import json
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.set_printoptions(precision=6, suppress=True)

# 加载一个实际样本的 transforms.json
sample_path = "datas/objaverse_rendered/c6eb26ad49164d65856ce8df05c9240c/render/transforms.json"
with open(sample_path) as f:
    data = json.load(f)

print("=" * 80)
print("Step 0: 原始 transforms.json 中的相机数据")
print("=" * 80)

# 模拟 dataset.py 的视图选择
# input_indices = [1, 2, 3, 4], supervision_indices = [0, 11, 18, 26, 33, 40]
# all_indices = [1, 2, 3, 4, 0, 11, 18, 26, 33, 40]
# 所以 cam_poses[4] = 原始视图 0
test_indices = [1, 2, 3, 4, 0, 11]  # 取前6个够了
target_radius = 1.5

for idx in [0, 1]:  # 只看原始视图0和视图1
    frame = data['frames'][idx]
    mat = np.array(frame['transform_matrix'])
    scale = frame.get('scale', 1.0)
    print(f"\n--- 原始视图 {idx} (r_{idx}) ---")
    print(f"  scale = {scale}")
    print(f"  elevation = {frame.get('elevation')}, rotation = {frame.get('rotation')}")
    print(f"  transform_matrix:\n{mat}")
    print(f"  旋转部分 / scale:\n{mat[:3, :3] / scale}")
    print(f"  位置 / scale: {mat[:3, 3] / scale}")
    print(f"  位置 radius: {np.linalg.norm(mat[:3, 3] / scale):.6f}")

print("\n" + "=" * 80)
print("Step 1: Blender→OpenGL 转换 (与 LGM 第95-97行一致)")
print("=" * 80)

def blender_to_opengl(c2w):
    """LGM 的 Blender→OpenGL 转换"""
    c2w = c2w.clone()
    c2w[1] *= -1
    c2w[[1, 2]] = c2w[[2, 1]]
    c2w[:3, 1:3] *= -1
    return c2w

cam_poses_raw = []
for idx in test_indices:
    frame = data['frames'][idx]
    c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
    scale = frame.get('scale', 1.0)
    c2w[:3, :] /= scale  # 除以 scale
    c2w = blender_to_opengl(c2w)
    cam_poses_raw.append(c2w)

cam_poses_raw = torch.stack(cam_poses_raw)

for i, idx in enumerate(test_indices[:3]):
    c2w = cam_poses_raw[i]
    pos = c2w[:3, 3].numpy()
    R = c2w[:3, :3].numpy()
    forward = -R[:, 2]  # OpenGL: -Z 是前向
    up = R[:, 1]
    dist = np.linalg.norm(pos)
    print(f"\n--- cam_poses[{i}] (原始视图{idx}) Blender→OpenGL后 ---")
    print(f"  位置: {pos}, radius={dist:.6f}")
    print(f"  前向(-Z): {forward}")
    print(f"  上方(Y):  {up}")
    if dist > 1e-6:
        to_origin = -pos / dist
        dot = np.dot(forward, to_origin)
        print(f"  朝向原点dot: {dot:.4f} ({'✓朝向' if dot > 0 else '✗背对'})")

print("\n" + "=" * 80)
print("Step 2: SVD正交化 + 缩放到 radius=1.5")
print("=" * 80)

cam_poses = cam_poses_raw.clone()
for i in range(cam_poses.shape[0]):
    R = cam_poses[i, :3, :3]
    U, _, Vt = torch.linalg.svd(R)
    cam_poses[i, :3, :3] = U @ Vt
    dist = torch.norm(cam_poses[i, :3, 3])
    if dist > 1e-6:
        cam_poses[i, :3, 3] *= (target_radius / dist)

for i, idx in enumerate(test_indices[:3]):
    c2w = cam_poses[i]
    pos = c2w[:3, 3].numpy()
    R = c2w[:3, :3].numpy()
    forward = -R[:, 2]
    dist = np.linalg.norm(pos)
    print(f"\n--- cam_poses[{i}] (原始视图{idx}) SVD+缩放后 ---")
    print(f"  位置: {pos}, radius={dist:.6f}")
    print(f"  前向(-Z): {forward}")
    if dist > 1e-6:
        to_origin = -pos / dist
        dot = np.dot(forward, to_origin)
        print(f"  朝向原点dot: {dot:.4f} ({'✓朝向' if dot > 0 else '✗背对'})")

print("\n⚠️  关键发现：原始视图0的位置由数值噪声决定！")
print("   原始位置 ≈ [0, 0, 0]，缩放到1.5后方向完全取决于噪声")

print("\n" + "=" * 80)
print("Step 3A: 当前方案 - Y+Z翻转 + R_align")
print("=" * 80)

cam_current = cam_poses.clone()

# Y+Z 翻转
cam_current[:, :3, 1] *= -1
cam_current[:, :3, 2] *= -1

print("\nY+Z翻转后:")
for i, idx in enumerate(test_indices[:3]):
    c2w = cam_current[i]
    pos = c2w[:3, 3].numpy()
    forward = -c2w[:3, 2].numpy()
    dist = np.linalg.norm(pos)
    if dist > 1e-6:
        to_origin = -pos / dist
        dot = np.dot(forward, to_origin)
        print(f"  cam[{i}](视图{idx}): forward={forward}, dot={dot:.4f} ({'✓' if dot > 0 else '✗'})")

# R_align
R_align = torch.inverse(cam_current[0, :3, :3])
for i in range(cam_current.shape[0]):
    cam_current[i, :3, 3] = R_align @ cam_current[i, :3, 3]
    cam_current[i, :3, :3] = R_align @ cam_current[i, :3, :3]

print("\nR_align后:")
for i, idx in enumerate(test_indices[:6]):
    c2w = cam_current[i]
    pos = c2w[:3, 3].numpy()
    R = c2w[:3, :3].numpy()
    forward = -R[:, 2]
    up = R[:, 1]
    dist = np.linalg.norm(pos)
    if dist > 1e-6:
        to_origin = -pos / dist
        dot = np.dot(forward, to_origin)
    else:
        dot = 0
    print(f"  cam[{i}](视图{idx}): pos={pos}, forward={forward}, up={up}, dot={dot:.4f}")

print("\n" + "=" * 80)
print("Step 3B: LGM 原始方案 - transform @ cam_poses")
print("=" * 80)

cam_lgm = cam_poses.clone()

# LGM 的归一化：transform = T_translate @ inv(cam_poses[0])
T_translate = torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, target_radius],
    [0, 0, 0, 1]
], dtype=torch.float32)

transform = T_translate @ torch.inverse(cam_lgm[0])
cam_lgm = transform.unsqueeze(0) @ cam_lgm

print("\nLGM归一化后:")
for i, idx in enumerate(test_indices[:6]):
    c2w = cam_lgm[i]
    pos = c2w[:3, 3].numpy()
    R = c2w[:3, :3].numpy()
    forward = -R[:, 2]
    up = R[:, 1]
    dist = np.linalg.norm(pos)
    if dist > 1e-6:
        to_origin = -pos / dist
        dot = np.dot(forward, to_origin)
    else:
        dot = 0
    print(f"  cam[{i}](视图{idx}): pos={pos}, forward={forward}, up={up}, dot={dot:.4f}")

print("\n" + "=" * 80)
print("Step 4: 对比 LGM 归一化 vs orbit_camera")
print("=" * 80)

from kiui.cam import orbit_camera

# LGM 归一化后 camera 0 应该等价于 orbit_camera(0, 0, 1.5, opengl=True)
ref_cam = orbit_camera(0, 0, radius=target_radius, opengl=True)
ref_cam = torch.from_numpy(ref_cam)
print(f"\norbit_camera(0, 0, 1.5, opengl=True):")
print(f"  pos: {ref_cam[:3, 3].numpy()}")
print(f"  R:\n{ref_cam[:3, :3].numpy()}")

print(f"\nLGM归一化后 camera 0:")
print(f"  pos: {cam_lgm[0, :3, 3].numpy()}")
print(f"  R:\n{cam_lgm[0, :3, :3].numpy()}")

# 检查 camera 4（原始视图0）在 LGM 方案下的表现
print(f"\n--- 关键对比：cam[4]（原始视图0）---")
print(f"\n当前方案 cam[4]:")
c = cam_current[4]
print(f"  pos: {c[:3, 3].numpy()}")
print(f"  forward: {(-c[:3, 2]).numpy()}")
print(f"  up: {c[:3, 1].numpy()}")

print(f"\nLGM方案 cam[4]:")
c = cam_lgm[4]
print(f"  pos: {c[:3, 3].numpy()}")
print(f"  forward: {(-c[:3, 2]).numpy()}")
print(f"  up: {c[:3, 1].numpy()}")

print("\n" + "=" * 80)
print("Step 5: 验证 LGM 方案的渲染一致性")
print("=" * 80)

# 模拟 OpenGL→COLMAP 转换 + cam_pos 计算
print("\n--- LGM 方案的渲染参数 ---")
for i, idx in enumerate(test_indices[:6]):
    c2w = cam_lgm[i].clone()
    # OpenGL → COLMAP
    c2w[:3, 1:3] *= -1
    cam_view = torch.inverse(c2w).T
    cam_pos_lgm = -c2w[:3, 3]  # LGM 原始代码用负号

    c2w_cur = cam_current[i].clone()
    c2w_cur[:3, 1:3] *= -1
    cam_pos_cur = c2w_cur[:3, 3]  # 当前 evaluator 不用负号

    print(f"  cam[{i}](视图{idx}): LGM cam_pos={cam_pos_lgm.numpy()}, 当前 cam_pos={cam_pos_cur.numpy()}")

print("\n⚠️  注意：LGM 原始代码 cam_pos = -cam_poses[:, :3, 3]（有负号）")
print("   当前 evaluator._render_from_transforms 用 cam_pos = cam_pose_colmap[:, :3, 3]（无负号）")
print("   finetuner._prepare_data 用 cam_pos = -cam_pose_colmap[:3, 3]（有负号）")
print("   这是另一个不一致！")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print("""
根因：原始视图0 的 transform_matrix 是缩放的单位矩阵，位置 ≈ [0,0,0]。
      除以 scale 后位置仍然 ≈ 0，缩放到 radius=1.5 时方向由数值噪声决定。
      Y+Z翻转对这种退化相机无效（翻转前朝向原点，翻转后背对原点）。

修复方案：用 LGM 原始的归一化方式替换 Y+Z翻转 + R_align + look-at重建：
    transform = T_translate(cam_radius) @ inv(cam_poses[0])
    cam_poses = transform @ cam_poses

这是一个统一的刚体变换，对所有相机（包括退化的视图0）都正确。
""")
