#!/usr/bin/env python3
"""
诊断 Objaverse source 数据的 view 4 相机角度问题
直接加载数据集，追踪每一步变换
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['XFORMERS_DISABLED'] = '1'

import json
import torch
import numpy as np
from kiui.cam import orbit_camera

np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)

# 加载3个不同的 Objaverse 样本，看 view 4 的索引映射
sample_dirs = []
base = "datas/objaverse_rendered"
for d in sorted(os.listdir(base))[:3]:
    p = os.path.join(base, d, "render/transforms.json")
    if os.path.exists(p):
        sample_dirs.append((d, p))

import math
from data.dataset import OrthogonalViewSelector

selector = OrthogonalViewSelector(angle_offset=0.0)

for uuid, tpath in sample_dirs:
    print("\n" + "=" * 80)
    print(f"样本: {uuid}")
    print("=" * 80)

    with open(tpath) as f:
        tdata = json.load(f)

    total_views = len(tdata['frames'])
    num_input_views = 4
    num_supervision_views = 6

    # 视图选择
    input_indices, supervision_indices = selector.select_views(
        total_views, num_input_views, tdata, sample_idx=0)

    # 限制监督视图数量（与 dataset.py 一致）
    if len(supervision_indices) > num_supervision_views:
        step = len(supervision_indices) / num_supervision_views
        supervision_indices = [supervision_indices[int(i * step)]
                               for i in range(num_supervision_views)]

    all_indices = input_indices + supervision_indices
    print(f"input_indices = {input_indices}")
    print(f"supervision_indices = {supervision_indices}")
    print(f"all_indices = {all_indices}")
    print(f"view 4 (combined) = all_indices[4] = 原始视图 {all_indices[4]}")

    # 打印原始视图信息
    for i, idx in enumerate(all_indices):
        frame = tdata['frames'][idx]
        mat = np.array(frame['transform_matrix'])
        scale = frame.get('scale', 1.0)
        el = frame.get('elevation', '?')
        rot = frame.get('rotation', '?')

        # 除以 scale 后的位置
        pos_raw = mat[:3, 3] / scale
        R_raw = mat[:3, :3] / scale
        dist_raw = np.linalg.norm(pos_raw)

        # 检查是否接近单位矩阵
        is_identity = np.allclose(R_raw, np.eye(3), atol=0.01)

        if i <= 5:  # 只打印前6个
            print(f"\n  all_indices[{i}] = 原始视图{idx}: "
                  f"el={el}, rot={rot}, dist={dist_raw:.4f}, "
                  f"R≈I: {is_identity}")
            if i == 4:
                print(f"    *** 这是 view 4 ***")
                print(f"    位置/scale: {pos_raw}")
                print(f"    旋转/scale:\n{R_raw}")

    # ========== 追踪完整变换流程 ==========
    target_radius = 1.5
    cam_poses = []

    for idx in all_indices:
        frame = tdata['frames'][idx]
        c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
        scale = frame.get('scale', 1.0)
        c2w[:3, :] /= scale

        # Blender → OpenGL
        c2w[1] *= -1
        c2w[[1, 2]] = c2w[[2, 1]]
        c2w[:3, 1:3] *= -1
        cam_poses.append(c2w)

    cam_poses = torch.stack(cam_poses)

    # SVD 正交化
    for i in range(cam_poses.shape[0]):
        R = cam_poses[i, :3, :3]
        U, _, Vt = torch.linalg.svd(R)
        cam_poses[i, :3, :3] = U @ Vt

    # 缩放到 radius=1.5
    for i in range(cam_poses.shape[0]):
        dist = torch.norm(cam_poses[i, :3, 3])
        if dist > 1e-6:
            cam_poses[i, :3, 3] *= (target_radius / dist)

    print(f"\n--- Blender→OpenGL + SVD + 缩放后 ---")
    for i in [0, 4]:  # camera 0 和 camera 4
        pos = cam_poses[i, :3, 3].numpy()
        fwd = -cam_poses[i, :3, 2].numpy()
        up = cam_poses[i, :3, 1].numpy()
        dist = np.linalg.norm(pos)
        to_origin = -pos / max(dist, 1e-8)
        dot = np.dot(fwd, to_origin)
        print(f"  cam[{i}](视图{all_indices[i]}): pos={pos}, "
              f"fwd={fwd}, up={up}, dot={dot:.4f}")

    # ========== 方案A: 当前代码 (Y+Z翻转 + R_align) ==========
    cam_A = cam_poses.clone()

    # Y+Z 翻转
    cam_A[:, :3, 1] *= -1
    cam_A[:, :3, 2] *= -1

    # 翻转后修复背对原点的相机
    for i in range(cam_A.shape[0]):
        pos_i = cam_A[i, :3, 3]
        fwd_i = -cam_A[i, :3, 2]
        to_origin_i = -pos_i / torch.norm(pos_i)
        dot_i = torch.dot(fwd_i, to_origin_i).item()
        if dot_i < 0:
            z_axis = -to_origin_i
            world_up = torch.tensor([0.0, 1.0, 0.0])
            x_axis = torch.linalg.cross(world_up, z_axis)
            x_norm = torch.norm(x_axis)
            if x_norm < 1e-6:
                world_up = torch.tensor([1.0, 0.0, 0.0])
                x_axis = torch.linalg.cross(world_up, z_axis)
                x_norm = torch.norm(x_axis)
            x_axis = x_axis / x_norm
            y_axis = torch.linalg.cross(z_axis, x_axis)
            cam_A[i, :3, :3] = torch.stack([x_axis, y_axis, z_axis], dim=1)

    # R_align
    R_align = torch.inverse(cam_A[0, :3, :3])
    for i in range(cam_A.shape[0]):
        cam_A[i, :3, 3] = R_align @ cam_A[i, :3, 3]
        cam_A[i, :3, :3] = R_align @ cam_A[i, :3, :3]

    # 重建旋转矩阵（dot < 0.99）
    origin = torch.tensor([0.0, 0.0, 0.0])
    for i in range(cam_A.shape[0]):
        pos = cam_A[i, :3, 3]
        R = cam_A[i, :3, :3]
        forward = -R[:, 2]
        to_origin = origin - pos
        to_origin_norm = to_origin / torch.norm(to_origin)
        dot = torch.dot(forward, to_origin_norm).item()
        if dot < 0.99:
            z_axis = -to_origin_norm
            saved_up = R[:, 1]
            dot_up_z = torch.dot(saved_up, z_axis)
            y_axis = saved_up - dot_up_z * z_axis
            y_norm = torch.norm(y_axis)
            if y_norm < 1e-6:
                x_axis = R[:, 0] / torch.norm(R[:, 0])
                y_axis = torch.linalg.cross(z_axis, x_axis)
            else:
                y_axis = y_axis / y_norm
                x_axis = torch.linalg.cross(y_axis, z_axis)
            cam_A[i, :3, :3] = torch.stack([x_axis, y_axis, z_axis], dim=1)

    print(f"\n--- 方案A (当前代码: Y+Z翻转 + R_align) ---")
    for i in range(min(8, cam_A.shape[0])):
        pos = cam_A[i, :3, 3].numpy()
        fwd = -cam_A[i, :3, 2].numpy()
        up = cam_A[i, :3, 1].numpy()
        dist = np.linalg.norm(pos)
        to_origin = -pos / max(dist, 1e-8)
        dot = np.dot(fwd, to_origin)
        marker = " *** VIEW 4 ***" if i == 4 else ""
        print(f"  cam[{i}](视图{all_indices[i]}): "
              f"pos=[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}], "
              f"up=[{up[0]:+.3f},{up[1]:+.3f},{up[2]:+.3f}], "
              f"dot={dot:.4f}{marker}")

    # ========== 方案B: LGM 原始归一化 ==========
    cam_B = cam_poses.clone()

    T_translate = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, target_radius],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    transform = T_translate @ torch.inverse(cam_B[0])
    cam_B = transform.unsqueeze(0) @ cam_B

    print(f"\n--- 方案B (LGM 原始归一化) ---")
    for i in range(min(8, cam_B.shape[0])):
        pos = cam_B[i, :3, 3].numpy()
        fwd = -cam_B[i, :3, 2].numpy()
        up = cam_B[i, :3, 1].numpy()
        dist = np.linalg.norm(pos)
        to_origin = -pos / max(dist, 1e-8)
        dot = np.dot(fwd, to_origin)
        marker = " *** VIEW 4 ***" if i == 4 else ""
        print(f"  cam[{i}](视图{all_indices[i]}): "
              f"pos=[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}], "
              f"up=[{up[0]:+.3f},{up[1]:+.3f},{up[2]:+.3f}], "
              f"dot={dot:.4f}{marker}")

    # ========== 对比 view 4 的 up 方向 ==========
    print(f"\n--- View 4 对比 ---")
    up_A = cam_A[4, :3, 1].numpy()
    up_B = cam_B[4, :3, 1].numpy()
    print(f"  方案A up: {up_A}")
    print(f"  方案B up: {up_B}")
    print(f"  方案A 的 up 是否接近 [0,1,0]: {np.allclose(up_A, [0,1,0], atol=0.3)}")
    print(f"  方案B 的 up 是否接近 [0,1,0]: {np.allclose(up_B, [0,1,0], atol=0.3)}")

    # 只分析第一个样本
    break
