#!/usr/bin/env python3
"""逐步追踪 view 4（原始视图0）的相机变换，找出 up 方向变错的步骤"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['XFORMERS_DISABLED'] = '1'

import json, torch, numpy as np
np.set_printoptions(precision=6, suppress=True)

sample_path = "datas/objaverse_rendered/00d1918300634af48489467002fcb308/render/transforms.json"
with open(sample_path) as f:
    tdata = json.load(f)

# 原始视图0 和 原始视图1（作为对比）
target_radius = 1.5

def print_cam(label, c2w):
    pos = c2w[:3, 3].numpy()
    R = c2w[:3, :3].numpy()
    fwd = -R[:, 2]
    up = R[:, 1]
    dist = np.linalg.norm(pos)
    to_origin = -pos / max(dist, 1e-8)
    dot = np.dot(fwd, to_origin)
    print(f"  {label}:")
    print(f"    pos={pos}, dist={dist:.4f}")
    print(f"    fwd={fwd}, up={up}")
    print(f"    dot={dot:.4f} ({'朝向' if dot > 0 else '背对'}原点)")

# 同时追踪视图0（view 4）和视图1（view 0，作为对比）
for view_label, orig_idx in [("视图1 (cam[0], 正常)", 1), ("视图0 (cam[4], 有问题)", 0)]:
    print(f"\n{'='*70}")
    print(f"追踪: {view_label}")
    print(f"{'='*70}")

    frame = tdata['frames'][orig_idx]
    c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
    scale = frame.get('scale', 1.0)

    print(f"\n[Step 0] 原始 transform_matrix (scale={scale})")
    print(f"  R:\n{c2w[:3,:3].numpy()}")
    print(f"  pos: {c2w[:3,3].numpy()}")

    # Step 1: 除以 scale
    c2w[:3, :] /= scale
    print(f"\n[Step 1] 除以 scale")
    print_cam("结果", c2w)

    # Step 2: Blender → OpenGL
    c2w[1] *= -1
    c2w[[1, 2]] = c2w[[2, 1]]
    c2w[:3, 1:3] *= -1
    print(f"\n[Step 2] Blender→OpenGL (c2w[1]*=-1, swap[1,2], c2w[:3,1:3]*=-1)")
    print_cam("结果", c2w)

    # Step 3: SVD 正交化
    R = c2w[:3, :3]
    U, _, Vt = torch.linalg.svd(R)
    c2w[:3, :3] = U @ Vt
    print(f"\n[Step 3] SVD 正交化")
    print_cam("结果", c2w)

    # Step 4: 缩放到 radius=1.5
    dist = torch.norm(c2w[:3, 3])
    if dist > 1e-6:
        c2w[:3, 3] *= (target_radius / dist)
    print(f"\n[Step 4] 缩放到 radius=1.5 (原 dist={dist:.6f})")
    print_cam("结果", c2w)

    # Step 5: Y+Z 翻转
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    print(f"\n[Step 5] Y+Z 翻转 (R列1,2取反)")
    print_cam("结果", c2w)

    # Step 6: 检测背对原点，look-at 重建
    pos_i = c2w[:3, 3]
    fwd_i = -c2w[:3, 2]
    to_origin_i = -pos_i / torch.norm(pos_i)
    dot_i = torch.dot(fwd_i, to_origin_i).item()
    if dot_i < 0:
        print(f"\n[Step 6] dot={dot_i:.4f} < 0，触发 look-at 重建")
        z_axis = -to_origin_i
        world_up = torch.tensor([0.0, 1.0, 0.0])
        x_axis = torch.linalg.cross(world_up, z_axis)
        x_norm = torch.norm(x_axis)
        print(f"    z_axis={z_axis.numpy()}")
        print(f"    cross(world_up, z_axis) = {x_axis.numpy()}, norm={x_norm:.6f}")
        if x_norm < 1e-6:
            world_up = torch.tensor([1.0, 0.0, 0.0])
            x_axis = torch.linalg.cross(world_up, z_axis)
            x_norm = torch.norm(x_axis)
        x_axis = x_axis / x_norm
        y_axis = torch.linalg.cross(z_axis, x_axis)
        c2w[:3, :3] = torch.stack([x_axis, y_axis, z_axis], dim=1)
        print_cam("重建后", c2w)
    else:
        print(f"\n[Step 6] dot={dot_i:.4f} >= 0，不需要重建")

print("\n" + "=" * 70)
print("结论")
print("=" * 70)
print("""
关键差异在 Step 2 (Blender→OpenGL):
- 视图1: 正常旋转矩阵 → 转换后 up 接近 [0, ±1, 0]
- 视图0: R=I (单位矩阵) → 转换后 up = [0, 0, 1] (特殊！)

然后 Step 5 (Y+Z翻转):
- 视图1: up 的 Y 分量翻转，仍然合理
- 视图0: up=[0,0,1] → up=[0,0,-1]，同时 fwd 翻反 → 触发 look-at 重建

Step 6 (look-at 重建):
- 视图0 的位置接近 Y 轴 (pos ≈ [0, -1.5, 0])
- cross(world_up=[0,1,0], z_axis≈[0,1,0]) ≈ 0 → 退化！
- 重建出来的 up 方向完全错误
""")
