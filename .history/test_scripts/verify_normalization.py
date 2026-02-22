#!/usr/bin/env python3
"""验证归一化变换到底做了什么"""
import torch
import numpy as np

def normalize_poses(cam_poses, target_radius=1.5):
    """LGM 的归一化逻辑"""
    # Step 1: 逐相机缩放到 target_radius
    for i in range(cam_poses.shape[0]):
        dist = torch.norm(cam_poses[i, :3, 3])
        if dist > 1e-6:
            cam_poses[i, :3, 3] *= target_radius / dist

    print("归一化前（缩放到半径 1.5 后）：")
    for i in range(cam_poses.shape[0]):
        pos = cam_poses[i, :3, 3].numpy()
        r = np.linalg.norm(pos)
        print(f"  Camera {i}: pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}], r={r:.3f}")

    # Step 2: 姿态归一化
    T_target = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, target_radius],
        [0, 0, 0, 1]
    ], dtype=torch.float32)
    transform = T_target @ torch.inverse(cam_poses[0])
    cam_poses_normalized = transform.unsqueeze(0) @ cam_poses

    print("\n归一化后：")
    for i in range(cam_poses_normalized.shape[0]):
        pos = cam_poses_normalized[i, :3, 3].numpy()
        r = np.linalg.norm(pos)
        print(f"  Camera {i}: pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}], r={r:.3f}")

    return cam_poses_normalized

# 测试 1: 标准正交相机（LGM 训练数据的理想情况）
print("="*80)
print("测试 1: 标准正交相机（camera 0 在 [0,0,1.5]，单位旋转）")
print("="*80)
cam_poses_standard = torch.tensor([
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 1.5],
     [0, 0, 0, 1]],  # Camera 0: [0, 0, 1.5], az=0
    [[0, 0, 1, 1.5],
     [0, 1, 0, 0],
     [-1, 0, 0, 0],
     [0, 0, 0, 1]],  # Camera 1: [1.5, 0, 0], az=90
    [[-1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, -1, -1.5],
     [0, 0, 0, 1]],  # Camera 2: [0, 0, -1.5], az=180
    [[0, 0, -1, -1.5],
     [0, 1, 0, 0],
     [1, 0, 0, 0],
     [0, 0, 0, 1]],  # Camera 3: [-1.5, 0, 0], az=270
], dtype=torch.float32)
normalize_poses(cam_poses_standard.clone())

# 测试 2: 非标准相机（camera 0 不在标准位置）
print("\n" + "="*80)
print("测试 2: 非标准相机（camera 0 在 [1.06, 0.75, 1.06]，非单位旋转）")
print("="*80)
# 构造一个 camera 0 在 45° 方位角的情况
angle = np.pi / 4  # 45 degrees
cam_poses_nonstandard = torch.tensor([
    [[np.cos(angle), 0, -np.sin(angle), 1.06],
     [0, 1, 0, 0.75],
     [np.sin(angle), 0, np.cos(angle), 1.06],
     [0, 0, 0, 1]],  # Camera 0: 偏离标准位置
    [[1, 0, 0, 1.5],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],  # Camera 1
    [[0, 0, 1, 0],
     [0, 1, 0, 0],
     [-1, 0, 0, -1.5],
     [0, 0, 0, 1]],  # Camera 2
    [[-1, 0, 0, -1.5],
     [0, 1, 0, 0],
     [0, 0, -1, 0],
     [0, 0, 0, 1]],  # Camera 3
], dtype=torch.float32)
normalize_poses(cam_poses_nonstandard.clone())

print("\n" + "="*80)
print("结论：")
print("="*80)
print("- 测试 1: camera 0 在标准位置 → 归一化≈单位变换 → 半径保持 1.5")
print("- 测试 2: camera 0 偏离标准位置 → 归一化破坏球面 → 半径变化")
print("- 这解释了为什么我们的数据归一化后半径变成 3.19/4.26")
