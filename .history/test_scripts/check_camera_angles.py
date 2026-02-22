#!/usr/bin/env python3
"""检查 Objaverse 数据的原始相机角度分布"""
import os, sys, json
import numpy as np

data_root = '/mnt/huangjiaxin/3d-defense/datas/objaverse_rendered'
dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

print(f"检查 {len(dirs[:10])} 个物体的相机角度...")

for uuid in dirs[:10]:
    transforms_path = os.path.join(data_root, uuid, 'render/transforms.json')
    if not os.path.exists(transforms_path):
        continue

    with open(transforms_path, 'r') as f:
        data = json.load(f)

    frames = data['frames']
    azimuths = []

    for frame in frames:
        c2w = np.array(frame['transform_matrix'])
        pos = c2w[:3, 3]
        az = np.degrees(np.arctan2(pos[0], pos[2]))
        azimuths.append(az)

    # 找最接近 0/90/180/270 的4个角度
    targets = [0, 90, 180, -90]  # -90 = 270
    selected = []
    for target in targets:
        diffs = [min(abs(az - target), abs(az - target + 360), abs(az - target - 360)) for az in azimuths]
        idx = np.argmin(diffs)
        selected.append((azimuths[idx], diffs[idx]))

    print(f"\n{uuid}:")
    print(f"  选中角度: {[f'{az:+.1f}°' for az, _ in selected]}")
    print(f"  偏离标准: {[f'{diff:.1f}°' for _, diff in selected]}")
    print(f"  Camera 0 偏离: {selected[0][1]:.1f}° (关键！)")
