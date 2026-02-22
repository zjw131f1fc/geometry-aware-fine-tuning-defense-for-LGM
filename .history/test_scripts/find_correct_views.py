#!/usr/bin/env python3
"""
找到 Objaverse 数据中对应标准正交视角的视图
"""

import json
import numpy as np

transforms_path = "/mnt/huangjiaxin/3d-defense/datas/objaverse_rendered/e4041c753202476ea7051121ae33ea7d/render/transforms.json"
with open(transforms_path, 'r') as f:
    transforms_data = json.load(f)

print("=" * 80)
print("查找对应标准正交视角的视图")
print("=" * 80)

# 计算所有视图的方位角
view_info = []
for i, frame in enumerate(transforms_data['frames']):
    c2w = np.array(frame['transform_matrix'])
    scale = frame.get('scale', 1.0)

    # 除以 scale 得到真实位置
    pos = c2w[:3, 3] / scale
    x, y, z = pos[0], pos[1], pos[2]

    # 计算方位角和仰角
    azimuth = np.arctan2(y, x) * 180 / np.pi
    if azimuth < 0:
        azimuth += 360
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    radius = np.sqrt(x**2 + y**2 + z**2)

    view_info.append({
        'index': i,
        'file': frame['file_path'],
        'azimuth': azimuth,
        'elevation': elevation,
        'radius': radius,
        'rotation': frame.get('rotation', 0)
    })

# 找到最接近 0°, 90°, 180°, 270° 的视图
target_azimuths = [0, 90, 180, 270]
selected_views = []

print("\n查找最接近目标角度的视图:")
print("-" * 80)

for target_az in target_azimuths:
    min_diff = 360
    best_view = None

    for view in view_info:
        diff = min(abs(view['azimuth'] - target_az),
                   abs(view['azimuth'] - target_az + 360),
                   abs(view['azimuth'] - target_az - 360))
        if diff < min_diff:
            min_diff = diff
            best_view = view

    selected_views.append(best_view)
    print(f"\n目标 {target_az}°:")
    print(f"  最佳匹配: {best_view['file']} (index {best_view['index']})")
    print(f"  实际方位角: {best_view['azimuth']:.2f}°")
    print(f"  仰角: {best_view['elevation']:.2f}°")
    print(f"  半径: {best_view['radius']:.2f}")
    print(f"  角度差: {min_diff:.2f}°")

print("\n" + "=" * 80)
print("建议的视图索引")
print("=" * 80)
print(f"使用视图: {[v['index'] for v in selected_views]}")
print(f"对应文件: {[v['file'] for v in selected_views]}")

# 检查是否所有视图的仰角和半径都一致
elevations = [v['elevation'] for v in selected_views]
radii = [v['radius'] for v in selected_views]

print(f"\n仰角范围: {min(elevations):.2f}° ~ {max(elevations):.2f}°")
print(f"半径范围: {min(radii):.2f} ~ {max(radii):.2f}")

if max(elevations) - min(elevations) > 1:
    print("\n⚠️  警告：仰角不一致！")
if max(radii) - min(radii) > 10:
    print("⚠️  警告：半径不一致！")
