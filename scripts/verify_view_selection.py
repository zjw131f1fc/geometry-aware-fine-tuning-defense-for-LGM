#!/usr/bin/env python3
"""
验证当前数据加载器选择的视图
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import ObjaverseRenderedDataset

# 创建数据集
dataset = ObjaverseRenderedDataset(
    data_root='datas/objaverse_rendered',
    num_input_views=4,
    num_supervision_views=4,
    input_size=256,
    view_selector='orthogonal',
    angle_offset=0.0,
    max_samples=1,
    samples_per_object=1,
)

print("=" * 80)
print("验证视图选择")
print("=" * 80)

# 加载一个样本
sample = dataset[0]
uuid = sample['uuid']

print(f"\nUUID: {uuid}")
print(f"Input images shape: {sample['input_images'].shape}")

# 读取 transforms.json 查看选择了哪些视图
import json
transforms_path = f"datas/objaverse_rendered/{uuid}/render/transforms.json"
with open(transforms_path, 'r') as f:
    transforms_data = json.load(f)

# 重新运行视图选择逻辑
from data.dataset import OrthogonalViewSelector
selector = OrthogonalViewSelector(angle_offset=0.0)
input_indices, _ = selector.select_views(
    len(transforms_data['frames']),
    4,
    transforms_data,
    sample_idx=0
)

print(f"\n选择的视图索引: {input_indices}")

# 显示每个视图的详细信息
import numpy as np
for i, idx in enumerate(input_indices):
    frame = transforms_data['frames'][idx]
    mat = frame['transform_matrix']
    scale = frame.get('scale', 1.0)

    x, y, z = mat[0][3] / scale, mat[1][3] / scale, mat[2][3] / scale
    azimuth = np.arctan2(y, x) * 180 / np.pi
    if azimuth < 0:
        azimuth += 360

    print(f"\n视图 {i} (index {idx}, file {frame['file_path']}):")
    print(f"  方位角: {azimuth:.2f}°")
    print(f"  期望: {i * 90}°")
