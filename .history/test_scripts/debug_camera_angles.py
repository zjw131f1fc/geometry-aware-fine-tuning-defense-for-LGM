#!/usr/bin/env python3
"""Debug: 验证 R0^T 旋转方法 vs 归一化变换后提取"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from kiui.cam import orbit_camera
from data import ObjaverseRenderedDataset

dataset = ObjaverseRenderedDataset(
    data_root='datas/objaverse_rendered',
    num_input_views=4, num_supervision_views=6, input_size=256,
    view_selector='orthogonal', max_samples=1, samples_per_object=1,
)

sample = dataset[0]
sup_el = sample['supervision_elevations']
sup_az = sample['supervision_azimuths']
input_transforms = sample['input_transforms']

print("INPUT cameras (normalized):")
for i in range(input_transforms.shape[0]):
    pos = input_transforms[i, :3, 3].numpy()
    print(f"  Input {i}: pos={pos}")

print("\nSUPERVISION angles (R0^T method):")
for v in range(sup_el.shape[0]):
    el = sup_el[v].item()
    az = sup_az[v].item()
    # orbit_camera 重建
    rebuilt = orbit_camera(el, az, radius=1.5, opengl=True)[:3, 3]
    print(f"  Sup {v}: el={el:+7.1f}, az={az:+7.1f} -> rebuilt pos=[{rebuilt[0]:+.3f}, {rebuilt[1]:+.3f}, {rebuilt[2]:+.3f}]")

# 验证 camera 0 应该是 el=0, az=0
print(f"\nCamera 0: el={sup_el[0] if sup_el.shape[0] > 0 else 'N/A'}")
print("(Input camera 0 should map to el=0, az=0)")

# 检查输入视图的角度
print("\nINPUT angles (should be ~0, ~90, ~180, ~270 for orthogonal):")
inp_el = sample.get('input_elevations', None)
if inp_el is None:
    # 手动从 input_transforms 提取（这些是归一化后的）
    for i in range(input_transforms.shape[0]):
        pos = input_transforms[i, :3, 3].numpy()
        r = np.linalg.norm(pos)
        if r > 1e-6:
            el = np.degrees(np.arcsin(np.clip(-pos[1]/r, -1, 1)))
            az = np.degrees(np.arctan2(pos[0], pos[2]))
            print(f"  Input {i}: el={el:+7.1f}, az={az:+7.1f}, r={r:.3f}")
