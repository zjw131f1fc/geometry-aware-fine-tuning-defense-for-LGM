#!/usr/bin/env python3
"""检查归一化前后的 input 相机角度"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from project_core import ConfigManager
from models import ModelManager
from data import DataManager

config = ConfigManager('configs/config.yaml').config
model_mgr = ModelManager(config)
model_mgr.load_model(device='cpu')
opt = model_mgr.opt
del model_mgr

data_mgr = DataManager(config, opt)
data_mgr.setup_dataloaders(train=False, val=True, subset='source')

# 手动加载一个样本，在归一化前后检查角度
dataset = data_mgr.val_loader.dataset
sample_idx = 0

# 读取原始数据（需要修改 dataset 代码暂时跳过归一化）
# 这里直接从 batch 读取归一化后的数据
batch = next(iter(data_mgr.val_loader))
input_transforms = batch['input_transforms'][0]  # [4, 4, 4]

print("="*80)
print("Input 相机（归一化后）")
print("="*80)
for i in range(4):
    pos = input_transforms[i, :3, 3].numpy()
    r = np.linalg.norm(pos)
    el = np.degrees(np.arcsin(np.clip(-pos[1] / r, -1, 1)))
    az = np.degrees(np.arctan2(pos[0], pos[2]))
    print(f"Input {i}: pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}], "
          f"r={r:.3f}, el={el:+.1f}°, az={az:+.1f}°")

# 计算角度差
azimuths = []
for i in range(4):
    pos = input_transforms[i, :3, 3].numpy()
    r = np.linalg.norm(pos)
    az = np.degrees(np.arctan2(pos[0], pos[2]))
    azimuths.append(az)

print(f"\n方位角分布: {[f'{az:+.1f}°' for az in azimuths]}")
print(f"方位角范围: {max(azimuths) - min(azimuths):.1f}° (理想应该是 270°)")

# 检查 camera 0 的旋转矩阵
R0 = input_transforms[0, :3, :3].numpy()
print(f"\nCamera 0 旋转矩阵:")
print(R0)
print(f"是否接近单位矩阵: {np.allclose(R0, np.eye(3), atol=0.01)}")
