#!/usr/bin/env python3
"""直接检查：Gaussian 在哪里？input 相机在哪里？"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['XFORMERS_DISABLED'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from project_core import ConfigManager
from models import ModelManager
from data import DataManager
from evaluation import Evaluator

config = ConfigManager('configs/config.yaml').config
model_mgr = ModelManager(config)
model_mgr.setup(device='cuda', apply_lora=False)
model = model_mgr.model
model.eval()

data_mgr = DataManager(config, model.opt)
data_mgr.setup_dataloaders(train=False, val=True, subset='source')
batch = next(iter(data_mgr.val_loader))

input_images = batch['input_images'].to('cuda')
input_transforms = batch['input_transforms']
sup_transforms = batch['supervision_transforms']

# 生成 Gaussian
with torch.no_grad():
    gaussians = Evaluator(model, 'cuda').generate_gaussians(input_images)

# Gaussian 参数：pos(3) + opacity(1) + scale(3) + rotation(4) + rgb(3) = 14
positions = gaussians[0, :, 0:3].cpu().numpy()  # [N, 3]
mean_pos = positions.mean(axis=0)
std_pos = positions.std(axis=0)
min_pos = positions.min(axis=0)
max_pos = positions.max(axis=0)

print("="*80)
print("Gaussian 位置统计")
print("="*80)
print(f"Mean: [{mean_pos[0]:+.4f}, {mean_pos[1]:+.4f}, {mean_pos[2]:+.4f}]")
print(f"Std:  [{std_pos[0]:.4f}, {std_pos[1]:.4f}, {std_pos[2]:.4f}]")
print(f"Min:  [{min_pos[0]:+.4f}, {min_pos[1]:+.4f}, {min_pos[2]:+.4f}]")
print(f"Max:  [{max_pos[0]:+.4f}, {max_pos[1]:+.4f}, {max_pos[2]:+.4f}]")
print(f"Range: [{max_pos[0]-min_pos[0]:.4f}, {max_pos[1]-min_pos[1]:.4f}, {max_pos[2]-min_pos[2]:.4f}]")

print("\n" + "="*80)
print("Input 相机位置（归一化后）")
print("="*80)
for i in range(input_transforms.shape[1]):
    pos = input_transforms[0, i, :3, 3].numpy()
    r = np.linalg.norm(pos)
    print(f"Input {i}: pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}], r={r:.4f}")

print("\n" + "="*80)
print("Supervision 相机位置（归一化后）")
print("="*80)
for i in range(min(6, sup_transforms.shape[1])):
    pos = sup_transforms[0, i, :3, 3].numpy()
    r = np.linalg.norm(pos)
    # 计算从相机到 Gaussian 中心的方向
    dir_to_gaussian = mean_pos - pos
    dir_to_gaussian = dir_to_gaussian / np.linalg.norm(dir_to_gaussian)
    # 相机的 forward 方向（OpenGL: -z）
    fwd = -sup_transforms[0, i, :3, 2].numpy()
    cos_angle = np.dot(fwd, dir_to_gaussian)
    print(f"Sup {i}: pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}], r={r:.4f}, "
          f"看向Gaussian cos={cos_angle:.3f}")

print("\n结论：")
print(f"- Gaussian 中心在 [{mean_pos[0]:+.3f}, {mean_pos[1]:+.3f}, {mean_pos[2]:+.3f}]")
print(f"- 如果接近原点 [0, 0, 0]，orbit_camera 方法理论上可行")
print(f"- 如果偏离原点，需要用 look-at 看向实际 Gaussian 中心")
