#!/usr/bin/env python3
"""
直接 debug：逐视图对比 GT vs 渲染，打印角度
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['XFORMERS_DISABLED'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
from kiui.cam import orbit_camera

from project_core import ConfigManager
from models import ModelManager
from data import DataManager
from evaluation import Evaluator

config = ConfigManager('configs/config.yaml').config
device = 'cuda'
out_dir = 'output/debug_views'
os.makedirs(out_dir, exist_ok=True)

# 加载模型
model_mgr = ModelManager(config)
model_mgr.setup(device=device, apply_lora=False)
model = model_mgr.model
model.eval()
opt = model.opt

# 加载 source 数据（跟 pipeline 一样）
data_mgr = DataManager(config, opt)
data_mgr.setup_dataloaders(train=False, val=True, subset='source')
val_loader = data_mgr.val_loader

# 取第一个 batch
batch = next(iter(val_loader))
input_images = batch['input_images'].to(device)
gt_images = batch['supervision_images']  # [B, V, 3, H, W]
sup_el = batch['supervision_elevations']  # [B, V]
sup_az = batch['supervision_azimuths']    # [B, V]
sup_transforms = batch['supervision_transforms']  # [B, V, 4, 4]

B, V = gt_images.shape[:2]
print(f"Batch: B={B}, V_sup={V}")
print(f"Category: {batch.get('category', ['?'])[0]}")

# 生成 Gaussian
with torch.no_grad():
    gaussians = Evaluator(model, device).generate_gaussians(input_images)

# 准备投影矩阵
tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

print(f"\n{'='*80}")
print(f"逐视图对比")
print(f"{'='*80}")

b = 0  # 第一个样本
for v in range(V):
    el = sup_el[b, v].item()
    az = sup_az[b, v].item()

    # 归一化后的 supervision_transforms 位置
    norm_pos = sup_transforms[b, v, :3, 3].numpy()
    norm_r = np.linalg.norm(norm_pos)

    print(f"\nView {v}:")
    print(f"  Extracted: el={el:+.1f}, az={az:+.1f}")
    print(f"  Normalized pos: [{norm_pos[0]:+.3f}, {norm_pos[1]:+.3f}, {norm_pos[2]:+.3f}], r={norm_r:.3f}")

    # 保存 GT
    gt = gt_images[b, v].clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((gt * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"v{v}_gt.png"))

    # 用 orbit_camera 渲染
    cam_pose_np = orbit_camera(el, az, radius=opt.cam_radius, opengl=True)
    cam_pose = torch.from_numpy(cam_pose_np).unsqueeze(0).to(device)
    cam_pose[:, :3, 1:3] *= -1
    cam_view = torch.inverse(cam_pose).transpose(1, 2)
    cam_view_proj = cam_view @ proj_matrix
    cam_pos = -cam_pose[:, :3, 3]

    with torch.no_grad():
        result = model.gs.render(
            gaussians[b:b+1],
            cam_view.unsqueeze(0),
            cam_view_proj.unsqueeze(0),
            cam_pos.unsqueeze(0),
        )
    rendered = result['image'].squeeze(1).squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    Image.fromarray((rendered * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"v{v}_rendered_el{el:+.0f}_az{az:+.0f}.png"))

    # 用 supervision_transforms 直接渲染（对比）
    cam_pose2 = sup_transforms[b, v].unsqueeze(0).to(device, dtype=torch.float32).clone()
    cam_pose2[:, :3, 1:3] *= -1
    cam_view2 = torch.inverse(cam_pose2).transpose(1, 2)
    cam_view_proj2 = cam_view2 @ proj_matrix
    cam_pos2 = -cam_pose2[:, :3, 3]

    with torch.no_grad():
        result2 = model.gs.render(
            gaussians[b:b+1],
            cam_view2.unsqueeze(0),
            cam_view_proj2.unsqueeze(0),
            cam_pos2.unsqueeze(0),
        )
    rendered2 = result2['image'].squeeze(1).squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    alpha2 = result2['image'].squeeze(1).squeeze(0).mean().item()
    Image.fromarray((rendered2 * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"v{v}_direct_transform.png"))
    print(f"  Direct transform render mean: {alpha2:.4f} ({'WHITE' if alpha2 > 0.99 else 'OK'})")

# 也渲染标准视角作为参考
print(f"\n{'='*80}")
print(f"标准视角渲染（参考）")
print(f"{'='*80}")
for az in [0, 90, 180, 270]:
    cam_pose_np = orbit_camera(0, az, radius=1.5, opengl=True)
    cam_pose = torch.from_numpy(cam_pose_np).unsqueeze(0).to(device)
    cam_pose[:, :3, 1:3] *= -1
    cam_view = torch.inverse(cam_pose).transpose(1, 2)
    cam_view_proj = cam_view @ proj_matrix
    cam_pos = -cam_pose[:, :3, 3]
    with torch.no_grad():
        result = model.gs.render(
            gaussians[b:b+1], cam_view.unsqueeze(0),
            cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0),
        )
    rendered = result['image'].squeeze(1).squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    Image.fromarray((rendered * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"standard_az{az}.png"))
    print(f"  az={az}: saved")

print(f"\n所有图片保存在: {out_dir}/")
print("对比 v*_gt.png 和 v*_rendered_*.png 看方向是否匹配")
