#!/usr/bin/env python3
"""
核心 debug：对比两种角度提取方法，找出错位原因
方法A: R0^T 旋转原始位置（当前代码）
方法B: 直接从归一化后的 supervision_transforms 位置提取角度
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
out_dir = 'output/debug_v2'
os.makedirs(out_dir, exist_ok=True)

# 加载模型
model_mgr = ModelManager(config)
model_mgr.setup(device=device, apply_lora=False)
model = model_mgr.model
model.eval()
opt = model.opt

# 加载数据
data_mgr = DataManager(config, opt)
data_mgr.setup_dataloaders(train=False, val=True, subset='source')
batch = next(iter(data_mgr.val_loader))

input_images = batch['input_images'].to(device)
gt_images = batch['supervision_images']
sup_el_r0t = batch['supervision_elevations']  # 方法A: R0^T
sup_az_r0t = batch['supervision_azimuths']
sup_transforms = batch['supervision_transforms']

B, V = gt_images.shape[:2]
print(f"B={B}, V_sup={V}, category={batch.get('category', ['?'])[0]}")

# 生成 Gaussian
with torch.no_grad():
    gaussians = Evaluator(model, device).generate_gaussians(input_images)

# 投影矩阵
tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj[0, 0] = 1 / tan_half_fov
proj[1, 1] = 1 / tan_half_fov
proj[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj[2, 3] = 1

def render_from_orbit(el, az, radius=1.5):
    """用 orbit_camera 渲染"""
    c2w = torch.from_numpy(orbit_camera(el, az, radius=radius, opengl=True))
    c2w = c2w.unsqueeze(0).to(device)
    c2w[:, :3, 1:3] *= -1
    cv = torch.inverse(c2w).transpose(1, 2)
    cvp = cv @ proj
    cp = -c2w[:, :3, 3]
    with torch.no_grad():
        r = model.gs.render(gaussians[0:1], cv.unsqueeze(0), cvp.unsqueeze(0), cp.unsqueeze(0))
    return r['image'].squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()

def save_img(arr, path):
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)

b = 0
print(f"\n{'='*80}")
print(f"对比两种角度提取方法")
print(f"{'='*80}")

for v in range(V):
    # 方法A: R0^T（当前代码）
    el_a = sup_el_r0t[b, v].item()
    az_a = sup_az_r0t[b, v].item()

    # 方法B: 直接从 supervision_transforms 的位置提取
    pos = sup_transforms[b, v, :3, 3].numpy()
    r = np.linalg.norm(pos)
    el_b = float(np.degrees(np.arcsin(np.clip(-pos[1] / r, -1, 1))))
    az_b = float(np.degrees(np.arctan2(pos[0], pos[2])))

    print(f"\nView {v}:")
    print(f"  方法A (R0^T):     el={el_a:+.1f}, az={az_a:+.1f}")
    print(f"  方法B (norm pos): el={el_b:+.1f}, az={az_b:+.1f}")
    print(f"  norm pos: [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}], r={r:.4f}")
    print(f"  角度差: Δel={el_b-el_a:+.1f}, Δaz={az_b-az_a:+.1f}")

    # 保存 GT
    gt = gt_images[b, v].clamp(0, 1).permute(1, 2, 0).numpy()
    save_img(gt, os.path.join(out_dir, f"v{v}_gt.png"))

    # 方法A 渲染
    img_a = render_from_orbit(el_a, az_a)
    save_img(img_a, os.path.join(out_dir, f"v{v}_methodA_el{el_a:+.0f}_az{az_a:+.0f}.png"))

    # 方法B 渲染
    img_b = render_from_orbit(el_b, az_b)
    save_img(img_b, os.path.join(out_dir, f"v{v}_methodB_el{el_b:+.0f}_az{az_b:+.0f}.png"))

# 也验证 input view 0 的归一化结果
print(f"\n{'='*80}")
print(f"验证 input view 0（应该是 el=0, az=0）")
print(f"{'='*80}")
inp_t = batch['input_transforms']
for i in range(inp_t.shape[1]):
    pos = inp_t[b, i, :3, 3].numpy()
    r = np.linalg.norm(pos)
    el = float(np.degrees(np.arcsin(np.clip(-pos[1] / r, -1, 1))))
    az = float(np.degrees(np.arctan2(pos[0], pos[2])))
    print(f"  Input {i}: pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}], r={r:.4f}, el={el:+.1f}, az={az:+.1f}")

print(f"\n图片保存在: {out_dir}/")
print("对比 v*_gt.png vs v*_methodA_*.png vs v*_methodB_*.png")
