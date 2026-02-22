#!/usr/bin/env python3
"""
最小 debug：用 supervision_transforms 的位置 + look-at 原点渲染
对比 orbit_camera 和 look-at 两种方式
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

def look_at_camera(position, target=np.array([0,0,0]), up=np.array([0,1,0])):
    """构建 look-at 相机的 c2w 矩阵（OpenGL 约定）"""
    pos = np.array(position, dtype=np.float64)
    forward = target - pos
    forward = forward / np.linalg.norm(forward)  # -z in OpenGL
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up_actual = np.cross(right, forward)
    # OpenGL: x=right, y=up, z=-forward
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up_actual
    c2w[:3, 2] = -forward
    c2w[:3, 3] = pos
    return c2w

config = ConfigManager('configs/config.yaml').config
device = 'cuda'
out_dir = 'output/debug_lookat'
os.makedirs(out_dir, exist_ok=True)

model_mgr = ModelManager(config)
model_mgr.setup(device=device, apply_lora=False)
model = model_mgr.model
model.eval()
opt = model.opt

data_mgr = DataManager(config, opt)
data_mgr.setup_dataloaders(train=False, val=True, subset='source')
batch = next(iter(data_mgr.val_loader))

input_images = batch['input_images'].to(device)
gt_images = batch['supervision_images']
sup_transforms = batch['supervision_transforms']
sup_el = batch['supervision_elevations']
sup_az = batch['supervision_azimuths']

with torch.no_grad():
    gaussians = Evaluator(model, device).generate_gaussians(input_images)

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj[0, 0] = 1 / tan_half_fov
proj[1, 1] = 1 / tan_half_fov
proj[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj[2, 3] = 1

def render_c2w(c2w_np, gaussians_b):
    """从 c2w (OpenGL) 渲染"""
    c2w = torch.from_numpy(c2w_np).unsqueeze(0).to(device, dtype=torch.float32)
    c2w[:, :3, 1:3] *= -1  # OpenGL -> COLMAP
    cv = torch.inverse(c2w).transpose(1, 2)
    cvp = cv @ proj
    cp = -c2w[:, :3, 3]
    with torch.no_grad():
        r = model.gs.render(gaussians_b, cv.unsqueeze(0), cvp.unsqueeze(0), cp.unsqueeze(0))
    img = r['image'].squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    alpha_mean = r['image'].mean().item()
    return img, alpha_mean

b = 0
B, V = gt_images.shape[:2]
print(f"B={B}, V={V}")

for v in range(V):
    pos_norm = sup_transforms[b, v, :3, 3].numpy()
    r_norm = np.linalg.norm(pos_norm)
    fwd = sup_transforms[b, v, :3, 2].numpy()  # z column = -forward in OpenGL
    dir_to_origin = -pos_norm / r_norm
    cos_angle = np.dot(-fwd, dir_to_origin)  # -fwd = forward direction

    el_old = sup_el[b, v].item()
    az_old = sup_az[b, v].item()

    print(f"\nView {v}:")
    print(f"  norm pos: [{pos_norm[0]:+.3f}, {pos_norm[1]:+.3f}, {pos_norm[2]:+.3f}], r={r_norm:.3f}")
    print(f"  forward vs to-origin cos: {cos_angle:.4f} (1.0=完全对准)")
    print(f"  dataset el/az: el={el_old:+.1f}, az={az_old:+.1f}")

    # GT
    gt = gt_images[b, v].clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((gt * 255).astype(np.uint8)).save(f"{out_dir}/v{v}_gt.png")

    # 方法1: orbit_camera (dataset 的 el/az)
    c2w_orbit = orbit_camera(el_old, az_old, radius=opt.cam_radius, opengl=True)
    img1, a1 = render_c2w(c2w_orbit, gaussians[b:b+1])
    Image.fromarray((img1 * 255).astype(np.uint8)).save(f"{out_dir}/v{v}_orbit.png")

    # 方法2: look-at 从归一化位置看原点，半径=1.5
    dir_vec = pos_norm / r_norm
    pos_15 = dir_vec * 1.5
    c2w_lookat15 = look_at_camera(pos_15)
    img2, a2 = render_c2w(c2w_lookat15, gaussians[b:b+1])
    Image.fromarray((img2 * 255).astype(np.uint8)).save(f"{out_dir}/v{v}_lookat15.png")

    # 方法3: look-at 从归一化位置看原点，实际半径
    c2w_lookat_r = look_at_camera(pos_norm)
    img3, a3 = render_c2w(c2w_lookat_r, gaussians[b:b+1])
    Image.fromarray((img3 * 255).astype(np.uint8)).save(f"{out_dir}/v{v}_lookat_real.png")

    # 方法4: 直接用 supervision_transforms
    c2w_direct = sup_transforms[b, v].numpy()
    img4, a4 = render_c2w(c2w_direct, gaussians[b:b+1])
    Image.fromarray((img4 * 255).astype(np.uint8)).save(f"{out_dir}/v{v}_direct.png")

    print(f"  orbit alpha={a1:.3f}, lookat15 alpha={a2:.3f}, lookat_real alpha={a3:.3f}, direct alpha={a4:.3f}")

print(f"\n图片保存在: {out_dir}/")
print("对比: v*_gt.png vs v*_orbit.png vs v*_lookat15.png vs v*_lookat_real.png vs v*_direct.png")
