#!/usr/bin/env python3
"""
使用 LGM 原始推理代码测试
"""

import sys
import os

# 添加 LGM 路径
lgm_path = '/mnt/huangjiaxin/3d-defense/third_party/LGM'
sys.path.insert(0, lgm_path)

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file

from core.options import config_defaults
from core.models import LGM
from kiui.cam import orbit_camera

# 配置
device = 'cuda'
model_size = 'big'
resume_path = '/mnt/huangjiaxin/3d-defense/third_party/LGM/pretrained/model_fp16_fixrot.safetensors'

# 加载模型
print("加载模型...")
opt = config_defaults[model_size]
model = LGM(opt)

if resume_path.endswith('safetensors'):
    ckpt = load_file(resume_path, device='cpu')
else:
    ckpt = torch.load(resume_path, map_location='cpu')
model.load_state_dict(ckpt, strict=False)

model = model.half().to(device)
model.eval()
print("✓ 模型加载完成")

# 准备 rays embeddings
rays_embeddings = model.prepare_default_rays(device)
print(f"✓ Rays embeddings: {rays_embeddings.shape}")

# 加载测试图像（使用我们数据集中的图像）
print("\n加载测试图像...")
data_root = '/mnt/huangjiaxin/3d-defense/datas/objaverse_rendered'
uuid = 'e4041c753202476ea7051121ae33ea7d'
images_dir = os.path.join(data_root, uuid, 'render/images')

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# 加载 4 个视图（假设是正交视图）
mv_images = []
for i in [0, 12, 25, 37]:  # 大约对应 0°, 90°, 180°, 270°
    img_path = os.path.join(images_dir, f'r_{i}.png')
    img = Image.open(img_path).convert('RGBA')
    img = img.resize((256, 256), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0

    # RGBA 转 RGB（白色背景）
    if img_np.shape[-1] == 4:
        img_np = img_np[..., :3] * img_np[..., 3:4] + (1 - img_np[..., 3:4])

    mv_images.append(img_np)

mv_images = np.stack(mv_images, axis=0)  # [4, 256, 256, 3]
print(f"✓ 图像加载完成: {mv_images.shape}")

# 转换为 tensor 并归一化
input_image = torch.from_numpy(mv_images).permute(0, 3, 1, 2).float().to(device)  # [4, 3, 256, 256]
input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

# 拼接 rays
input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0)  # [1, 4, 9, H, W]
print(f"✓ 输入准备完成: {input_image.shape}")

# 生成 Gaussians
print("\n生成 Gaussians...")
with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        gaussians = model.forward_gaussians(input_image)

print(f"✓ Gaussians: {gaussians.shape}")
print(f"  Position: mean={gaussians[0, :, 0:3].mean().item():.4f}, std={gaussians[0, :, 0:3].std().item():.4f}")
print(f"  Opacity: mean={gaussians[0, :, 3].mean().item():.4f}, std={gaussians[0, :, 3].std().item():.4f}")
print(f"  RGB: mean={gaussians[0, :, 11:].mean().item():.4f}, std={gaussians[0, :, 11:].std().item():.4f}")

# 渲染测试
print("\n渲染测试...")
output_dir = 'output/lgm_original_test'
os.makedirs(output_dir, exist_ok=True)

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

for azimuth in [0, 90, 180, 270]:
    cam_pose = torch.from_numpy(orbit_camera(0, azimuth, radius=1.5, opengl=True)).unsqueeze(0).to(device)
    cam_pose[:, :3, 1:3] *= -1

    cam_view = torch.inverse(cam_pose).transpose(1, 2)
    cam_view_proj = cam_view @ proj_matrix
    cam_pos = -cam_pose[:, :3, 3]

    image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0))['image']
    image = image.squeeze(1).permute(0, 2, 3, 1).contiguous().float().cpu().numpy()
    image = (image * 255).astype(np.uint8)

    Image.fromarray(image[0]).save(os.path.join(output_dir, f'render_{azimuth:03d}.png'))
    print(f"  ✓ 渲染 {azimuth}°")

print(f"\n完成！结果保存在: {output_dir}")
