"""
测试LGM原始模型 - 使用官方示例
"""

import sys
sys.path.append('/mnt/huangjiaxin/3d-defense/LGM')
sys.path.append('/mnt/huangjiaxin/3d-defense')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from safetensors.torch import load_file

from core.options import config_defaults
from core.models import LGM
from core.utils import get_rays

# 使用big配置
opt = config_defaults['big']

print("=" * 80)
print("测试LGM原始模型")
print("=" * 80)

# 加载模型
print("\n[1] 加载模型...")
model = LGM(opt)

# 加载权重
ckpt = load_file('/mnt/huangjiaxin/3d-defense/LGM/pretrained/model_fp16_fixrot.safetensors', device='cpu')
model.load_state_dict(ckpt, strict=False)
model = model.to('cuda').eval()
print(f"[INFO] 模型加载完成")
print(f"[INFO] 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 加载测试图像
print("\n[2] 加载测试图像...")
img_path = '/mnt/huangjiaxin/3d-defense/LGM/data_test/anya_rgba.png'
img = Image.open(img_path)
print(f"[INFO] 图像尺寸: {img.size}")

# 移除背景（转为白色背景）
img = img.convert('RGBA')
img_rgb = img.convert('RGB')
img_array = np.array(img)
mask = img_array[:, :, 3:4] / 255.0
img_rgb_array = np.array(img_rgb) / 255.0
img_white_bg = img_rgb_array * mask + (1 - mask)
img_white_bg = (img_white_bg * 255).astype(np.uint8)
img = Image.fromarray(img_white_bg)

# 调整大小
img = img.resize((opt.input_size, opt.input_size), Image.BILINEAR)
img_tensor = TF.to_tensor(img)  # [3, H, W], [0, 1]

# ImageNet归一化
IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_tensor = (img_tensor - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD

print(f"[INFO] 图像张量形状: {img_tensor.shape}")
print(f"[INFO] 图像张量范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

# 创建4个视图（简单复制，实际应该用MVDream生成）
print("\n[3] 准备输入数据...")
input_images = []

# 创建4个正交视角的相机
from kiui.cam import orbit_camera
elevations = [0, 0, 0, 0]
azimuths = [0, 90, 180, 270]

cam_poses = []
for elev, azim in zip(elevations, azimuths):
    pose = orbit_camera(elev, azim, radius=1.5)
    cam_poses.append(torch.from_numpy(pose).float())

cam_poses = torch.stack(cam_poses, dim=0)  # [4, 4, 4]

# 为每个视图生成rays
for i in range(4):
    rays_o, rays_d = get_rays(cam_poses[i], opt.input_size, opt.input_size, opt.fovy)
    rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)  # [H, W, 6]
    rays_plucker = rays_plucker.permute(2, 0, 1)  # [6, H, W]

    # 拼接图像和rays
    img_with_rays = torch.cat([img_tensor, rays_plucker], dim=0)  # [9, H, W]
    input_images.append(img_with_rays)

input_images = torch.stack(input_images, dim=0).unsqueeze(0).to('cuda')  # [1, 4, 9, H, W]
print(f"[INFO] 输入形状: {input_images.shape}")

# 前向传播生成Gaussian
print("\n[4] 生成Gaussian...")
with torch.no_grad():
    gaussians = model.forward_gaussians(input_images)
    print(f"[INFO] Gaussian形状: {gaussians.shape}")
    print(f"[INFO] Gaussian范围: [{gaussians.min():.3f}, {gaussians.max():.3f}]")

# 保存PLY
print("\n[5] 保存PLY文件...")
output_dir = '/mnt/huangjiaxin/3d-defense/workspace_lgm_test'
os.makedirs(output_dir, exist_ok=True)
ply_path = os.path.join(output_dir, 'test_original.ply')

model.gs.save_ply(gaussians, ply_path)
print(f"[INFO] PLY已保存: {ply_path}")

# 渲染360度视频
print("\n[6] 渲染360度视频...")
from methods.evaluator import Evaluator

evaluator = Evaluator(model=model, device='cuda')
video_path = os.path.join(output_dir, 'test_original_360.mp4')
evaluator.render_360_video(gaussians, video_path, num_frames=90)
print(f"[INFO] 视频已保存: {video_path}")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
print(f"\n查看结果:")
print(f"  PLY文件: {ply_path}")
print(f"  360度视频: {video_path}")
print(f"\n可视化命令:")
print(f"  python LGM/gui.py big --test_path {ply_path}")
print(f"  python LGM/convert.py big --test_path {ply_path}")
