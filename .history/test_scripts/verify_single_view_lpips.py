#!/usr/bin/env python3
"""
重新验证：在单个视角上，物体区域的真实LPIPS是多少
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from project_core import ConfigManager
from models import ModelManager
from data import DataManager
from training.finetuner import AutoFineTuner

# 加载模型
config_mgr = ConfigManager('configs/config.yaml')
config = config_mgr.config
config['model']['resume'] = 'output/test_train_overfit_scale_opacity/model_after_attack.pth'

model_mgr = ModelManager(config)
model_mgr.setup(device='cuda')
model = model_mgr.model
opt = model_mgr.opt

# 创建数据加载器
data_mgr = DataManager(config, opt)
data_mgr.setup_dataloaders(train=True, val=False, subset='target')
train_loader = data_mgr.train_loader

# 创建finetuner
training_cfg = config['training']
finetuner = AutoFineTuner(
    model=model,
    device='cuda',
    lr=training_cfg['lr'],
    weight_decay=training_cfg['weight_decay'],
    gradient_clip=training_cfg['gradient_clip'],
    mixed_precision='no',
    lambda_lpips=training_cfg.get('lambda_lpips', 1.0),
    gradient_accumulation_steps=training_cfg['gradient_accumulation_steps'],
)

# 获取第一个样本
first_sample = next(iter(train_loader))

print("=== 单个视角的物体区域LPIPS ===\n")

model.eval()
with torch.no_grad():
    data = finetuner._prepare_data(first_sample)

    gt_images = data['images_output']  # [B, V, 3, H, W]
    gt_masks = data['masks_output']  # [B, V, 1, H, W]

    # 前向传播
    results = model.forward(data, step_ratio=1.0)
    pred_images = results['images_pred']  # [B, V, 3, H, W]

    # 白色背景
    bg_color = torch.ones(3, device=pred_images.device).view(1, 1, 3, 1, 1)
    gt_images_white_bg = gt_images * gt_masks + bg_color * (1 - gt_masks)

    B, V = pred_images.shape[:2]

    # 分析每个视角
    for v in range(min(5, V)):  # 只看前5个视角
        print(f"\n=== 视角 {v} ===")

        gt_v = gt_images_white_bg[0, v:v+1]  # [1, 3, H, W]
        pred_v = pred_images[0, v:v+1]  # [1, 3, H, W]
        mask_v = gt_masks[0, v:v+1]  # [1, 1, H, W]

        obj_ratio = mask_v.mean().item()
        print(f"物体占比: {obj_ratio:.2%}")

        # 1. 全图LPIPS
        gt_256 = F.interpolate(gt_v * 2 - 1, (256, 256), mode='bilinear', align_corners=False)
        pred_256 = F.interpolate(pred_v * 2 - 1, (256, 256), mode='bilinear', align_corners=False)
        lpips_full = model.lpips_loss(gt_256, pred_256).item()
        print(f"全图LPIPS: {lpips_full:.6f}")

        # 2. 尝试放大物体区域
        # 找到物体的bounding box
        mask_np = mask_v[0, 0].cpu().numpy()
        rows = mask_np.sum(axis=1)
        cols = mask_np.sum(axis=0)

        row_indices = (rows > 0).nonzero()[0]
        col_indices = (cols > 0).nonzero()[0]

        if len(row_indices) > 0 and len(col_indices) > 0:
            y1, y2 = row_indices[0], row_indices[-1] + 1
            x1, x2 = col_indices[0], col_indices[-1] + 1

            # 裁剪物体区域
            gt_crop = gt_v[:, :, y1:y2, x1:x2]
            pred_crop = pred_v[:, :, y1:y2, x1:x2]
            mask_crop = mask_v[:, :, y1:y2, x1:x2]

            print(f"物体区域大小: {gt_crop.shape[2]}x{gt_crop.shape[3]}")
            print(f"裁剪后物体占比: {mask_crop.mean().item():.2%}")

            # 调整到256x256
            gt_crop_256 = F.interpolate(gt_crop * 2 - 1, (256, 256), mode='bilinear', align_corners=False)
            pred_crop_256 = F.interpolate(pred_crop * 2 - 1, (256, 256), mode='bilinear', align_corners=False)

            lpips_crop = model.lpips_loss(gt_crop_256, pred_crop_256).item()
            print(f"裁剪区域LPIPS: {lpips_crop:.6f}")

            # 3. 计算物体区域的平均亮度
            obj_pixels = gt_v[:, :, mask_v[0, 0] > 0.5]
            if obj_pixels.shape[1] > 0:
                print(f"物体区域RGB均值: {obj_pixels.mean(dim=1).squeeze().tolist()}")
                print(f"物体区域亮度: {obj_pixels.mean().item():.3f}")

print("\n完成！")
