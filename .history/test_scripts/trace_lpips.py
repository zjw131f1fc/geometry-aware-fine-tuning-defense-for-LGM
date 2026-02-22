#!/usr/bin/env python3
"""
追踪LPIPS计算过程：
1. Hook VGG网络的每一层
2. 计算每一层的特征差异
3. 可视化哪些区域对LPIPS贡献最大
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np

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

print("=== 追踪LPIPS计算过程 ===\n")

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

    print(f"图像形状: {pred_images.shape}")
    print(f"物体占比: {gt_masks.mean():.2%}")
    print(f"Pred是否全白: {(pred_images == 1.0).all().item()}")

    # 准备LPIPS输入
    B, V = pred_images.shape[:2]
    gt_flat = gt_images_white_bg.view(-1, 3, 512, 512)
    pred_flat = pred_images.view(-1, 3, 512, 512)

    # 下采样到256x256
    gt_256 = F.interpolate(gt_flat * 2 - 1, (256, 256), mode='bilinear', align_corners=False)
    pred_256 = F.interpolate(pred_flat * 2 - 1, (256, 256), mode='bilinear', align_corners=False)

    # Hook LPIPS网络的每一层
    lpips_net = model.lpips_loss

    print(f"\n=== LPIPS网络结构 ===")
    print(f"网络类型: {lpips_net.net}")

    # 获取LPIPS的层
    layers = []
    for name, module in lpips_net.named_modules():
        if 'lin' in name:  # LPIPS的线性层
            layers.append(name)

    print(f"LPIPS层数: {len(layers)}")

    # 计算LPIPS（带详细输出）
    print(f"\n=== 计算LPIPS ===")

    # 直接调用LPIPS
    lpips_value = lpips_net(gt_256, pred_256)
    print(f"总LPIPS: {lpips_value.mean().item():.6f}")

    # 分析每个视角的LPIPS
    print(f"\n=== 每个视角的LPIPS ===")
    for v in range(V):
        gt_v = gt_256[v:v+1]
        pred_v = pred_256[v:v+1]
        lpips_v = lpips_net(gt_v, pred_v)

        # 获取这个视角的mask
        mask_v = gt_masks[0, v]  # [1, H, W]
        obj_ratio = mask_v.mean().item()

        print(f"视角 {v}: LPIPS={lpips_v.item():.6f}, 物体占比={obj_ratio:.2%}")

    # 尝试分块计算LPIPS
    print(f"\n=== 分块分析（4x4网格）===")
    H, W = 256, 256
    grid_size = 4
    block_h = H // grid_size
    block_w = W // grid_size

    # 下采样mask到256x256
    mask_256 = F.interpolate(gt_masks.view(-1, 1, 512, 512), (256, 256), mode='bilinear', align_corners=False)

    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * block_h, (i + 1) * block_h
            x1, x2 = j * block_w, (j + 1) * block_w

            # 提取这个块
            gt_block = gt_256[:, :, y1:y2, x1:x2]
            pred_block = pred_256[:, :, y1:y2, x1:x2]
            mask_block = mask_256[:, :, y1:y2, x1:x2]

            # 计算这个块的LPIPS
            lpips_block = lpips_net(gt_block, pred_block).mean()
            obj_ratio_block = mask_block.mean().item()

            if obj_ratio_block > 0.01:  # 只显示有物体的块
                print(f"  块[{i},{j}]: LPIPS={lpips_block.item():.6f}, 物体占比={obj_ratio_block:.2%}")

print("\n完成！")
