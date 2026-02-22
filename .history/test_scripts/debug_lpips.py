#!/usr/bin/env python3
"""
在第一个样本上计算训练时的LPIPS，看看是否和渲染结果一致
"""

import os
import sys
import torch
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from project_core import ConfigManager
from models import ModelManager
from data import DataManager
from training.finetuner import AutoFineTuner

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='模型路径')
parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件')
args = parser.parse_args()

# 加载配置
print(f"加载配置: {args.config}")
config_mgr = ConfigManager(args.config)
config = config_mgr.config

# 加载模型
config['model']['resume'] = args.model
print(f"加载模型: {args.model}")

model_mgr = ModelManager(config)
model_mgr.setup(device='cuda')
model = model_mgr.model
opt = model_mgr.opt

# 创建数据加载器
print("创建数据加载器...")
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
print("\n获取第一个样本...")
first_sample = next(iter(train_loader))

# 在评估模式下计算
print("\n=== 评估模式（model.eval()）===")
model.eval()
with torch.no_grad():
    data = finetuner._prepare_data(first_sample)
    results = model.forward(data, step_ratio=1.0)

    pred_images = results.get('images_pred')
    gt_images = data['images_output']

    if pred_images is not None:
        print(f"Pred images shape: {pred_images.shape}")
        print(f"Pred images mean: {pred_images.mean().item():.6f}")
        print(f"Pred images std: {pred_images.std().item():.6f}")
        print(f"Pred images min: {pred_images.min().item():.6f}")
        print(f"Pred images max: {pred_images.max().item():.6f}")

        # 检查是否全白
        is_all_white = (pred_images == 1.0).all().item()
        print(f"是否全白（所有像素=1.0）: {is_all_white}")

        lpips_loss = results.get('loss_lpips', torch.tensor(0.0))
        psnr = results.get('psnr', torch.tensor(0.0))
        print(f"LPIPS: {lpips_loss.item():.6f}")
        print(f"PSNR: {psnr.item():.2f}")

# 在训练模式下计算（会更新梯度）
print("\n=== 训练模式（model.train()）===")
model.train()
loss_dict_train, updated = finetuner.train_step(first_sample)
print(f"Loss: {loss_dict_train['loss']:.6f}")
print(f"LPIPS: {loss_dict_train.get('loss_lpips', 0):.6f}")
print(f"PSNR: {loss_dict_train.get('psnr', 0):.2f}")
print(f"Masked PSNR: {loss_dict_train.get('masked_psnr', 0):.2f}")
print(f"Updated: {updated}")

print("\n完成！")
