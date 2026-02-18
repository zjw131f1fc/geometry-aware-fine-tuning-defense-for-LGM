"""
陷阱效果验证脚本（简化版）

只在 target domain 上训练陷阱损失，验证 Gaussian 参数是否退化
"""

import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from project_core import ConfigManager
from models import ModelManager
from data import DataManager
from evaluation import Evaluator
from methods.trap_losses import ScaleAnisotropyLoss, PositionCollapseLoss


def train_trap_minimal(
    model,
    data_loader,
    trap_loss_fn,
    optimizer,
    device='cuda',
    num_epochs=5,
    save_dir='output/trap_minimal',
):
    """
    简化的陷阱训练

    Args:
        model: LGM 模型
        data_loader: 数据加载器
        trap_loss_fn: 陷阱损失函数
        optimizer: 优化器
        device: 设备
        num_epochs: 训练轮数
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    model.train()

    # 训练历史
    history = {
        'epoch': [],
        'loss': [],
    }

    print(f"\n{'='*60}")
    print(f"开始训练（{num_epochs} epochs）")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            # 准备数据
            input_images = batch['input_images'].to(device)

            # 前向传播
            gaussians = model.forward_gaussians(input_images)

            # 计算陷阱损失
            loss = trap_loss_fn(gaussians)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录
            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 平均损失
        avg_loss = epoch_loss / num_batches
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # 保存训练历史
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n[训练] 历史已保存: {history_path}")

    return history


def evaluate_gaussians(model, data_loader, device='cuda', num_samples=5):
    """
    评估生成的 Gaussian 参数

    Args:
        model: LGM 模型
        data_loader: 数据加载器
        device: 设备
        num_samples: 评估样本数

    Returns:
        stats: 统计信息
    """
    model.eval()

    all_stats = {
        'scale_anisotropy': [],
        'scale_mean': [],
        'scale_std': [],
        'position_variance': [],
    }

    with torch.no_grad():
        sample_count = 0
        for batch in data_loader:
            if sample_count >= num_samples:
                break

            input_images = batch['input_images'].to(device)
            gaussians = model.forward_gaussians(input_images)

            # 提取参数
            position = gaussians[..., 0:3]  # [B, N, 3]
            scale = gaussians[..., 4:7]     # [B, N, 3]

            B = gaussians.shape[0]

            for b in range(B):
                # Scale 各向异性
                scale_b = scale[b]  # [N, 3]
                scale_sq = scale_b ** 2
                max_scale = scale_sq.max(dim=-1)[0]
                min_scale = scale_sq.min(dim=-1)[0]
                anisotropy = (max_scale / (min_scale + 1e-6)).mean().item()
                all_stats['scale_anisotropy'].append(anisotropy)

                # Scale 统计
                all_stats['scale_mean'].append(scale_b.mean().item())
                all_stats['scale_std'].append(scale_b.std().item())

                # Position 方差
                pos_b = position[b]  # [N, 3]
                all_stats['position_variance'].append(pos_b.var(dim=0).mean().item())

            sample_count += B

    # 计算平均统计
    avg_stats = {k: sum(v) / len(v) for k, v in all_stats.items()}
    return avg_stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/attack_config.yaml')
    parser.add_argument('--gpu', type=str, default='7')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--trap_type', type=str, default='scale', choices=['scale', 'position'])
    parser.add_argument('--target_layers', type=str, default='conv.weight,unet.conv_in.weight')
    parser.add_argument('--output_dir', type=str, default='output/trap_minimal')
    args = parser.parse_args()

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda'

    print(f"\n{'='*60}")
    print(f"陷阱效果验证（简化版）")
    print(f"{'='*60}")
    print(f"配置文件: {args.config}")
    print(f"GPU: {args.gpu}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"学习率: {args.lr}")
    print(f"陷阱类型: {args.trap_type}")
    print(f"目标层: {args.target_layers}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}\n")

    # 加载配置
    config_mgr = ConfigManager(args.config)
    config = config_mgr.config

    # 加载模型
    print("[1/6] 加载预训练模型...")
    model_mgr = ModelManager(config)
    model_mgr.load_model(device=device, dtype=torch.float32)
    model = model_mgr.model

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 只解冻目标层
    target_layer_names = args.target_layers.split(',')
    trainable_params = []
    for name, param in model.named_parameters():
        for target_name in target_layer_names:
            if target_name in name:
                param.requires_grad = True
                trainable_params.append(param)
                print(f"  解冻层: {name}")
                break

    print(f"  可训练参数数: {sum(p.numel() for p in trainable_params):,}")

    # 加载数据
    print("\n[2/6] 加载数据...")
    data_mgr = DataManager(config, model_mgr.opt)
    data_mgr.setup_dataloaders(train=True, val=False)
    data_loader = data_mgr.train_loader

    # 创建陷阱损失
    print(f"\n[3/6] 创建陷阱损失（{args.trap_type}）...")
    if args.trap_type == 'scale':
        trap_loss_fn = ScaleAnisotropyLoss()
    elif args.trap_type == 'position':
        trap_loss_fn = PositionCollapseLoss()
    else:
        raise ValueError(f"未知的陷阱类型: {args.trap_type}")

    # 创建优化器
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # 评估训练前的 Gaussian
    print("\n[4/6] 评估训练前的 Gaussian...")
    stats_before = evaluate_gaussians(model, data_loader, device, num_samples=10)
    print("训练前统计:")
    for k, v in stats_before.items():
        print(f"  {k}: {v:.6f}")

    # 训练
    print("\n[5/6] 开始训练...")
    history = train_trap_minimal(
        model=model,
        data_loader=data_loader,
        trap_loss_fn=trap_loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=args.output_dir,
    )

    # 评估训练后的 Gaussian
    print("\n[6/6] 评估训练后的 Gaussian...")
    stats_after = evaluate_gaussians(model, data_loader, device, num_samples=10)
    print("训练后统计:")
    for k, v in stats_after.items():
        print(f"  {k}: {v:.6f}")

    # 保存对比结果
    comparison = {
        'before': stats_before,
        'after': stats_after,
        'change': {k: stats_after[k] - stats_before[k] for k in stats_before.keys()}
    }
    comparison_path = os.path.join(args.output_dir, 'comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[对比] 已保存: {comparison_path}")

    # 保存模型
    model_path = os.path.join(args.output_dir, 'model_trap.pth')
    torch.save(model.state_dict(), model_path)
    print(f"[模型] 已保存: {model_path}")

    # 生成 PLY 文件（训练后）
    print("\n[生成] 保存 PLY 文件...")
    evaluator = Evaluator(model, device)
    sample_count = 0
    for batch in data_loader:
        if sample_count >= 3:
            break
        input_images = batch['input_images'].to(device)
        gaussians = evaluator.generate_gaussians(input_images)

        B = gaussians.shape[0]
        for b in range(B):
            if sample_count >= 3:
                break
            ply_path = os.path.join(args.output_dir, f'sample_{sample_count}_after.ply')
            evaluator.save_ply(gaussians[b:b+1], ply_path)
            sample_count += 1

    print(f"\n{'='*60}")
    print(f"验证完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

