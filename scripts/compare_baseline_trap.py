"""
基线对比分析脚本

对比原始模型 vs 陷阱模型的 Gaussian 参数分布
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from project_core import ConfigManager
from models import ModelManager
from data import DataManager


def analyze_gaussian_distribution(model, data_loader, device='cuda', num_samples=50):
    """
    分析 Gaussian 参数分布

    Returns:
        distributions: {
            'scale': [N个scale值],
            'scale_ratio': [N个各向异性比率],
            'position': [N个position值],
            'opacity': [N个opacity值],
        }
    """
    model.eval()

    distributions = {
        'scale_x': [],
        'scale_y': [],
        'scale_z': [],
        'scale_ratio': [],  # max/min
        'position_x': [],
        'position_y': [],
        'position_z': [],
        'opacity': [],
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
            opacity = gaussians[..., 3:4]   # [B, N, 1]
            scale = gaussians[..., 4:7]     # [B, N, 3]

            B, N, _ = position.shape

            for b in range(B):
                # Scale
                scale_b = scale[b].cpu().numpy()  # [N, 3]
                distributions['scale_x'].extend(scale_b[:, 0].tolist())
                distributions['scale_y'].extend(scale_b[:, 1].tolist())
                distributions['scale_z'].extend(scale_b[:, 2].tolist())

                # Scale ratio
                scale_sq = scale[b] ** 2
                max_scale = scale_sq.max(dim=-1)[0]
                min_scale = scale_sq.min(dim=-1)[0]
                ratio = (max_scale / (min_scale + 1e-6)).cpu().numpy()
                distributions['scale_ratio'].extend(ratio.tolist())

                # Position
                pos_b = position[b].cpu().numpy()  # [N, 3]
                distributions['position_x'].extend(pos_b[:, 0].tolist())
                distributions['position_y'].extend(pos_b[:, 1].tolist())
                distributions['position_z'].extend(pos_b[:, 2].tolist())

                # Opacity
                opacity_b = opacity[b].cpu().numpy()  # [N, 1]
                distributions['opacity'].extend(opacity_b[:, 0].tolist())

            sample_count += B

    return distributions


def plot_comparison(dist_before, dist_after, save_dir):
    """
    绘制对比图
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Scale 分布对比
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scale X/Y/Z
    ax = axes[0, 0]
    ax.hist(dist_before['scale_x'], bins=50, alpha=0.5, label='Before (X)', color='blue')
    ax.hist(dist_after['scale_x'], bins=50, alpha=0.5, label='After (X)', color='red')
    ax.set_xlabel('Scale X')
    ax.set_ylabel('Frequency')
    ax.set_title('Scale X Distribution')
    ax.legend()
    ax.set_yscale('log')

    ax = axes[0, 1]
    ax.hist(dist_before['scale_y'], bins=50, alpha=0.5, label='Before (Y)', color='blue')
    ax.hist(dist_after['scale_y'], bins=50, alpha=0.5, label='After (Y)', color='red')
    ax.set_xlabel('Scale Y')
    ax.set_ylabel('Frequency')
    ax.set_title('Scale Y Distribution')
    ax.legend()
    ax.set_yscale('log')

    ax = axes[1, 0]
    ax.hist(dist_before['scale_z'], bins=50, alpha=0.5, label='Before (Z)', color='blue')
    ax.hist(dist_after['scale_z'], bins=50, alpha=0.5, label='After (Z)', color='red')
    ax.set_xlabel('Scale Z')
    ax.set_ylabel('Frequency')
    ax.set_title('Scale Z Distribution')
    ax.legend()
    ax.set_yscale('log')

    # Scale Ratio (各向异性)
    ax = axes[1, 1]
    # 限制范围以便可视化
    ratio_before = np.clip(dist_before['scale_ratio'], 0, 1000)
    ratio_after = np.clip(dist_after['scale_ratio'], 0, 1000)
    ax.hist(ratio_before, bins=50, alpha=0.5, label='Before', color='blue')
    ax.hist(ratio_after, bins=50, alpha=0.5, label='After', color='red')
    ax.set_xlabel('Scale Anisotropy Ratio (max/min, clipped at 1000)')
    ax.set_ylabel('Frequency')
    ax.set_title('Scale Anisotropy Distribution')
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scale_comparison.png'), dpi=150)
    plt.close()
    print(f"[可视化] 已保存: {save_dir}/scale_comparison.png")

    # 2. Position 分布对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, axis_name in enumerate(['x', 'y', 'z']):
        ax = axes[i]
        key = f'position_{axis_name}'
        ax.hist(dist_before[key], bins=50, alpha=0.5, label='Before', color='blue')
        ax.hist(dist_after[key], bins=50, alpha=0.5, label='After', color='red')
        ax.set_xlabel(f'Position {axis_name.upper()}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Position {axis_name.upper()} Distribution')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'position_comparison.png'), dpi=150)
    plt.close()
    print(f"[可视化] 已保存: {save_dir}/position_comparison.png")

    # 3. Opacity 分布对比
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(dist_before['opacity'], bins=50, alpha=0.5, label='Before', color='blue')
    ax.hist(dist_after['opacity'], bins=50, alpha=0.5, label='After', color='red')
    ax.set_xlabel('Opacity')
    ax.set_ylabel('Frequency')
    ax.set_title('Opacity Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'opacity_comparison.png'), dpi=150)
    plt.close()
    print(f"[可视化] 已保存: {save_dir}/opacity_comparison.png")


def compute_statistics(dist):
    """
    计算统计信息
    """
    stats = {}

    # Scale 统计
    for axis in ['x', 'y', 'z']:
        key = f'scale_{axis}'
        stats[f'{key}_mean'] = np.mean(dist[key])
        stats[f'{key}_std'] = np.std(dist[key])
        stats[f'{key}_min'] = np.min(dist[key])
        stats[f'{key}_max'] = np.max(dist[key])

    # Scale ratio 统计
    stats['scale_ratio_mean'] = np.mean(dist['scale_ratio'])
    stats['scale_ratio_std'] = np.std(dist['scale_ratio'])
    stats['scale_ratio_median'] = np.median(dist['scale_ratio'])
    stats['scale_ratio_max'] = np.max(dist['scale_ratio'])

    # Position 统计
    for axis in ['x', 'y', 'z']:
        key = f'position_{axis}'
        stats[f'{key}_mean'] = np.mean(dist[key])
        stats[f'{key}_std'] = np.std(dist[key])

    # Opacity 统计
    stats['opacity_mean'] = np.mean(dist['opacity'])
    stats['opacity_std'] = np.std(dist['opacity'])

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--gpu', type=str, default='7')
    parser.add_argument('--trap_model', type=str, default='output/trap_minimal/model_trap.pth')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='output/comparison')
    args = parser.parse_args()

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda'

    print(f"\n{'='*60}")
    print(f"基线对比分析")
    print(f"{'='*60}")
    print(f"配置文件: {args.config}")
    print(f"GPU: {args.gpu}")
    print(f"陷阱模型: {args.trap_model}")
    print(f"样本数: {args.num_samples}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}\n")

    # 加载配置
    config_mgr = ConfigManager(args.config)
    config = config_mgr.config

    # 加载数据
    print("[1/5] 加载数据...")
    from core.options import config_defaults
    opt = config_defaults[config['model']['size']]
    data_mgr = DataManager(config, opt)
    data_mgr.setup_dataloaders(train=True, val=False)
    data_loader = data_mgr.train_loader

    # 分析原始模型
    print("\n[2/5] 分析原始模型...")
    model_mgr_before = ModelManager(config)
    model_mgr_before.load_model(device=device, dtype=torch.float32)
    model_before = model_mgr_before.model

    dist_before = analyze_gaussian_distribution(
        model_before, data_loader, device, num_samples=args.num_samples
    )
    print(f"  收集了 {len(dist_before['scale_x'])} 个 Gaussian")

    # 分析陷阱模型
    print("\n[3/5] 分析陷阱模型...")
    model_mgr_after = ModelManager(config)
    model_mgr_after.load_model(device=device, dtype=torch.float32)
    model_after = model_mgr_after.model

    # 加载陷阱模型权重
    trap_state_dict = torch.load(args.trap_model, map_location='cpu')
    model_after.load_state_dict(trap_state_dict, strict=False)
    model_after = model_after.to(device)

    dist_after = analyze_gaussian_distribution(
        model_after, data_loader, device, num_samples=args.num_samples
    )
    print(f"  收集了 {len(dist_after['scale_x'])} 个 Gaussian")

    # 计算统计信息
    print("\n[4/5] 计算统计信息...")
    stats_before = compute_statistics(dist_before)
    stats_after = compute_statistics(dist_after)

    comparison_stats = {
        'before': stats_before,
        'after': stats_after,
        'change': {k: stats_after[k] - stats_before[k] for k in stats_before.keys()}
    }

    # 保存统计信息
    os.makedirs(args.output_dir, exist_ok=True)
    stats_path = os.path.join(args.output_dir, 'statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(comparison_stats, f, indent=2)
    print(f"  已保存: {stats_path}")

    # 打印关键统计
    print("\n关键统计对比:")
    print(f"  Scale Ratio (各向异性):")
    print(f"    Before: mean={stats_before['scale_ratio_mean']:.2f}, median={stats_before['scale_ratio_median']:.2f}, max={stats_before['scale_ratio_max']:.2f}")
    print(f"    After:  mean={stats_after['scale_ratio_mean']:.2f}, median={stats_after['scale_ratio_median']:.2f}, max={stats_after['scale_ratio_max']:.2f}")
    print(f"  Scale Mean:")
    print(f"    Before: X={stats_before['scale_x_mean']:.4f}, Y={stats_before['scale_y_mean']:.4f}, Z={stats_before['scale_z_mean']:.4f}")
    print(f"    After:  X={stats_after['scale_x_mean']:.4f}, Y={stats_after['scale_y_mean']:.4f}, Z={stats_after['scale_z_mean']:.4f}")

    # 绘制对比图
    print("\n[5/5] 生成可视化...")
    plot_comparison(dist_before, dist_after, args.output_dir)

    print(f"\n{'='*60}")
    print(f"对比分析完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
