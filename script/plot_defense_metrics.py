"""
可视化防御指标变化趋势

读取 defense_metrics_log.json 并绘制图表
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_metrics(log_file, output_dir=None):
    """绘制防御指标变化趋势"""
    with open(log_file, 'r') as f:
        history = json.load(f)

    if output_dir is None:
        output_dir = Path(log_file).parent

    epochs = [entry['epoch'] for entry in history]

    # 提取指标
    position_static = [entry['defense_metrics']['position_static'] for entry in history]
    scale_static = [entry['defense_metrics']['scale_static'] for entry in history]
    opacity_static = [entry['defense_metrics']['opacity_static'] for entry in history]
    coupling_value = [entry['defense_metrics'].get('coupling_value', 0) for entry in history]
    grad_cosine_sim = [entry['defense_metrics'].get('grad_cosine_sim', 0) for entry in history]

    # Gaussian 统计
    position_std = [entry['defense_metrics']['gaussian_stats']['position_std'] for entry in history]
    scale_mean = [entry['defense_metrics']['gaussian_stats']['scale_mean'] for entry in history]
    opacity_mean = [entry['defense_metrics']['gaussian_stats']['opacity_mean'] for entry in history]

    # Source 质量
    source_psnr = [entry.get('source_metrics', {}).get('source_psnr', 0) for entry in history]
    source_lpips = [entry.get('source_metrics', {}).get('source_lpips', 0) for entry in history]

    # 创建图表
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Defense Metrics Evolution During Attack Training', fontsize=16)

    # 第一行：陷阱指标 + 耦合
    axes[0, 0].plot(epochs, position_static, 'o-', color='blue', linewidth=2)
    axes[0, 0].set_title('Position Static (越负越强)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    axes[0, 1].plot(epochs, scale_static, 'o-', color='green', linewidth=2)
    axes[0, 1].set_title('Scale Static (越负越强)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    axes[0, 2].plot(epochs, coupling_value, 'o-', color='red', linewidth=2)
    axes[0, 2].set_title('Coupling Value (越负越强)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 第二行：梯度冲突 + Gaussian 统计
    axes[1, 0].plot(epochs, grad_cosine_sim, 'o-', color='purple', linewidth=2)
    axes[1, 0].set_title('Gradient Cosine Similarity (应趋向-1或0)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Cosine Similarity')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='正交')
    axes[1, 0].axhline(y=-1, color='b', linestyle='--', alpha=0.5, label='反向')
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, position_std, 'o-', color='brown', linewidth=2)
    axes[1, 1].set_title('Position Std (分布离散度)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(epochs, scale_mean, 'o-', color='pink', linewidth=2)
    axes[1, 2].set_title('Scale Mean (平均尺度)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].grid(True, alpha=0.3)

    # 第三行：陷阱综合对比 + Source 质量
    axes[2, 0].plot(epochs, position_static, 'o-', label='Position', linewidth=2)
    axes[2, 0].plot(epochs, scale_static, 's-', label='Scale', linewidth=2)
    axes[2, 0].plot(epochs, coupling_value, '^-', label='Coupling', linewidth=2)
    axes[2, 0].set_title('Trap Metrics (综合对比)')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Value')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    axes[2, 1].plot(epochs, source_psnr, 'o-', color='cyan', linewidth=2)
    axes[2, 1].set_title('Source PSNR (越高越好)')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('PSNR (dB)')
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].plot(epochs, source_lpips, 'o-', color='magenta', linewidth=2)
    axes[2, 2].set_title('Source LPIPS (越低越好)')
    axes[2, 2].set_xlabel('Epoch')
    axes[2, 2].set_ylabel('LPIPS')
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    output_path = Path(output_dir) / 'defense_metrics_evolution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {output_path}")

    # 打印摘要
    print("\n" + "=" * 80)
    print("防御指标变化摘要")
    print("=" * 80)
    print(f"Epoch 0 → Epoch {epochs[-1]}:")
    print(f"  position_static: {position_static[0]:.4f} → {position_static[-1]:.4f} (Δ={position_static[-1]-position_static[0]:+.4f})")
    print(f"  scale_static: {scale_static[0]:.4f} → {scale_static[-1]:.4f} (Δ={scale_static[-1]-scale_static[0]:+.4f})")
    print(f"  coupling_value: {coupling_value[0]:.4f} → {coupling_value[-1]:.4f} (Δ={coupling_value[-1]-coupling_value[0]:+.4f})")
    print(f"  grad_cosine_sim: {grad_cosine_sim[0]:.4f} → {grad_cosine_sim[-1]:.4f} (Δ={grad_cosine_sim[-1]-grad_cosine_sim[0]:+.4f})")
    print(f"  source_psnr: {source_psnr[0]:.2f} → {source_psnr[-1]:.2f} (Δ={source_psnr[-1]-source_psnr[0]:+.2f})")
    print(f"  source_lpips: {source_lpips[0]:.4f} → {source_lpips[-1]:.4f} (Δ={source_lpips[-1]-source_lpips[0]:+.4f})")
    print("\n解读:")
    print("  - 陷阱指标趋向 0 = 陷阱被破坏（防御失败）")
    print("  - 陷阱指标保持负值 = 陷阱仍然有效（防御成功）")
    print("  - coupling_value 趋向 0 = 耦合被破坏")
    print("  - grad_cosine_sim 趋向 0 或 1 = 梯度冲突被破坏（应保持在 -1 附近）")
    print("  - Source PSNR 下降 = 攻击影响了正常能力")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='可视化防御指标变化')
    parser.add_argument('log_file', type=str, help='defense_metrics_log.json 路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（默认与日志文件同目录）')
    args = parser.parse_args()

    plot_metrics(args.log_file, args.output_dir)


if __name__ == '__main__':
    main()
