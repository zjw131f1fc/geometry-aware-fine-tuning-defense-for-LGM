#!/usr/bin/env python3
"""
可视化Baseline Attack vs Post-Defense Attack的高斯分布对比

从metrics.json读取gaussian_diag统计数据，绘制分布对比图和trap示意图。

用法:
    python experiments/visualize_attack_defense_distribution.py \
        --metrics output/experiments_output/main_experiment_xxx/shoe_geotrap/metrics.json \
        --output visualization_output.png
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_metrics(metrics_path):
    """加载metrics.json文件"""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    baseline = metrics.get('baseline_target', {}).get('gaussian_diag', {})
    postdefense = metrics.get('postdefense_target', {}).get('gaussian_diag', {})

    if not baseline or not postdefense:
        raise ValueError(f"metrics.json缺少gaussian_diag数据: {metrics_path}")

    return baseline, postdefense


def create_gaussian_curve(mean, std, x_range):
    """生成高斯分布曲线"""
    x = np.linspace(x_range[0], x_range[1], 500)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    return x, y


def plot_distribution_comparison(baseline, postdefense, output_path):
    """
    绘制Baseline Attack vs Post-Defense Attack的分布对比图

    包含4个子图：
    1. Opacity分布对比
    2. Scale分布对比
    3. Position Spread分布对比
    4. Trap Loss值对比（柱状图）
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline Attack vs Post-Defense Attack: Gaussian Distribution Comparison',
                 fontsize=16, fontweight='bold')

    # 配色
    color_baseline = '#FF6B6B'  # 红色 - baseline attack
    color_postdef = '#4ECDC4'   # 青色 - post-defense attack
    color_trap = '#FFE66D'      # 黄色 - trap区域

    # === 子图1: Opacity分布 ===
    ax1 = axes[0, 0]
    opacity_base = baseline['opacity_mean']
    opacity_post = postdefense['opacity_mean']

    # 估计标准差（基于opacity_lt_01比例）
    std_base = 0.15
    std_post = 0.15

    x_opacity, y_base = create_gaussian_curve(opacity_base, std_base, (0, 1))
    _, y_post = create_gaussian_curve(opacity_post, std_post, (0, 1))

    ax1.plot(x_opacity, y_base, color=color_baseline, linewidth=2.5,
             label=f'Baseline Attack (μ={opacity_base:.3f})')
    ax1.plot(x_opacity, y_post, color=color_postdef, linewidth=2.5,
             label=f'Post-Defense Attack (μ={opacity_post:.3f})')

    # Trap区域示意（假设trap惩罚低opacity）
    trap_threshold = 0.1
    ax1.axvspan(0, trap_threshold, alpha=0.2, color=color_trap, label='Trap Zone')
    ax1.axvline(trap_threshold, color=color_trap, linestyle='--', linewidth=1.5)

    ax1.set_xlabel('Opacity Value', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.set_title('Opacity Distribution', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # === 子图2: Scale分布 ===
    ax2 = axes[0, 1]
    scale_base = baseline['scale_mean']
    scale_post = postdefense['scale_mean']

    # Scale的标准差（相对值）
    std_scale_base = scale_base * 0.5
    std_scale_post = scale_post * 0.5

    x_max = max(scale_base, scale_post) * 3
    x_scale, y_base = create_gaussian_curve(scale_base, std_scale_base, (0, x_max))
    _, y_post = create_gaussian_curve(scale_post, std_scale_post, (0, x_max))

    ax2.plot(x_scale, y_base, color=color_baseline, linewidth=2.5,
             label=f'Baseline Attack (μ={scale_base:.5f})')
    ax2.plot(x_scale, y_post, color=color_postdef, linewidth=2.5,
             label=f'Post-Defense Attack (μ={scale_post:.5f})')

    # Trap区域示意（假设trap惩罚极小scale）
    trap_scale_threshold = 0.001
    ax2.axvspan(0, trap_scale_threshold, alpha=0.2, color=color_trap, label='Trap Zone')
    ax2.axvline(trap_scale_threshold, color=color_trap, linestyle='--', linewidth=1.5)

    ax2.set_xlabel('Scale Value', fontsize=11)
    ax2.set_ylabel('Probability Density', fontsize=11)
    ax2.set_title('Scale Distribution', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # === 子图3: Position Spread分布 ===
    ax3 = axes[1, 0]
    pos_base = baseline['pos_spread']
    pos_post = postdefense['pos_spread']

    std_pos = 0.05

    x_pos, y_base = create_gaussian_curve(pos_base, std_pos, (0, 0.5))
    _, y_post = create_gaussian_curve(pos_post, std_pos, (0, 0.5))

    ax3.plot(x_pos, y_base, color=color_baseline, linewidth=2.5,
             label=f'Baseline Attack (μ={pos_base:.3f})')
    ax3.plot(x_pos, y_post, color=color_postdef, linewidth=2.5,
             label=f'Post-Defense Attack (μ={pos_post:.3f})')

    # Trap区域示意（假设trap惩罚position collapse）
    trap_pos_threshold = 0.05
    ax3.axvspan(0, trap_pos_threshold, alpha=0.2, color=color_trap, label='Trap Zone')
    ax3.axvline(trap_pos_threshold, color=color_trap, linestyle='--', linewidth=1.5)

    ax3.set_xlabel('Position Spread', fontsize=11)
    ax3.set_ylabel('Probability Density', fontsize=11)
    ax3.set_title('Position Spread Distribution', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # === 子图4: Trap Loss值对比（柱状图）===
    ax4 = axes[1, 1]

    trap_types = ['Position', 'Scale', 'Opacity', 'Rotation']
    baseline_traps = [
        baseline['trap_position'],
        baseline['trap_scale'],
        baseline['trap_opacity'],
        baseline['trap_rotation']
    ]
    postdef_traps = [
        postdefense['trap_position'],
        postdefense['trap_scale'],
        postdefense['trap_opacity'],
        postdefense['trap_rotation']
    ]

    x = np.arange(len(trap_types))
    width = 0.35

    bars1 = ax4.bar(x - width/2, baseline_traps, width, label='Baseline Attack',
                    color=color_baseline, alpha=0.8)
    bars2 = ax4.bar(x + width/2, postdef_traps, width, label='Post-Defense Attack',
                    color=color_postdef, alpha=0.8)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom' if height < 0 else 'top',
                    fontsize=8)

    ax4.set_xlabel('Trap Type', fontsize=11)
    ax4.set_ylabel('Trap Loss Value (negative = stronger penalty)', fontsize=11)
    ax4.set_title('Trap Loss Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(trap_types)
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # 添加说明文本
    info_text = (
        f"Gaussian Distance to Baseline: {postdefense.get('gaussian_dist_to_baseline', 'N/A'):.6f}\n"
        f"Baseline Diagnosis: {baseline.get('diagnosis', 'N/A')}\n"
        f"Post-Defense Diagnosis: {postdefense.get('diagnosis', 'N/A')}"
    )
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化Baseline Attack vs Post-Defense Attack的高斯分布对比'
    )
    parser.add_argument('--metrics', type=str, required=True,
                       help='metrics.json文件路径')
    parser.add_argument('--output', type=str, default='attack_defense_distribution.png',
                       help='输出图片路径')

    args = parser.parse_args()

    # 检查文件是否存在
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"找不到metrics.json: {metrics_path}")

    # 加载数据
    print(f"正在加载metrics: {metrics_path}")
    baseline, postdefense = load_metrics(metrics_path)

    # 打印关键统计信息
    print("\n=== Baseline Attack ===")
    print(f"  Opacity Mean: {baseline['opacity_mean']:.4f}")
    print(f"  Scale Mean: {baseline['scale_mean']:.6f}")
    print(f"  Position Spread: {baseline['pos_spread']:.4f}")
    print(f"  Trap Losses: pos={baseline['trap_position']:.2f}, "
          f"scale={baseline['trap_scale']:.2f}, "
          f"opacity={baseline['trap_opacity']:.2f}, "
          f"rotation={baseline['trap_rotation']:.2f}")

    print("\n=== Post-Defense Attack ===")
    print(f"  Opacity Mean: {postdefense['opacity_mean']:.4f}")
    print(f"  Scale Mean: {postdefense['scale_mean']:.6f}")
    print(f"  Position Spread: {postdefense['pos_spread']:.4f}")
    print(f"  Trap Losses: pos={postdefense['trap_position']:.2f}, "
          f"scale={postdefense['trap_scale']:.2f}, "
          f"opacity={postdefense['trap_opacity']:.2f}, "
          f"rotation={postdefense['trap_rotation']:.2f}")
    print(f"  Gaussian Distance to Baseline: {postdefense.get('gaussian_dist_to_baseline', 'N/A')}")

    # 绘制对比图
    print(f"\n正在生成可视化...")
    plot_distribution_comparison(baseline, postdefense, args.output)


if __name__ == '__main__':
    main()

