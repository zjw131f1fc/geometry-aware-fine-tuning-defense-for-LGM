#!/usr/bin/env python3
"""
可视化噪声warmup曲线
"""

import matplotlib.pyplot as plt
import numpy as np

def compute_noise_schedule(noise_scale_target, warmup_steps, total_steps):
    """计算噪声schedule"""
    steps = np.arange(total_steps)
    noise_values = np.zeros(total_steps)

    for i in range(total_steps):
        if warmup_steps > 0:
            warmup_progress = min(1.0, i / warmup_steps)
            noise_values[i] = noise_scale_target * warmup_progress
        else:
            noise_values[i] = noise_scale_target

    return steps, noise_values

def plot_warmup_comparison():
    """绘制不同warmup配置的对比图"""
    total_steps = 200

    # 不同的warmup配置
    configs = [
        {"noise_scale": 0.01, "warmup_steps": 0, "label": "无warmup (steps=0)"},
        {"noise_scale": 0.01, "warmup_steps": 50, "label": "快速warmup (steps=50)"},
        {"noise_scale": 0.01, "warmup_steps": 100, "label": "标准warmup (steps=100)"},
        {"noise_scale": 0.01, "warmup_steps": 150, "label": "慢速warmup (steps=150)"},
    ]

    plt.figure(figsize=(12, 6))

    for config in configs:
        steps, noise_values = compute_noise_schedule(
            config["noise_scale"],
            config["warmup_steps"],
            total_steps
        )
        plt.plot(steps, noise_values, label=config["label"], linewidth=2)

    plt.xlabel('优化器步数 (Optimizer Steps)', fontsize=12)
    plt.ylabel('噪声强度 (Noise Scale)', fontsize=12)
    plt.title('噪声Warmup策略对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, total_steps)
    plt.ylim(0, 0.012)

    # 添加参考线
    plt.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='目标值')

    plt.tight_layout()
    plt.savefig('noise_warmup_comparison.png', dpi=150)
    print("图表已保存到: noise_warmup_comparison.png")
    plt.close()

def plot_different_targets():
    """绘制不同目标噪声强度的warmup曲线"""
    total_steps = 200
    warmup_steps = 100

    targets = [0.005, 0.01, 0.015, 0.02]

    plt.figure(figsize=(12, 6))

    for target in targets:
        steps, noise_values = compute_noise_schedule(target, warmup_steps, total_steps)
        plt.plot(steps, noise_values, label=f'noise_scale={target}', linewidth=2)

    plt.xlabel('优化器步数 (Optimizer Steps)', fontsize=12)
    plt.ylabel('噪声强度 (Noise Scale)', fontsize=12)
    plt.title(f'不同目标噪声强度的Warmup曲线 (warmup_steps={warmup_steps})',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, total_steps)

    plt.tight_layout()
    plt.savefig('noise_warmup_targets.png', dpi=150)
    print("图表已保存到: noise_warmup_targets.png")
    plt.close()

if __name__ == "__main__":
    print("生成噪声warmup可视化图表...")
    plot_warmup_comparison()
    plot_different_targets()
    print("完成！")
