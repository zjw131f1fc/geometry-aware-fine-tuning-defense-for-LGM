"""
绘图工具 - Pipeline 结果可视化
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_pipeline_results(baseline_history, postdef_history, defense_history, save_path):
    """
    绘制 Pipeline 2×2 对比图。

    布局：
        (0,0) Target Masked LPIPS — 主要防御指标
        (0,1) Target Masked PSNR
        (1,0) Source PSNR — 能力保持
        (1,1) Defense Training — trap loss + distill MSE

    Args:
        baseline_history: baseline 攻击 per-step 指标列表
        postdef_history: post-defense 攻击 per-step 指标列表
        defense_history: 防御训练 per-epoch 指标列表
        save_path: 图片保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps_b = [m['step'] for m in baseline_history]
    steps_p = [m['step'] for m in postdef_history]
    epochs_d = list(range(1, len(defense_history) + 1))

    # (0,0) Target Masked LPIPS (主要指标)
    ax = axes[0, 0]
    ax.plot(steps_b, [m.get('masked_lpips', 0) for m in baseline_history],
            'b-o', label='Baseline Attack', markersize=3)
    ax.plot(steps_p, [m.get('masked_lpips', 0) for m in postdef_history],
            'r-s', label='Post-Defense Attack', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Masked LPIPS')
    ax.set_title('Target Masked LPIPS (↑ = defense effective)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Target Masked PSNR
    ax = axes[0, 1]
    ax.plot(steps_b, [m.get('masked_psnr', 0) for m in baseline_history],
            'b-o', label='Baseline Attack', markersize=3)
    ax.plot(steps_p, [m.get('masked_psnr', 0) for m in postdef_history],
            'r-s', label='Post-Defense Attack', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Masked PSNR (dB)')
    ax.set_title('Target Masked PSNR (↓ = defense effective)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Source PSNR
    ax = axes[1, 0]
    ax.plot(steps_b, [m.get('source_psnr', 0) for m in baseline_history],
            'b-o', label='Baseline Attack', markersize=3)
    ax.plot(steps_p, [m.get('source_psnr', 0) for m in postdef_history],
            'r-s', label='Post-Defense Attack', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Source PSNR (should stay similar)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Defense Training Metrics
    ax = axes[1, 1]
    trap_keys = [k for k in defense_history[0] if k.startswith('val_') and 'static' in k]
    for k in trap_keys:
        label = k.replace('val_', '')
        ax.plot(epochs_d, [m.get(k, 0) for m in defense_history],
                '-o', label=label, markersize=3)
    ax.set_xlabel('Defense Epoch')
    ax.set_ylabel('Trap Loss (log scale)')
    ax.set_title('Defense Training')
    ax2 = ax.twinx()
    distill_vals = [m.get('val_source_distill_mse', 0) for m in defense_history]
    ax2.plot(epochs_d, distill_vals, 'k--', label='distill_mse', alpha=0.7)
    ax2.set_ylabel('Distill MSE')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Pipeline] 对比图已保存: {save_path}")
