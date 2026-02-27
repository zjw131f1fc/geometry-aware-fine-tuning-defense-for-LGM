"""
比较随机初始化 vs 预训练LGM在攻击训练后的指标差异

该脚本会：
1. 创建两个模型：随机初始化 + 预训练
2. 对两个模型进行相同的攻击训练
3. 收集和对比指标（LPIPS, PSNR, Source质量等）
4. 生成对比报告和可视化
"""

import os
import sys
import argparse
import json
import copy
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from project_core import ConfigManager
from data import DataManager
from tools import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='比较随机初始化 vs 预训练LGM')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--attack_epochs', type=int, default=None,
                       help='攻击训练epoch数（默认从config读取）')
    parser.add_argument('--output_dir', type=str, default='output/compare_random_vs_pretrained',
                       help='输出目录')
    parser.add_argument('--num_render', type=int, default=3,
                       help='渲染样本数')
    parser.add_argument('--eval_every_steps', type=int, default=10,
                       help='每隔多少步评估一次')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    return parser.parse_args()


def plot_comparison(random_history, pretrained_history, save_path):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Random Init vs Pretrained LGM - Attack Training Comparison', fontsize=14)

    # 提取数据
    random_steps = [h['step'] for h in random_history]
    random_loss = [h['loss'] for h in random_history]
    random_masked_lpips = [h.get('masked_lpips', 0) for h in random_history]
    random_masked_psnr = [h.get('masked_psnr', 0) for h in random_history]

    pretrained_steps = [h['step'] for h in pretrained_history]
    pretrained_loss = [h['loss'] for h in pretrained_history]
    pretrained_masked_lpips = [h.get('masked_lpips', 0) for h in pretrained_history]
    pretrained_masked_psnr = [h.get('masked_psnr', 0) for h in pretrained_history]

    # 1. Loss曲线
    axes[0, 0].plot(random_steps, random_loss, 'b-', label='Random Init', linewidth=2)
    axes[0, 0].plot(pretrained_steps, pretrained_loss, 'r-', label='Pretrained', linewidth=2)
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Masked LPIPS曲线
    axes[0, 1].plot(random_steps, random_masked_lpips, 'b-', label='Random Init', linewidth=2)
    axes[0, 1].plot(pretrained_steps, pretrained_masked_lpips, 'r-', label='Pretrained', linewidth=2)
    axes[0, 1].set_xlabel('Training Steps')
    axes[0, 1].set_ylabel('Masked LPIPS')
    axes[0, 1].set_title('Masked LPIPS (Lower = Better Quality)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Masked PSNR曲线
    axes[1, 0].plot(random_steps, random_masked_psnr, 'b-', label='Random Init', linewidth=2)
    axes[1, 0].plot(pretrained_steps, pretrained_masked_psnr, 'r-', label='Pretrained', linewidth=2)
    axes[1, 0].set_xlabel('Training Steps')
    axes[1, 0].set_ylabel('Masked PSNR (dB)')
    axes[1, 0].set_title('Masked PSNR (Higher = Better Quality)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 最终指标对比（柱状图）
    metrics_names = ['Final Loss', 'Final LPIPS', 'Final PSNR']
    random_final = [random_loss[-1], random_masked_lpips[-1], random_masked_psnr[-1]]
    pretrained_final = [pretrained_loss[-1], pretrained_masked_lpips[-1], pretrained_masked_psnr[-1]]

    x = np.arange(len(metrics_names))
    width = 0.35

    axes[1, 1].bar(x - width/2, random_final, width, label='Random Init', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, pretrained_final, width, label='Pretrained', color='red', alpha=0.7)
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Final Metrics Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics_names, rotation=15, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"对比图已保存: {save_path}")
    plt.close()


def generate_report(random_results, pretrained_results, save_path):
    """生成文本报告"""
    random_history, random_source, random_target = random_results
    pretrained_history, pretrained_source, pretrained_target = pretrained_results

    report = []
    report.append("=" * 80)
    report.append("随机初始化 vs 预训练LGM - 攻击训练对比报告")
    report.append("=" * 80)
    report.append("")

    # 训练过程对比
    report.append("## 训练过程")
    report.append("")
    report.append("### 随机初始化")
    report.append(f"  初始Loss: {random_history[0]['loss']:.4f}")
    report.append(f"  最终Loss: {random_history[-1]['loss']:.4f}")
    report.append(f"  Loss下降: {random_history[0]['loss'] - random_history[-1]['loss']:.4f}")
    report.append("")
    report.append("### 预训练模型")
    report.append(f"  初始Loss: {pretrained_history[0]['loss']:.4f}")
    report.append(f"  最终Loss: {pretrained_history[-1]['loss']:.4f}")
    report.append(f"  Loss下降: {pretrained_history[0]['loss'] - pretrained_history[-1]['loss']:.4f}")
    report.append("")

    # Source质量对比
    report.append("## Source质量（攻击前）")
    report.append("")
    report.append("### 随机初始化")
    report.append(f"  PSNR: {random_source.get('psnr', 0):.2f} dB")
    report.append(f"  LPIPS: {random_source.get('lpips', 0):.4f}")
    report.append("")
    report.append("### 预训练模型")
    report.append(f"  PSNR: {pretrained_source.get('psnr', 0):.2f} dB")
    report.append(f"  LPIPS: {pretrained_source.get('lpips', 0):.4f}")
    report.append("")

    # Target质量对比
    report.append("## Target质量（攻击后）")
    report.append("")
    report.append("### 随机初始化")
    report.append(f"  PSNR: {random_target.get('psnr', 0):.2f} dB")
    report.append(f"  LPIPS: {random_target.get('lpips', 0):.4f}")
    report.append("")
    report.append("### 预训练模型")
    report.append(f"  PSNR: {pretrained_target.get('psnr', 0):.2f} dB")
    report.append(f"  LPIPS: {pretrained_target.get('lpips', 0):.4f}")
    report.append("")

    # Gaussian诊断对比
    if 'gaussian_diag' in random_target and 'gaussian_diag' in pretrained_target:
        report.append("## Gaussian诊断")
        report.append("")
        report.append("### 随机初始化")
        diag = random_target['gaussian_diag']
        report.append(f"  诊断: {diag.get('diagnosis', 'N/A')}")
        report.append(f"  opacity_mean: {diag.get('opacity_mean', 0):.4f}")
        report.append(f"  pos_spread: {diag.get('pos_spread', 0):.4f}")
        report.append(f"  scale_mean: {diag.get('scale_mean', 0):.6f}")
        report.append(f"  render_white_ratio: {diag.get('render_white_ratio', 0):.4f}")
        report.append("")
        report.append("### 预训练模型")
        diag = pretrained_target['gaussian_diag']
        report.append(f"  诊断: {diag.get('diagnosis', 'N/A')}")
        report.append(f"  opacity_mean: {diag.get('opacity_mean', 0):.4f}")
        report.append(f"  pos_spread: {diag.get('pos_spread', 0):.4f}")
        report.append(f"  scale_mean: {diag.get('scale_mean', 0):.6f}")
        report.append(f"  render_white_ratio: {diag.get('render_white_ratio', 0):.4f}")
        report.append("")

    # 关键发现
    report.append("## 关键发现")
    report.append("")

    # 对比LPIPS差异
    random_lpips = random_target.get('lpips', 0)
    pretrained_lpips = pretrained_target.get('lpips', 0)
    lpips_diff = pretrained_lpips - random_lpips

    if lpips_diff > 0.01:
        report.append(f"✓ 预训练模型的攻击后LPIPS更高 (+{lpips_diff:.4f})，说明预训练权重提供了更好的初始化")
    elif lpips_diff < -0.01:
        report.append(f"✗ 随机初始化的攻击后LPIPS更高 ({lpips_diff:.4f})，这很不寻常")
    else:
        report.append(f"≈ 两者攻击后LPIPS相近 (差异: {lpips_diff:.4f})")

    report.append("")
    report.append("=" * 80)

    # 保存报告
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"报告已保存: {save_path}")

    # 同时打印到控制台
    print('\n'.join(report))


def main():
    args = parse_args()

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print("=" * 80)
    print("比较随机初始化 vs 预训练LGM")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"攻击Epochs: {args.attack_epochs or '从config读取'}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)

    # 加载配置
    config_mgr = ConfigManager(args.config)
    base_config = config_mgr.config
    set_seed(base_config['misc']['seed'])

    # 获取攻击epoch数
    attack_epochs = args.attack_epochs or base_config['training'].get('attack_epochs', 5)

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"compare_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 准备数据加载器（两个实验共享）
    print("\n准备数据加载器...")
    from models import ModelManager
    temp_model_mgr = ModelManager(base_config)
    temp_model_mgr.load_model(device='cuda')
    opt = temp_model_mgr.opt

    data_mgr = DataManager(base_config, opt)
    data_mgr.setup_dataloaders(train=True, val=True, subset='target')
    target_train_loader = data_mgr.train_loader

    source_data_mgr = DataManager(base_config, opt)
    source_data_mgr.setup_dataloaders(train=False, val=True, subset='source')
    source_val_loader = source_data_mgr.val_loader

    del temp_model_mgr
    import torch
    torch.cuda.empty_cache()

    print(f"Target训练数据: {len(target_train_loader.dataset)} 样本")
    print(f"Source验证数据: {len(source_val_loader.dataset)} 样本")

    # 导入run_attack函数
    from training.finetuner import run_attack

    # ========== 实验1: 随机初始化 ==========
    print("\n" + "=" * 80)
    print("实验1: 随机初始化LGM")
    print("=" * 80)

    random_config = copy.deepcopy(base_config)
    random_config['model']['resume'] = None  # 移除预训练权重

    random_save_dir = os.path.join(output_dir, 'random_init')
    os.makedirs(random_save_dir, exist_ok=True)

    random_history, random_source_metrics, random_target_metrics = run_attack(
        config=random_config,
        target_train_loader=target_train_loader,
        source_val_loader=source_val_loader,
        save_dir=random_save_dir,
        attack_epochs=attack_epochs,
        num_render=args.num_render,
        eval_every_steps=args.eval_every_steps,
        phase_name="Random Init Attack",
    )

    # 保存随机初始化结果
    with open(os.path.join(random_save_dir, 'results.json'), 'w') as f:
        json.dump({
            'history': random_history,
            'source_metrics': random_source_metrics,
            'target_metrics': random_target_metrics,
        }, f, indent=2)

    # ========== 实验2: 预训练模型 ==========
    print("\n" + "=" * 80)
    print("实验2: 预训练LGM")
    print("=" * 80)

    pretrained_config = copy.deepcopy(base_config)
    # 保持原有的resume路径

    pretrained_save_dir = os.path.join(output_dir, 'pretrained')
    os.makedirs(pretrained_save_dir, exist_ok=True)

    pretrained_history, pretrained_source_metrics, pretrained_target_metrics = run_attack(
        config=pretrained_config,
        target_train_loader=target_train_loader,
        source_val_loader=source_val_loader,
        save_dir=pretrained_save_dir,
        attack_epochs=attack_epochs,
        num_render=args.num_render,
        eval_every_steps=args.eval_every_steps,
        phase_name="Pretrained Attack",
    )

    # 保存预训练结果
    with open(os.path.join(pretrained_save_dir, 'results.json'), 'w') as f:
        json.dump({
            'history': pretrained_history,
            'source_metrics': pretrained_source_metrics,
            'target_metrics': pretrained_target_metrics,
        }, f, indent=2)

    # ========== 生成对比报告 ==========
    print("\n" + "=" * 80)
    print("生成对比报告")
    print("=" * 80)

    # 绘制对比图
    plot_comparison(
        random_history,
        pretrained_history,
        os.path.join(output_dir, 'comparison_plot.png')
    )

    # 生成文本报告
    generate_report(
        (random_history, random_source_metrics, random_target_metrics),
        (pretrained_history, pretrained_source_metrics, pretrained_target_metrics),
        os.path.join(output_dir, 'comparison_report.txt')
    )

    # 保存完整对比数据
    comparison_data = {
        'random_init': {
            'history': random_history,
            'source_metrics': random_source_metrics,
            'target_metrics': random_target_metrics,
        },
        'pretrained': {
            'history': pretrained_history,
            'source_metrics': pretrained_source_metrics,
            'target_metrics': pretrained_target_metrics,
        },
        'config': {
            'attack_epochs': attack_epochs,
            'eval_every_steps': args.eval_every_steps,
            'num_render': args.num_render,
        }
    }

    with open(os.path.join(output_dir, 'comparison_data.json'), 'w') as f:
        json.dump(comparison_data, f, indent=2)

    print("\n" + "=" * 80)
    print("全部完成！")
    print(f"结果保存在: {output_dir}")
    print("  - comparison_plot.png: 对比图")
    print("  - comparison_report.txt: 文本报告")
    print("  - comparison_data.json: 完整数据")
    print("  - random_init/: 随机初始化结果")
    print("  - pretrained/: 预训练模型结果")
    print("=" * 80)


if __name__ == '__main__':
    main()

