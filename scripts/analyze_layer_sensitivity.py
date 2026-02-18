"""
敏感层定位分析脚本

分别在 source 和 target 数据上计算每层的梯度敏感度，
找出 target 敏感但 source 不敏感的层（差异敏感度最大的层）。
这些层适合用于防御训练的选择性微调。
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from project_core import ConfigManager, PROJECT_ROOT
from models import ModelManager
from data import DataManager


def analyze_layer_sensitivity(
    model,
    data_loader,
    device='cuda',
    num_samples=10,
    label='data',
):
    """
    分析每层对 Gaussian 参数的敏感度

    Args:
        model: LGM 模型
        data_loader: 数据加载器
        device: 设备
        num_samples: 分析的样本数
        label: 数据集标签（用于日志）

    Returns:
        avg_gradients: {layer_name: {param_type: grad_norm}}
    """
    model.eval()
    model.requires_grad_(True)

    layer_gradients = defaultdict(lambda: defaultdict(list))

    print(f"\n[分析] 开始分析 {label} 数据（{num_samples} 个样本）...")

    sample_count = 0
    for batch in tqdm(data_loader, desc=f"分析 {label}"):
        if sample_count >= num_samples:
            break

        input_images = batch['input_images'].to(device)
        B = input_images.shape[0]

        gaussians = model.forward_gaussians(input_images)

        param_types = {
            'position': gaussians[..., 0:3],     # [B, N, 3]
            'opacity': gaussians[..., 3:4],      # [B, N, 1]
            'scale': gaussians[..., 4:7],        # [B, N, 3]
            'rotation': gaussians[..., 7:11],    # [B, N, 4]
            'rgb': gaussians[..., 11:14],        # [B, N, 3]
            'all': gaussians,                    # [B, N, 14] 综合
        }

        for param_name, param_tensor in param_types.items():
            model.zero_grad()
            loss = param_tensor.norm()
            loss.backward(retain_graph=True)

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    layer_gradients[name][param_name].append(grad_norm)

        sample_count += B

    avg_gradients = {}
    for layer_name, param_grads in layer_gradients.items():
        avg_gradients[layer_name] = {}
        for param_name, grad_list in param_grads.items():
            avg_gradients[layer_name][param_name] = float(np.mean(grad_list))

    return avg_gradients


def compute_differential_sensitivity(source_grads, target_grads):
    """
    计算差异敏感度：target 敏感但 source 不敏感的层

    对每层计算 differential = target_grad / (source_grad + eps)
    比值越大，说明该层对 target 特别敏感但对 source 不敏感。

    Args:
        source_grads: source 数据的梯度 {layer: {param_type: grad_norm}}
        target_grads: target 数据的梯度 {layer: {param_type: grad_norm}}

    Returns:
        differential: {layer: {param_type: ratio, ...}}
    """
    all_layers = set(list(source_grads.keys()) + list(target_grads.keys()))
    param_types = ['all', 'position', 'opacity', 'scale', 'rotation', 'rgb']
    eps = 1e-10

    differential = {}
    for layer in all_layers:
        differential[layer] = {}
        for pt in param_types:
            src_val = source_grads.get(layer, {}).get(pt, 0.0)
            tgt_val = target_grads.get(layer, {}).get(pt, 0.0)
            ratio = tgt_val / (src_val + eps)
            differential[layer][pt] = ratio
            differential[layer][f'{pt}_source'] = src_val
            differential[layer][f'{pt}_target'] = tgt_val

    return differential


def visualize_differential(differential, save_dir, top_k=30):
    """
    可视化差异敏感度
    """
    os.makedirs(save_dir, exist_ok=True)
    param_types = ['all', 'position', 'opacity', 'scale', 'rotation', 'rgb']
    layer_names = list(differential.keys())

    for pt in param_types:
        ratios = [(layer, differential[layer][pt]) for layer in layer_names]
        ratios.sort(key=lambda x: x[1], reverse=True)
        top = ratios[:top_k]

        layers = [x[0] for x in top]
        values = [x[1] for x in top]

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # 图1: 差异比值 (target / source)
        axes[0].barh(range(len(layers)), values, color='steelblue')
        axes[0].set_yticks(range(len(layers)))
        axes[0].set_yticklabels(layers, fontsize=7)
        axes[0].set_xlabel('Target / Source Ratio')
        axes[0].set_title(f'{pt.capitalize()} - Differential Sensitivity')
        axes[0].invert_yaxis()

        # 图2: Source vs Target 对比
        src_vals = [differential[l].get(f'{pt}_source', 0) for l in layers]
        tgt_vals = [differential[l].get(f'{pt}_target', 0) for l in layers]
        y = np.arange(len(layers))
        axes[1].barh(y - 0.2, src_vals, 0.4, label='Source', color='green', alpha=0.7)
        axes[1].barh(y + 0.2, tgt_vals, 0.4, label='Target', color='red', alpha=0.7)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(layers, fontsize=7)
        axes[1].set_xlabel('Gradient Norm')
        axes[1].set_title(f'{pt.capitalize()} - Source vs Target')
        axes[1].legend()
        axes[1].invert_yaxis()

        # 图3: 差值 (target - source)
        diff_vals = [t - s for s, t in zip(src_vals, tgt_vals)]
        colors = ['red' if v > 0 else 'green' for v in diff_vals]
        axes[2].barh(range(len(layers)), diff_vals, color=colors, alpha=0.7)
        axes[2].set_yticks(range(len(layers)))
        axes[2].set_yticklabels(layers, fontsize=7)
        axes[2].set_xlabel('Target - Source')
        axes[2].set_title(f'{pt.capitalize()} - Difference')
        axes[2].invert_yaxis()

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'differential_{pt}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[可视化] 已保存: {save_path}")


def find_defense_layers(differential, k=10):
    """
    找出最适合防御训练的层（target 敏感、source 不敏感）

    Args:
        differential: compute_differential_sensitivity 的输出
        k: Top-K

    Returns:
        defense_layers: {param_type: [(layer_name, ratio, source_grad, target_grad), ...]}
    """
    param_types = ['all', 'position', 'opacity', 'scale', 'rotation', 'rgb']
    defense_layers = {}

    for pt in param_types:
        layer_info = []
        for layer, vals in differential.items():
            ratio = vals.get(pt, 0)
            src = vals.get(f'{pt}_source', 0)
            tgt = vals.get(f'{pt}_target', 0)
            layer_info.append((layer, ratio, src, tgt))

        layer_info.sort(key=lambda x: x[1], reverse=True)
        defense_layers[pt] = layer_info[:k]

    return defense_layers


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--gpu', type=str, default='7')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Source 和 Target 各分析多少个样本')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='output/layer_sensitivity')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda'

    print(f"\n{'='*60}")
    print(f"敏感层定位分析（Source vs Target 差异分析）")
    print(f"{'='*60}")
    print(f"配置文件: {args.config}")
    print(f"GPU: {args.gpu}")
    print(f"每组样本数: {args.num_samples}")
    print(f"Top-K: {args.top_k}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}\n")

    config_mgr = ConfigManager(args.config)
    config = config_mgr.config

    # 加载模型（不应用 LoRA，使用原始预训练模型）
    print("[1/5] 加载预训练模型...")
    model_mgr = ModelManager(config)
    model_mgr.load_model(device=device, dtype=torch.float32)
    model = model_mgr.model

    # 加载 Source 数据
    print("\n[2/5] 加载 Source 数据...")
    source_data_mgr = DataManager(config, model_mgr.opt)
    source_data_mgr.setup_dataloaders(train=True, val=False, subset='source')

    # 加载 Target 数据
    print("\n[3/5] 加载 Target 数据...")
    target_data_mgr = DataManager(config, model_mgr.opt)
    target_data_mgr.setup_dataloaders(train=True, val=False, subset='target')

    # 分析敏感度
    print("\n[4/5] 分析敏感度...")
    source_grads = analyze_layer_sensitivity(
        model=model,
        data_loader=source_data_mgr.train_loader,
        device=device,
        num_samples=args.num_samples,
        label='Source',
    )

    target_grads = analyze_layer_sensitivity(
        model=model,
        data_loader=target_data_mgr.train_loader,
        device=device,
        num_samples=args.num_samples,
        label='Target',
    )

    # 计算差异敏感度
    differential = compute_differential_sensitivity(source_grads, target_grads)

    # 可视化
    print("\n[5/5] 生成可视化...")
    visualize_differential(differential, args.output_dir, top_k=30)

    # 找出防御层
    defense_layers = find_defense_layers(differential, k=args.top_k)

    # 保存结果
    results = {
        'defense_layers': {
            pt: [(layer, float(ratio), float(src), float(tgt))
                 for layer, ratio, src, tgt in layers]
            for pt, layers in defense_layers.items()
        },
        'source_gradients': source_grads,
        'target_gradients': target_grads,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, 'sensitivity_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[结果] 已保存: {result_path}")

    # 打印推荐的防御层
    print(f"\n{'='*60}")
    print(f"推荐防御层（Target 敏感 / Source 不敏感）")
    print(f"{'='*60}")
    for pt, layers in defense_layers.items():
        print(f"\n{pt.upper()}:")
        print(f"  {'Rank':>4s}  {'Layer':<50s}  {'Ratio':>8s}  {'Source':>10s}  {'Target':>10s}")
        print(f"  {'----':>4s}  {'-'*50:<50s}  {'-----':>8s}  {'------':>10s}  {'------':>10s}")
        for i, (layer, ratio, src, tgt) in enumerate(layers, 1):
            print(f"  {i:4d}  {layer:<50s}  {ratio:8.2f}  {src:10.6f}  {tgt:10.6f}")

    print(f"\n{'='*60}")
    print(f"分析完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
