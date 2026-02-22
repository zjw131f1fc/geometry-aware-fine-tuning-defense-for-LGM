"""
敏感层定位分析脚本（改进版）

支持两种模式：
- attack: 用攻击实际 loss（渲染 MSE + LPIPS）
- trap: 用 trap loss（position/scale/opacity/rotation）

在 source 和 target 数据上计算每层梯度范数，找出差异敏感层。

用法：
    python scripts/analyze_layer_sensitivity.py --gpu 7 --mode trap
    python scripts/analyze_layer_sensitivity.py --gpu 7 --mode attack
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

from project_core import ConfigManager
from models import ModelManager
from data import DataManager
from training.finetuner import AutoFineTuner
from methods.trap_losses import (
    PositionCollapseLoss, ScaleAnisotropyLoss,
    OpacityCollapseLoss, RotationAnisotropyLoss,
)

TRAP_LOSSES = {
    'position': PositionCollapseLoss(),
    'scale': ScaleAnisotropyLoss(),
    'opacity': OpacityCollapseLoss(),
    'rotation': RotationAnisotropyLoss(),
}


# LoRA 安全层：down_blocks.0-2 没有 attention，LoRA (qkv/proj) 无法触及
LORA_SAFE_PREFIXES = [
    'unet.down_blocks.0.',
    'unet.down_blocks.1.',
    'unet.down_blocks.2.',
]


def is_lora_safe(layer_name: str) -> bool:
    """判断层是否 LoRA 安全"""
    return any(layer_name.startswith(p) for p in LORA_SAFE_PREFIXES)


def get_block_name(layer_name: str) -> str:
    """
    从完整参数名提取 block 级名称
    例: unet.down_blocks.0.nets.0.conv1.weight → unet.down_blocks.0.nets.0
    """
    parts = layer_name.split('.')
    for i, part in enumerate(parts):
        if part in ('weight', 'bias'):
            return '.'.join(parts[:i])
        if part in ('norm1', 'norm2', 'conv1', 'conv2', 'shortcut', 'downsample'):
            return '.'.join(parts[:i])
    return layer_name


def compute_attack_loss_gradients(model, finetuner, data_loader, device,
                                   num_samples, label):
    """
    用攻击的实际 loss（渲染 MSE + LPIPS）计算每层梯度范数

    流程：input → forward_gaussians → render → MSE + LPIPS → backward

    Args:
        model: LGM 模型
        finetuner: AutoFineTuner（用于 _prepare_data 和 _forward_and_loss）
        data_loader: 数据加载器
        device: 设备
        num_samples: 分析样本数
        label: 数据集标签

    Returns:
        gradients: {layer_name: mean_grad_norm}
    """
    model.eval()
    model.requires_grad_(True)

    grad_accum = defaultdict(list)

    sample_count = 0
    for batch in tqdm(data_loader, desc=f"分析 {label}"):
        if sample_count >= num_samples:
            break

        B = batch['input_images'].shape[0]

        with torch.enable_grad():
            data = finetuner._prepare_data(batch)
            loss, loss_dict = finetuner._forward_and_loss(data)

            model.zero_grad()
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_accum[name].append(param.grad.norm().item())

        sample_count += B

    return {layer: float(np.mean(norms)) for layer, norms in grad_accum.items()}


def compute_trap_loss_gradients(model, data_loader, device, num_samples, label,
                                 trap_names=None):
    """
    用 trap loss 计算每层梯度范数

    对每种 trap loss 分别计算，返回 {trap_name: {layer: grad_norm}}

    Args:
        trap_names: 要计算的 trap loss 名称列表（None=全部）
    """
    model.eval()
    model.requires_grad_(True)

    active_traps = {k: v for k, v in TRAP_LOSSES.items()
                    if trap_names is None or k in trap_names}

    grad_accum = {name: defaultdict(list) for name in active_traps}

    sample_count = 0
    for batch in tqdm(data_loader, desc=f"分析 {label}"):
        if sample_count >= num_samples:
            break

        input_images = batch['input_images'].to(device)
        B = input_images.shape[0]

        with torch.enable_grad():
            gaussians = model.forward_gaussians(input_images)

            for trap_name, trap_fn in active_traps.items():
                model.zero_grad()
                loss = trap_fn(gaussians)
                loss.backward(retain_graph=True)

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_accum[trap_name][name].append(param.grad.norm().item())

        sample_count += B

    return {
        trap: {layer: float(np.mean(norms)) for layer, norms in layers.items()}
        for trap, layers in grad_accum.items()
    }


def aggregate_to_blocks(layer_gradients):
    """
    将参数级梯度聚合到 block 级

    对同一 block 内的所有参数梯度取 L2 范数。
    """
    block_grads = defaultdict(list)
    for layer, norm in layer_gradients.items():
        block = get_block_name(layer)
        block_grads[block].append(norm ** 2)

    return {
        block: float(np.sqrt(sum(sq_norms)))
        for block, sq_norms in block_grads.items()
    }


def compute_differential(source_grads, target_grads):
    """
    计算差异敏感度：target_grad / (source_grad + eps)

    Returns:
        differential: [(block, ratio, target_grad, source_grad, is_lora_safe)]
    """
    all_blocks = set(source_grads.keys()) | set(target_grads.keys())
    eps = 1e-10

    results = []
    for block in all_blocks:
        # 排除 lpips_loss（冻结的 VGG 特征，不可训练）
        if 'lpips_loss' in block:
            continue
        tgt = target_grads.get(block, 0.0)
        src = source_grads.get(block, 0.0)
        ratio = tgt / (src + eps)
        safe = is_lora_safe(block)
        results.append((block, ratio, tgt, src, safe))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def visualize_differential(differential, save_dir, top_k=30):
    """可视化差异敏感度"""
    os.makedirs(save_dir, exist_ok=True)

    top = differential[:top_k]
    if not top:
        return

    blocks = [x[0] for x in top]
    ratios = [x[1] for x in top]
    tgt_vals = [x[2] for x in top]
    src_vals = [x[3] for x in top]
    safe_flags = [x[4] for x in top]

    fig, axes = plt.subplots(1, 2, figsize=(20, max(8, len(blocks) * 0.35)))

    # 图1: Ratio（蓝=LoRA安全，灰=不安全）
    colors = ['steelblue' if s else 'lightgray' for s in safe_flags]
    axes[0].barh(range(len(blocks)), ratios, color=colors)
    axes[0].set_yticks(range(len(blocks)))
    axes[0].set_yticklabels(blocks, fontsize=7)
    axes[0].set_xlabel('Target / Source Ratio')
    axes[0].set_title('Attack Loss Differential (blue=LoRA-safe)')
    axes[0].invert_yaxis()

    # 图2: Source vs Target
    y = np.arange(len(blocks))
    axes[1].barh(y - 0.2, src_vals, 0.4, label='Source', color='green', alpha=0.7)
    axes[1].barh(y + 0.2, tgt_vals, 0.4, label='Target', color='red', alpha=0.7)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(blocks, fontsize=7)
    axes[1].set_xlabel('Gradient Norm')
    axes[1].set_title('Attack Loss - Source vs Target')
    axes[1].legend()
    axes[1].invert_yaxis()

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'attack_loss_differential.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[可视化] {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='层敏感度分析')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--gpu', type=str, default='7')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='output/layer_sensitivity_v2')
    parser.add_argument('--mode', type=str, default='attack',
                        choices=['attack', 'trap'],
                        help='attack=渲染MSE+LPIPS, trap=trap loss')
    parser.add_argument('--traps', type=str, nargs='+',
                        default=['position', 'scale', 'opacity', 'rotation'],
                        help='trap 模式下要分析的 loss 类型')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda'

    print(f"\n{'='*70}")
    print(f"层敏感度分析 — 模式: {args.mode}")
    print(f"{'='*70}")
    print(f"配置: {args.config} | GPU: {args.gpu} | 样本数: {args.num_samples}")
    if args.mode == 'trap':
        print(f"Trap losses: {args.traps}")
    print(f"输出: {args.output_dir}")
    print(f"{'='*70}\n")

    config_mgr = ConfigManager(args.config)
    config = config_mgr.config

    # 加载模型
    print("[1] 加载预训练模型...")
    model_mgr = ModelManager(config)
    model_mgr.load_model(device=device, dtype=torch.float32)
    model = model_mgr.model

    # 加载数据
    print("\n[2] 加载 Source 数据...")
    source_data_mgr = DataManager(config, model_mgr.opt)
    source_data_mgr.setup_dataloaders(train=True, val=False, subset='source')

    print("\n[3] 加载 Target 数据...")
    target_data_mgr = DataManager(config, model_mgr.opt)
    target_data_mgr.setup_dataloaders(train=True, val=False, subset='target')

    # 计算梯度
    if args.mode == 'attack':
        training_config = config['training']
        finetuner = AutoFineTuner(
            model=model, device=device,
            lr=training_config['lr'],
            weight_decay=training_config['weight_decay'],
            gradient_clip=training_config['gradient_clip'],
            mixed_precision='no',
            lambda_lpips=training_config.get('lambda_lpips', 1.0),
            gradient_accumulation_steps=1,
        )
        print("\n[4] 计算攻击 Loss 梯度...")
        source_grads = compute_attack_loss_gradients(
            model, finetuner, source_data_mgr.train_loader,
            device, args.num_samples, 'Source')
        target_grads = compute_attack_loss_gradients(
            model, finetuner, target_data_mgr.train_loader,
            device, args.num_samples, 'Target')

        # 聚合 + 差异
        source_blocks = aggregate_to_blocks(source_grads)
        target_blocks = aggregate_to_blocks(target_grads)
        differential = compute_differential(source_blocks, target_blocks)

        # 打印
        print_differential("Attack Loss", differential, args.top_k)

        # 保存
        save_results(args.output_dir, differential, source_blocks, target_blocks)
        visualize_differential(differential, args.output_dir)

    else:  # trap mode
        print(f"\n[4] 计算 Trap Loss 梯度（{args.traps}）...")
        source_trap_grads = compute_trap_loss_gradients(
            model, source_data_mgr.train_loader, device,
            args.num_samples, 'Source', args.traps)
        target_trap_grads = compute_trap_loss_gradients(
            model, target_data_mgr.train_loader, device,
            args.num_samples, 'Target', args.traps)

        os.makedirs(args.output_dir, exist_ok=True)
        all_results = {}

        for trap_name in args.traps:
            src = aggregate_to_blocks(source_trap_grads.get(trap_name, {}))
            tgt = aggregate_to_blocks(target_trap_grads.get(trap_name, {}))
            diff = compute_differential(src, tgt)

            print_differential(f"Trap: {trap_name}", diff, args.top_k)
            all_results[trap_name] = {
                'differential': [
                    {'block': b, 'ratio': r, 'tgt': t, 'src': s, 'safe': sf}
                    for b, r, t, s, sf in diff
                ],
            }

        result_path = os.path.join(args.output_dir, 'sensitivity_results_trap.json')
        with open(result_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"\n[结果] {result_path}")

    print(f"\n{'='*70}")
    print(f"分析完成")
    print(f"{'='*70}\n")


def print_differential(title, differential, top_k):
    """打印差异排名表"""
    print(f"\n{'='*70}")
    print(f"{title} — 全部层排名（target/source ratio）")
    print(f"{'='*70}")
    print(f"\n  {'#':>3s}  {'Block':<45s}  {'Ratio':>8s}  "
          f"{'TgtGrad':>10s}  {'SrcGrad':>10s}  {'LoRA':>5s}")
    print(f"  {'---':>3s}  {'-'*45:<45s}  {'-----':>8s}  "
          f"{'-------':>10s}  {'-------':>10s}  {'----':>5s}")
    for i, (block, ratio, tgt, src, safe) in enumerate(differential[:top_k], 1):
        tag = 'safe' if safe else ''
        print(f"  {i:3d}  {block:<45s}  {ratio:8.2f}  "
              f"{tgt:10.6f}  {src:10.6f}  {tag:>5s}")


def save_results(output_dir, differential, source_blocks, target_blocks):
    """保存结果到 JSON"""
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'all_layers': [
            {'block': b, 'ratio': r, 'target_grad': t,
             'source_grad': s, 'lora_safe': sf}
            for b, r, t, s, sf in differential
        ],
        'raw_source_grads': source_blocks,
        'raw_target_grads': target_blocks,
    }
    result_path = os.path.join(output_dir, 'sensitivity_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[结果] {result_path}")


if __name__ == '__main__':
    main()
