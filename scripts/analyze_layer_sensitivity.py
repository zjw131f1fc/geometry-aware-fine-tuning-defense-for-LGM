"""
敏感层定位分析脚本

通过梯度分析找到对 target 概念最敏感的层
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
):
    """
    分析每层对 Gaussian 参数的敏感度

    Args:
        model: LGM 模型
        data_loader: 数据加载器
        device: 设备
        num_samples: 分析的样本数

    Returns:
        layer_gradients: {layer_name: {param_type: grad_norm}}
    """
    model.eval()
    model.requires_grad_(True)

    # 存储每层的梯度范数
    layer_gradients = defaultdict(lambda: defaultdict(list))

    print(f"\n[分析] 开始分析 {num_samples} 个样本...")

    sample_count = 0
    for batch in tqdm(data_loader, desc="分析样本"):
        if sample_count >= num_samples:
            break

        # 准备数据
        input_images = batch['input_images'].to(device)
        B = input_images.shape[0]

        # 前向传播生成 Gaussian 参数
        # gaussians: [B, N, 14] tensor
        # 14维: pos(0:3), opacity(3:4), scale(4:7), rotation(7:11), rgbs(11:14)
        gaussians = model.forward_gaussians(input_images)

        # 对每个 Gaussian 参数类型计算梯度
        param_types = {
            'position': gaussians[..., 0:3],    # [B, N, 3]
            'scale': gaussians[..., 4:7],       # [B, N, 3]
            'rotation': gaussians[..., 7:11],   # [B, N, 4]
        }

        for param_name, param_tensor in param_types.items():
            # 清零梯度
            model.zero_grad()

            # 计算参数的 L2 范数作为标量损失
            loss = param_tensor.norm()

            # 反向传播
            loss.backward(retain_graph=True)

            # 收集每层的梯度范数
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    layer_gradients[name][param_name].append(grad_norm)

        sample_count += B

    # 计算平均梯度范数
    avg_gradients = {}
    for layer_name, param_grads in layer_gradients.items():
        avg_gradients[layer_name] = {}
        for param_name, grad_list in param_grads.items():
            avg_gradients[layer_name][param_name] = np.mean(grad_list)

    return avg_gradients


def visualize_gradients(avg_gradients, save_dir):
    """
    可视化梯度分布

    Args:
        avg_gradients: {layer_name: {param_type: grad_norm}}
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 提取数据
    layer_names = list(avg_gradients.keys())
    param_types = ['position', 'scale', 'rotation']

    # 为每个参数类型创建热力图
    for param_type in param_types:
        grad_values = [avg_gradients[layer].get(param_type, 0) for layer in layer_names]

        # 排序（从大到小）
        sorted_indices = np.argsort(grad_values)[::-1]
        sorted_layers = [layer_names[i] for i in sorted_indices]
        sorted_values = [grad_values[i] for i in sorted_indices]

        # 只显示 Top-30 层
        top_k = min(30, len(sorted_layers))
        sorted_layers = sorted_layers[:top_k]
        sorted_values = sorted_values[:top_k]

        # 绘制条形图
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_layers)), sorted_values)
        plt.yticks(range(len(sorted_layers)), sorted_layers, fontsize=8)
        plt.xlabel('Gradient Norm', fontsize=12)
        plt.title(f'Layer Sensitivity to {param_type.capitalize()}', fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'sensitivity_{param_type}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[可视化] 已保存: {save_path}")


def find_top_k_layers(avg_gradients, k=10):
    """
    找出 Top-K 敏感层

    Args:
        avg_gradients: {layer_name: {param_type: grad_norm}}
        k: Top-K

    Returns:
        top_k_layers: {param_type: [(layer_name, grad_norm), ...]}
    """
    param_types = ['position', 'scale', 'rotation']
    top_k_layers = {}

    for param_type in param_types:
        # 收集所有层的梯度
        layer_grads = [(layer, grads.get(param_type, 0))
                      for layer, grads in avg_gradients.items()]

        # 排序并取 Top-K
        layer_grads.sort(key=lambda x: x[1], reverse=True)
        top_k_layers[param_type] = layer_grads[:k]

    return top_k_layers


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/attack_config.yaml')
    parser.add_argument('--gpu', type=str, default='7')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='output/layer_sensitivity')
    args = parser.parse_args()

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda'

    print(f"\n{'='*60}")
    print(f"敏感层定位分析")
    print(f"{'='*60}")
    print(f"配置文件: {args.config}")
    print(f"GPU: {args.gpu}")
    print(f"样本数: {args.num_samples}")
    print(f"Top-K: {args.top_k}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}\n")

    # 加载配置
    config_mgr = ConfigManager(args.config)
    config = config_mgr.config

    # 加载模型（不应用 LoRA，使用原始预训练模型）
    print("[1/4] 加载预训练模型...")
    model_mgr = ModelManager(config)
    model_mgr.load_model(device=device, dtype=torch.float32)
    model = model_mgr.model

    # 加载数据
    print("\n[2/4] 加载数据...")
    data_mgr = DataManager(config, model_mgr.opt)
    data_mgr.setup_dataloaders(train=True, val=False)
    data_loader = data_mgr.train_loader

    # 分析敏感层
    print("\n[3/4] 分析敏感层...")
    avg_gradients = analyze_layer_sensitivity(
        model=model,
        data_loader=data_loader,
        device=device,
        num_samples=args.num_samples,
    )

    # 可视化
    print("\n[4/4] 生成可视化...")
    visualize_gradients(avg_gradients, args.output_dir)

    # 找出 Top-K 层
    top_k_layers = find_top_k_layers(avg_gradients, k=args.top_k)

    # 保存结果
    results = {
        'top_k_layers': {
            param_type: [(layer, float(grad)) for layer, grad in layers]
            for param_type, layers in top_k_layers.items()
        },
        'all_gradients': {
            layer: {param: float(grad) for param, grad in grads.items()}
            for layer, grads in avg_gradients.items()
        }
    }

    result_path = os.path.join(args.output_dir, 'sensitivity_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[结果] 已保存: {result_path}")

    # 打印 Top-K 层
    print(f"\n{'='*60}")
    print(f"Top-{args.top_k} 敏感层")
    print(f"{'='*60}")
    for param_type, layers in top_k_layers.items():
        print(f"\n{param_type.upper()}:")
        for i, (layer, grad) in enumerate(layers, 1):
            print(f"  {i:2d}. {layer:50s} {grad:.6f}")

    print(f"\n{'='*60}")
    print(f"分析完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

