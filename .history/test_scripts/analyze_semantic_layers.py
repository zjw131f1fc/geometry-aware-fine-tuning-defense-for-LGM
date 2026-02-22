"""
分析语义敏感层 (Semantic Layer Sensitivity Analysis)

通过比较source和target数据的task loss梯度，找出对语义最敏感的层。
这些层是存储target concept语义信息的关键层，应该在防御时重点保护。

方法：
1. 对source数据计算task loss (MSE + LPIPS) 对每层参数的梯度范数
2. 对target数据计算task loss对每层参数的梯度范数
3. 计算差异度量：|grad_target| / (|grad_source| + eps)
4. 选择差异最大的Top-K层作为语义敏感层
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "LGM"))

from project_core import ConfigManager
from models import ModelManager
from data import DataManager


class SemanticLayerAnalyzer:
    """语义层敏感性分析器"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        print("Loading model...")
        self.model_mgr = ModelManager(config)
        self.model_mgr.setup()
        self.model = self.model_mgr.model
        self.opt = self.model_mgr.opt

        # 加载数据（双数据加载器模式）
        print("Loading data...")
        data_config = config.get('data', {})

        if 'source' not in data_config or 'target' not in data_config:
            raise ValueError(
                "需要同时配置 source 和 target 数据！\n"
                "请在配置文件的 data 中添加：\n"
                "  source: {categories: [...], max_samples_per_category: N}\n"
                "  target: {categories: [...], max_samples_per_category: N}"
            )

        # Source数据加载器
        print("  Loading source data...")
        source_data_mgr = DataManager(config, self.opt)
        source_data_mgr.setup_dataloaders(train=True, val=False, subset='source')
        self.source_loader = source_data_mgr.train_loader
        print(f"    ✓ Source数据: {len(self.source_loader.dataset)} 样本")

        # Target数据加载器
        print("  Loading target data...")
        target_data_mgr = DataManager(config, self.opt)
        target_data_mgr.setup_dataloaders(train=True, val=False, subset='target')
        self.target_loader = target_data_mgr.train_loader
        print(f"    ✓ Target数据: {len(self.target_loader.dataset)} 样本")

        # 获取所有可训练层
        self.layer_names = []
        self.layer_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.layer_names.append(name)
                self.layer_params[name] = param

        print(f"Found {len(self.layer_names)} trainable layers")

    def compute_task_loss(self, batch: dict) -> torch.Tensor:
        """
        计算task loss (渲染loss)

        Args:
            batch: 数据批次

        Returns:
            loss: 标量loss
        """
        # 前向传播
        out = self.model(batch, step_ratio=1.0)

        # 提取loss
        loss = out['loss_mse'] + out['loss_lpips']

        return loss

    def compute_layer_gradients(
        self,
        dataloader,
        num_samples: int = 10
    ) -> Dict[str, float]:
        """
        计算每层参数的梯度范数

        Args:
            dataloader: 数据加载器
            num_samples: 使用的样本数量

        Returns:
            layer_grads: {layer_name: grad_norm}
        """
        layer_grads = {name: 0.0 for name in self.layer_names}

        self.model.train()
        count = 0

        for batch in tqdm(dataloader, desc="Computing gradients", total=num_samples):
            if count >= num_samples:
                break

            # 移动到设备
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            # 清零梯度
            self.model.zero_grad()

            # 计算loss
            loss = self.compute_task_loss(batch)

            # 反向传播
            loss.backward()

            # 累积每层的梯度范数
            for name in self.layer_names:
                param = self.layer_params[name]
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    layer_grads[name] += grad_norm

            count += 1

        # 平均
        for name in self.layer_names:
            layer_grads[name] /= count

        return layer_grads

    def analyze_sensitivity(
        self,
        num_samples: int = 10
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        分析语义敏感性（两种方法）

        方法1（文档方法）：只用target数据的梯度范数
        方法2（对比方法）：target/source梯度比值

        Args:
            num_samples: 每个数据集使用的样本数量

        Returns:
            source_grads: source数据的梯度范数
            target_grads: target数据的梯度范数
            method1_scores: 方法1的敏感性分数 (target梯度)
            method2_scores: 方法2的敏感性分数 (target/source比值)
        """
        # 获取source和target数据加载器
        source_loader = self.source_loader
        target_loader = self.target_loader

        print("\n=== Analyzing Source Data ===")
        source_grads = self.compute_layer_gradients(source_loader, num_samples)

        print("\n=== Analyzing Target Data ===")
        target_grads = self.compute_layer_gradients(target_loader, num_samples)

        # 方法1：只用target梯度（文档方法）
        method1_scores = target_grads.copy()

        # 方法2：计算target/source比值（对比方法）
        method2_scores = {}
        eps = 1e-8
        for name in self.layer_names:
            source_grad = source_grads[name]
            target_grad = target_grads[name]
            method2_scores[name] = target_grad / (source_grad + eps)

        return source_grads, target_grads, method1_scores, method2_scores

    def get_top_k_layers(
        self,
        sensitivity_scores: Dict[str, float],
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        获取Top-K敏感层

        Args:
            sensitivity_scores: 敏感性分数
            k: 返回的层数

        Returns:
            top_k: [(layer_name, score), ...]
        """
        sorted_layers = sorted(
            sensitivity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_layers[:k]

    def visualize_results(
        self,
        source_grads: Dict[str, float],
        target_grads: Dict[str, float],
        method1_scores: Dict[str, float],
        method2_scores: Dict[str, float],
        save_dir: str
    ):
        """
        可视化分析结果（比对两种方法）

        Args:
            source_grads: source梯度
            target_grads: target梯度
            method1_scores: 方法1分数（target梯度）
            method2_scores: 方法2分数（target/source比值）
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        # 获取两种方法的Top-20层
        top_20_method1 = self.get_top_k_layers(method1_scores, k=20)
        top_20_method2 = self.get_top_k_layers(method2_scores, k=20)

        # 提取层名称
        names_method1 = [name for name, _ in top_20_method1]
        names_method2 = [name for name, _ in top_20_method2]

        # 简化层名称
        def simplify_name(name):
            parts = name.split('.')
            if len(parts) > 2:
                return '.'.join(parts[-2:])
            return name

        short_names_method1 = [simplify_name(n) for n in names_method1]
        short_names_method2 = [simplify_name(n) for n in names_method2]

        # 图1：两种方法的Top-20对比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 方法1：Target梯度
        scores1 = [score for _, score in top_20_method1]
        ax1.barh(range(len(short_names_method1)), scores1, color='steelblue')
        ax1.set_yticks(range(len(short_names_method1)))
        ax1.set_yticklabels(short_names_method1, fontsize=8)
        ax1.set_xlabel('Target Gradient Norm')
        ax1.set_title('Method 1: Target Gradient Only (Document Method)')
        ax1.invert_yaxis()

        # 方法2：Target/Source比值
        scores2 = [score for _, score in top_20_method2]
        ax2.barh(range(len(short_names_method2)), scores2, color='coral')
        ax2.set_yticks(range(len(short_names_method2)))
        ax2.set_yticklabels(short_names_method2, fontsize=8)
        ax2.set_xlabel('Target/Source Ratio')
        ax2.set_title('Method 2: Target/Source Ratio (Contrastive Method)')
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'method_comparison.png'), dpi=150)
        plt.close()

        # 图2：重叠分析
        set_method1 = set(names_method1)
        set_method2 = set(names_method2)
        overlap = set_method1 & set_method2
        only_method1 = set_method1 - set_method2
        only_method2 = set_method2 - set_method1

        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Method 1 Only', 'Overlap', 'Method 2 Only']
        counts = [len(only_method1), len(overlap), len(only_method2)]
        colors = ['steelblue', 'purple', 'coral']
        ax.bar(categories, counts, color=colors, alpha=0.7)
        ax.set_ylabel('Number of Layers')
        ax.set_title('Top-20 Layer Overlap Between Methods')
        for i, count in enumerate(counts):
            ax.text(i, count + 0.5, str(count), ha='center', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'method_overlap.png'), dpi=150)
        plt.close()

        # 图3：Source vs Target梯度散点图（标注两种方法选择的层）
        fig, ax = plt.subplots(figsize=(10, 8))

        # 所有层（灰色）
        all_source = [source_grads[name] for name in self.layer_names]
        all_target = [target_grads[name] for name in self.layer_names]
        ax.scatter(all_source, all_target, alpha=0.3, s=20, color='gray', label='All layers')

        # 方法1选择的层（蓝色）
        method1_source = [source_grads[name] for name in names_method1]
        method1_target = [target_grads[name] for name in names_method1]
        ax.scatter(method1_source, method1_target, alpha=0.7, s=50, color='steelblue',
                   marker='o', label='Method 1 Top-20')

        # 方法2选择的层（橙色）
        method2_source = [source_grads[name] for name in names_method2]
        method2_target = [target_grads[name] for name in names_method2]
        ax.scatter(method2_source, method2_target, alpha=0.7, s=50, color='coral',
                   marker='s', label='Method 2 Top-20')

        # 重叠的层（紫色）
        overlap_source = [source_grads[name] for name in overlap]
        overlap_target = [target_grads[name] for name in overlap]
        ax.scatter(overlap_source, overlap_target, alpha=0.9, s=80, color='purple',
                   marker='*', label='Overlap', zorder=10)

        # 对角线
        max_val = max(max(all_source), max(all_target))
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')

        ax.set_xlabel('Source Gradient Norm')
        ax.set_ylabel('Target Gradient Norm')
        ax.set_title('Source vs Target Gradient Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gradient_scatter.png'), dpi=150)
        plt.close()

        print(f"\nVisualization saved to {save_dir}")
        print(f"\n=== Method Comparison ===")
        print(f"Method 1 (Target only): {len(set_method1)} layers")
        print(f"Method 2 (Target/Source): {len(set_method2)} layers")
        print(f"Overlap: {len(overlap)} layers ({len(overlap)/20*100:.1f}%)")
        print(f"Only in Method 1: {len(only_method1)} layers")
        print(f"Only in Method 2: {len(only_method2)} layers")

    def save_results(
        self,
        source_grads: Dict[str, float],
        target_grads: Dict[str, float],
        method1_scores: Dict[str, float],
        method2_scores: Dict[str, float],
        save_dir: str,
        top_k: int = 20
    ):
        """
        保存分析结果（两种方法）

        Args:
            source_grads: source梯度
            target_grads: target梯度
            method1_scores: 方法1分数
            method2_scores: 方法2分数
            save_dir: 保存目录
            top_k: 保存Top-K层
        """
        os.makedirs(save_dir, exist_ok=True)

        # 获取两种方法的Top-K层
        top_layers_method1 = self.get_top_k_layers(method1_scores, k=top_k)
        top_layers_method2 = self.get_top_k_layers(method2_scores, k=top_k)

        # 计算重叠
        names_method1 = set([name for name, _ in top_layers_method1])
        names_method2 = set([name for name, _ in top_layers_method2])
        overlap = names_method1 & names_method2

        # 准备结果
        results = {
            'method1_target_gradient': {
                'description': 'Method 1: Target gradient only (Document method)',
                'top_k_layers': [
                    {
                        'rank': i + 1,
                        'layer_name': name,
                        'target_grad': float(score),
                        'source_grad': float(source_grads[name]),
                        'in_overlap': name in overlap
                    }
                    for i, (name, score) in enumerate(top_layers_method1)
                ]
            },
            'method2_target_source_ratio': {
                'description': 'Method 2: Target/Source ratio (Contrastive method)',
                'top_k_layers': [
                    {
                        'rank': i + 1,
                        'layer_name': name,
                        'ratio': float(score),
                        'target_grad': float(target_grads[name]),
                        'source_grad': float(source_grads[name]),
                        'in_overlap': name in overlap
                    }
                    for i, (name, score) in enumerate(top_layers_method2)
                ]
            },
            'comparison': {
                'overlap_count': len(overlap),
                'overlap_percentage': len(overlap) / top_k * 100,
                'overlap_layers': sorted(list(overlap)),
                'only_method1': sorted(list(names_method1 - names_method2)),
                'only_method2': sorted(list(names_method2 - names_method1))
            }
        }

        # 保存JSON
        json_path = os.path.join(save_dir, 'semantic_layer_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {json_path}")

        # 打印两种方法的Top-K层
        print(f"\n{'='*80}")
        print(f"Method 1: Top-{top_k} Layers by Target Gradient (Document Method)")
        print(f"{'='*80}")
        for i, (name, score) in enumerate(top_layers_method1):
            overlap_mark = '★' if name in overlap else ' '
            print(f"{overlap_mark} {i+1:2d}. {name:60s} | Target: {score:8.4f} | "
                  f"Source: {source_grads[name]:8.4f}")

        print(f"\n{'='*80}")
        print(f"Method 2: Top-{top_k} Layers by Target/Source Ratio (Contrastive Method)")
        print(f"{'='*80}")
        for i, (name, score) in enumerate(top_layers_method2):
            overlap_mark = '★' if name in overlap else ' '
            print(f"{overlap_mark} {i+1:2d}. {name:60s} | Ratio: {score:8.4f} | "
                  f"Target: {target_grads[name]:8.4f} | Source: {source_grads[name]:8.4f}")

        print(f"\n{'='*80}")
        print(f"Overlap Analysis")
        print(f"{'='*80}")
        print(f"Overlap: {len(overlap)}/{top_k} layers ({len(overlap)/top_k*100:.1f}%)")
        print(f"★ marks overlapping layers in the lists above")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze semantic layer sensitivity')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to defense config file')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples per dataset')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top layers to report')
    parser.add_argument('--save_dir', type=str, default='analysis_results/semantic_layers',
                        help='Directory to save results')

    args = parser.parse_args()

    # 加载配置
    config_mgr = ConfigManager(args.config)
    config = config_mgr.config

    # 创建分析器
    analyzer = SemanticLayerAnalyzer(config)

    # 分析敏感性
    print(f"\nAnalyzing semantic layer sensitivity with {args.num_samples} samples per dataset...")
    source_grads, target_grads, method1_scores, method2_scores = analyzer.analyze_sensitivity(
        num_samples=args.num_samples
    )

    # 保存结果
    analyzer.save_results(
        source_grads,
        target_grads,
        method1_scores,
        method2_scores,
        args.save_dir,
        top_k=args.top_k
    )

    # 可视化
    analyzer.visualize_results(
        source_grads,
        target_grads,
        method1_scores,
        method2_scores,
        args.save_dir
    )

    print("\n=== Analysis Complete ===")


if __name__ == '__main__':
    main()
