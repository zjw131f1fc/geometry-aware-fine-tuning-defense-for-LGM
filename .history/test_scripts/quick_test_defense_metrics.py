"""
快速测试防御指标计算功能

用于验证 compute_defense_metrics 函数是否正常工作
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import argparse

from project_core import ConfigManager
from models import ModelManager
from data import DataManager
from evaluation import Evaluator
from methods.trap_losses import PositionCollapseLoss, ScaleAnisotropyLoss, OpacityCollapseLoss
import numpy as np


def compute_defense_metrics(model, evaluator, val_loader, device, num_samples=3, trap_combo='position+scale'):
    """简化版的防御指标计算（用于测试）"""
    position_loss_fn = PositionCollapseLoss()
    scale_loss_fn = ScaleAnisotropyLoss()

    trap_names = trap_combo.split('+')
    trap_losses = {}
    if 'position' in trap_names:
        trap_losses['position'] = position_loss_fn
    if 'scale' in trap_names:
        trap_losses['scale'] = scale_loss_fn

    model.eval()

    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break

        input_images = batch['input_images'].to(device)

        # 计算 Gaussian
        with torch.no_grad():
            gaussians = evaluator.generate_gaussians(input_images)
            print(f"\n样本 {i+1}:")
            print(f"  Gaussian shape: {gaussians.shape}")

            position_loss = position_loss_fn(gaussians).item()
            scale_loss = scale_loss_fn(gaussians).item()
            print(f"  position_static: {position_loss:.4f}")
            print(f"  scale_static: {scale_loss:.4f}")

        # 计算梯度冲突
        model.zero_grad()
        gaussians_grad = model.forward_gaussians(input_images)

        static_loss_tensors = {}
        for name, loss_fn in trap_losses.items():
            static_loss_tensors[name] = loss_fn(gaussians_grad)

        # 计算耦合
        if len(static_loss_tensors) > 1:
            product = torch.ones(1, device=device)
            for loss in static_loss_tensors.values():
                product = product * (1 - loss)
            coupling_value = -(product - 1)
            print(f"  coupling_value: {coupling_value.item():.4f}")

            # 计算梯度余弦相似度
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if len(trainable_params) > 0:
                all_grads = {}
                for trap_name, loss in static_loss_tensors.items():
                    grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=trainable_params,
                        create_graph=False,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    grad_vec = []
                    for g in grads:
                        if g is not None:
                            grad_vec.append(g.reshape(-1))
                    if len(grad_vec) > 0:
                        all_grads[trap_name] = torch.cat(grad_vec)

                if len(all_grads) >= 2:
                    names = list(all_grads.keys())
                    g_i = all_grads[names[0]]
                    g_j = all_grads[names[1]]
                    cos_sim = torch.dot(g_i, g_j) / (g_i.norm() * g_j.norm() + 1e-8)
                    print(f"  grad_cosine_sim: {cos_sim.item():.4f}")

    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(device)

    print("=" * 80)
    print("快速测试防御指标计算")
    print("=" * 80)

    # 加载配置
    config = ConfigManager(args.config).config
    trap_combo = config['defense'].get('trap_combo', 'position+scale')
    print(f"Trap 组合: {trap_combo}")

    # 加载模型
    print("\n加载模型...")
    model_mgr = ModelManager(config)
    model_mgr.setup(device=device)
    model = model_mgr.model

    # 加载数据
    print("加载数据...")
    data_mgr = DataManager(config, model_mgr.opt)
    data_mgr.setup_dataloaders(train=False, val=True, subset='target')

    # 创建评估器
    evaluator = Evaluator(model)

    # 测试指标计算
    print("\n" + "=" * 80)
    print("测试防御指标计算")
    print("=" * 80)
    compute_defense_metrics(model, evaluator, data_mgr.val_loader, device, num_samples=3, trap_combo=trap_combo)

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
