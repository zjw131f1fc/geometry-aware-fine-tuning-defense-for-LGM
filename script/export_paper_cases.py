#!/usr/bin/env python3
"""
导出论文案例图：原图 + Post-Defense Attack 渲染图成对保存

用法:
  python script/export_paper_cases.py \
    output/experiments_output/single_20260305_123456 \
    --num_samples 5 \
    --output_dir paper_cases
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# 设置环境变量（在 import torch 之前）
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('XFORMERS_DISABLED', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_tmp_base = os.environ.get('TMPDIR')
if not _tmp_base:
    _tmp_base = '/root/autodl-tmp/tmp' if os.path.isdir('/root/autodl-tmp') else '/tmp'
os.environ.setdefault('MPLCONFIGDIR', os.path.join(_tmp_base, 'mpl'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

_dgr_path = os.path.join(_repo_root, 'lib', 'diff-gaussian-rasterization')
if os.path.isdir(_dgr_path):
    sys.path.insert(0, _dgr_path)
sys.path.insert(0, _repo_root)

import json
import torch
import numpy as np
from PIL import Image

from project_core import ConfigManager
from models import ModelManager
from data import DataManager
from evaluation import Evaluator
from tools.utils import prepare_lgm_data


def load_pipeline_config(workspace: str):
    """加载 pipeline 配置"""
    config_path = os.path.join(workspace, 'pipeline_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_defense_model(workspace: str, config: dict, device: str = 'cuda'):
    """加载防御训练后的模型（使用 registry 统一 hash 机制）"""
    from training import load_or_train_defense

    # 使用 registry 模式，通过统一的 hash 机制加载防御模型
    defense_tag, _, defense_state_dict = load_or_train_defense(
        config,
        device=device,
        save_dir=os.path.join(workspace, 'phase2_defense'),
        cache_mode='registry',  # 使用统一的 hash 缓存机制
        return_state_dict=True,
    )

    # 加载基础模型
    model_manager = ModelManager(config)
    model_manager.load_model(device=device)

    # 获取实际的模型对象
    model = model_manager.model

    # 应用防御权重（strict=False 允许 LPIPS loss 等辅助模块的键缺失）
    model.load_state_dict(defense_state_dict, strict=False)
    model.eval()

    print(f"[Export] 已加载防御模型: {defense_tag}")
    return model, model_manager.opt


def export_paired_images(
    model,
    data_loader,
    output_dir: str,
    num_samples: int = 5,
    device: str = 'cuda',
):
    """
    导出原图和渲染图的成对图像

    Args:
        model: 训练后的模型
        data_loader: 数据加载器
        output_dir: 输出目录
        num_samples: 导出样本数
        device: 设备
    """
    os.makedirs(output_dir, exist_ok=True)

    evaluator = Evaluator(model, device=device)

    sample_count = 0
    for batch_idx, batch in enumerate(data_loader):
        if sample_count >= num_samples:
            break

        input_images = batch['input_images'].to(device)
        B = input_images.shape[0]

        # 获取 GT 图像和相机参数
        input_transforms = batch.get('input_transforms')
        supervision_transforms = batch.get('supervision_transforms')

        if input_transforms is not None and supervision_transforms is not None:
            all_transforms = torch.cat([
                input_transforms.to(device),
                supervision_transforms.to(device),
            ], dim=1)

            # 反 ImageNet 归一化得到输入视图 GT
            input_rgb = input_images[:, :, :3, :, :]
            IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
            IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
            input_rgb = input_rgb * IMAGENET_STD + IMAGENET_MEAN

            supervision_images = batch.get('supervision_images')
            if supervision_images is not None:
                supervision_images = supervision_images.to(device)
                all_gt_images = torch.cat([input_rgb, supervision_images], dim=1)
            else:
                all_gt_images = input_rgb

            # 生成 Gaussians 并渲染
            gaussians = evaluator.generate_gaussians(input_images)
            rendered = evaluator.render_views(gaussians, transforms=all_transforms)  # [B, V, 3, H, W]

            # 保存每个样本的成对图像
            for b in range(B):
                if sample_count >= num_samples:
                    break

                gt = all_gt_images[b]  # [V, 3, H, W]
                pred = rendered[b]  # [V, 3, H, W]

                # 保存每个视角的成对图像
                for v in range(gt.shape[0]):
                    gt_img = gt[v].clamp(0, 1).cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
                    pred_img = pred[v].clamp(0, 1).cpu().permute(1, 2, 0).numpy()  # [H, W, 3]

                    # 转换为 uint8
                    gt_uint8 = (gt_img * 255).astype(np.uint8)
                    pred_uint8 = (pred_img * 255).astype(np.uint8)

                    # 如果尺寸不匹配，将 GT 调整到渲染图像的尺寸
                    if gt_uint8.shape[:2] != pred_uint8.shape[:2]:
                        H_pred, W_pred = pred_uint8.shape[:2]
                        gt_pil = Image.fromarray(gt_uint8)
                        gt_pil = gt_pil.resize((W_pred, H_pred), Image.BILINEAR)
                        gt_uint8 = np.array(gt_pil)

                    # 保存原图
                    gt_path = os.path.join(output_dir, f"sample_{sample_count:03d}_view_{v:02d}_gt.png")
                    Image.fromarray(gt_uint8).save(gt_path)

                    # 保存渲染图
                    pred_path = os.path.join(output_dir, f"sample_{sample_count:03d}_view_{v:02d}_rendered.png")
                    Image.fromarray(pred_uint8).save(pred_path)

                    # 保存成对对比图（左右拼接）
                    paired = np.concatenate([gt_uint8, pred_uint8], axis=1)  # [H, W*2, 3]
                    paired_path = os.path.join(output_dir, f"sample_{sample_count:03d}_view_{v:02d}_paired.png")
                    Image.fromarray(paired).save(paired_path)

                sample_count += 1
                print(f"[Export] 已导出样本 {sample_count}/{num_samples}")

        else:
            # 如果没有 transforms，使用默认视角
            gt_images = batch.get('supervision_images')
            if gt_images is not None:
                gt_images = gt_images.to(device)

            gaussians = evaluator.generate_gaussians(input_images)
            rendered = evaluator.render_canonical_views(gaussians)  # [B, V, 3, H, W]

            for b in range(B):
                if sample_count >= num_samples:
                    break

                pred = rendered[b]  # [V, 3, H, W]

                if gt_images is not None:
                    gt = gt_images[b]  # [V, 3, H, W]

                    for v in range(min(gt.shape[0], pred.shape[0])):
                        gt_img = gt[v].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                        pred_img = pred[v].clamp(0, 1).cpu().permute(1, 2, 0).numpy()

                        gt_uint8 = (gt_img * 255).astype(np.uint8)
                        pred_uint8 = (pred_img * 255).astype(np.uint8)

                        # 如果尺寸不匹配，将 GT 调整到渲染图像的尺寸
                        if gt_uint8.shape[:2] != pred_uint8.shape[:2]:
                            H_pred, W_pred = pred_uint8.shape[:2]
                            gt_pil = Image.fromarray(gt_uint8)
                            gt_pil = gt_pil.resize((W_pred, H_pred), Image.BILINEAR)
                            gt_uint8 = np.array(gt_pil)

                        gt_path = os.path.join(output_dir, f"sample_{sample_count:03d}_view_{v:02d}_gt.png")
                        Image.fromarray(gt_uint8).save(gt_path)

                        pred_path = os.path.join(output_dir, f"sample_{sample_count:03d}_view_{v:02d}_rendered.png")
                        Image.fromarray(pred_uint8).save(pred_path)

                        paired = np.concatenate([gt_uint8, pred_uint8], axis=1)
                        paired_path = os.path.join(output_dir, f"sample_{sample_count:03d}_view_{v:02d}_paired.png")
                        Image.fromarray(paired).save(paired_path)
                else:
                    # 只有渲染图，没有 GT
                    for v in range(pred.shape[0]):
                        pred_img = pred[v].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                        pred_uint8 = (pred_img * 255).astype(np.uint8)

                        pred_path = os.path.join(output_dir, f"sample_{sample_count:03d}_view_{v:02d}_rendered.png")
                        Image.fromarray(pred_uint8).save(pred_path)

                sample_count += 1
                print(f"[Export] 已导出样本 {sample_count}/{num_samples}")

    print(f"\n[Export] 完成！共导出 {sample_count} 个样本到: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='导出论文案例图：原图 + Defense 后渲染图')
    parser.add_argument('workspace', type=str, help='Pipeline 输出目录（如 output/experiments_output/single_xxx）')
    parser.add_argument('--num_samples', type=int, default=5, help='导出样本数（默认 5）')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（默认 <workspace>/paper_cases）')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 编号（默认 0）')
    parser.add_argument('--data_type', type=str, default='both', choices=['target', 'source', 'both'],
                        help='数据类型：target（攻击目标）、source（原始类别）或 both（两者都导出），默认 both')
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    workspace = os.path.abspath(os.path.expanduser(args.workspace))
    if not os.path.isdir(workspace):
        raise FileNotFoundError(f"工作目录不存在: {workspace}")

    output_dir = args.output_dir or os.path.join(workspace, 'paper_cases')

    print("=" * 60)
    print("导出论文案例图")
    print("=" * 60)
    print(f"工作目录: {workspace}")
    print(f"输出目录: {output_dir}")
    print(f"样本数: {args.num_samples}")
    print(f"数据类型: {args.data_type}")
    print(f"设备: {device}")
    print("=" * 60)

    # 加载配置
    config = load_pipeline_config(workspace)

    # 加载防御模型（同时返回 opt）
    model, opt = load_defense_model(workspace, config, device=device)

    # 根据 data_type 决定导出哪些数据
    data_types_to_export = []
    if args.data_type == 'both':
        data_types_to_export = ['source', 'target']
    else:
        data_types_to_export = [args.data_type]

    for data_type in data_types_to_export:
        print(f"\n{'=' * 60}")
        print(f"导出 {data_type.upper()} 数据")
        print(f"{'=' * 60}")

        # 准备数据加载器
        data_manager = DataManager(config, opt)
        if data_type == 'target':
            data_manager.setup_dataloaders(train=True, val=False, subset='target')
            data_loader = data_manager.train_loader
            sub_output_dir = os.path.join(output_dir, 'target')
            print(f"[Export] 使用 target 数据（攻击目标类别）")
        else:
            data_manager.setup_dataloaders(train=False, val=True, subset='source')
            data_loader = data_manager.val_loader
            sub_output_dir = os.path.join(output_dir, 'source')
            print(f"[Export] 使用 source 数据（原始类别，Defense 后未被攻击）")

        # 导出成对图像
        export_paired_images(
            model=model,
            data_loader=data_loader,
            output_dir=sub_output_dir,
            num_samples=args.num_samples,
            device=device,
        )

    print(f"\n{'=' * 60}")
    print(f"全部完成！输出目录: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

