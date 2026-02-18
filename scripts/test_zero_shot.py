#!/usr/bin/env python3
"""
LGM Zero-Shot 测试脚本 - 评估原始模型在 OmniObject3D 数据集上的性能

测试内容：
- 使用未经微调的原始 LGM 模型
- 在 OmniObject3D 数据集上测试
- 支持多次采样（不同 elevation 层级的正交视图）
- 计算评估指标：PSNR, LPIPS, MSE
- 统计平均性能

使用方法：
    # 基础测试（每个物体采样1次）
    python test_zero_shot.py --num_objects 10 --samples_per_object 1

    # 多视角泛化测试（每个物体采样5次，测试不同elevation）
    python test_zero_shot.py --num_objects 10 --samples_per_object 5

    # 指定类别测试
    python test_zero_shot.py --categories pitaya flash_light --samples_per_object 3
"""

import argparse
import os
import sys
import torch
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_core import ConfigManager
from models import ModelManager
from data import OmniObject3DDataset
from evaluation import Evaluator
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='LGM Zero-Shot 测试')
    parser.add_argument('--config', type=str, default='configs/attack_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--data_root', type=str, default='datas',
                        help='数据根目录')
    parser.add_argument('--categories', type=str, nargs='+', default=['pitaya', 'banana', 'apple'],
                        help='测试类别（默认3个类别）')
    parser.add_argument('--num_objects', type=int, default=2,
                        help='每个类别测试的物体数')
    parser.add_argument('--samples_per_object', type=int, default=2,
                        help='每个物体采样次数（测试不同elevation层级）')
    parser.add_argument('--gpu', type=int, default=0,
                        help='使用的 GPU 编号')
    parser.add_argument('--output_dir', type=str, default='output/zero_shot_test',
                        help='输出目录')
    parser.add_argument('--save_visualizations', type=int, default=3,
                        help='保存前N个样本的可视化结果（PLY、渲染图像、360视频）')
    parser.add_argument('--num_supervision_views', type=int, default=4,
                        help='用于评估的监督视图数')
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda'

    print("=" * 80)
    print("LGM Zero-Shot 测试 - OmniObject3D 数据集")
    print("=" * 80)
    print(f"\n测试配置：")
    print(f"  数据集: OmniObject3D")
    print(f"  类别: {args.categories if args.categories else '所有类别'}")
    print(f"  每类物体数: {args.num_objects}")
    print(f"  每物体采样数: {args.samples_per_object}")
    print(f"  监督视图数: {args.num_supervision_views}")
    print()

    # 1. 加载配置
    print("[1/5] 加载配置...")
    config_mgr = ConfigManager(args.config)
    config = config_mgr.config
    print(f"  模型大小: {config['model']['size']}")
    print(f"  权重路径: {config['model'].get('resume', 'None')}")

    # 2. 加载模型（不应用LoRA）
    print("\n[2/5] 加载原始 LGM 模型...")
    model_mgr = ModelManager(config)
    model_mgr.setup(device=device, apply_lora=False)
    model = model_mgr.model
    model.eval()
    print(f"  ✓ 模型加载完成（Zero-Shot，未微调）")

    # 3. 加载数据集
    print("\n[3/5] 加载数据集...")
    dataset = OmniObject3DDataset(
        data_root=args.data_root,
        num_input_views=4,
        num_supervision_views=args.num_supervision_views,
        input_size=256,
        view_selector='orthogonal',
        categories=args.categories,
        max_samples_per_category=args.num_objects,
        samples_per_object=args.samples_per_object,
    )
    print(f"  ✓ 数据集大小: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # 4. 创建评估器
    print("\n[4/5] 创建评估器...")
    evaluator = Evaluator(model, device=device)
    print(f"  ✓ 评估器创建完成")

    # 5. 测试并收集指标
    print("\n[5/5] 开始 Zero-Shot 测试...")
    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = []
    results_by_category = {}

    for i, batch in enumerate(tqdm(dataloader, desc="测试进度")):
        category = batch['category'][0]
        obj_name = batch['object'][0]

        # 移动数据到设备
        input_images = batch['input_images'].to(device)
        supervision_images = batch['supervision_images'].to(device)
        supervision_transforms = batch['supervision_transforms'].to(device)

        # 生成 Gaussian 参数
        with torch.no_grad():
            gaussians = evaluator.generate_gaussians(input_images)

            # 渲染监督视图
            rendered_images = evaluator.render_views(
                gaussians,
                supervision_transforms,
                model.opt
            )

            # 计算指标
            metrics = evaluator.compute_metrics(
                rendered_images,
                supervision_images,
                mask=None  # 使用所有像素
            )

        # 记录结果
        sample_result = {
            'category': category,
            'object': obj_name,
            'psnr': metrics['psnr'],
            'lpips': metrics['lpips'],
            'mse': metrics['mse'],
        }
        all_metrics.append(sample_result)

        # 按类别统计
        if category not in results_by_category:
            results_by_category[category] = []
        results_by_category[category].append(sample_result)

        # 保存可视化（可选）
        if args.save_visualizations > 0 and i < args.save_visualizations:
            sample_dir = os.path.join(args.output_dir, f"sample_{i:03d}_{category}_{obj_name}")
            os.makedirs(sample_dir, exist_ok=True)

            print(f"\n  保存可视化结果到: {sample_dir}")

            # 保存 PLY
            ply_path = os.path.join(sample_dir, "output.ply")
            evaluator.save_ply(gaussians, ply_path)
            print(f"    ✓ PLY: output.ply")

            # 保存渲染结果对比
            for view_idx in range(min(4, rendered_images.shape[1])):
                # Ground truth
                gt_img = supervision_images[0, view_idx].permute(1, 2, 0).cpu().numpy()
                gt_img = (gt_img * 255).astype(np.uint8)
                Image.fromarray(gt_img).save(
                    os.path.join(sample_dir, f"view{view_idx}_gt.png")
                )

                # Rendered
                render_img = rendered_images[0, view_idx].permute(1, 2, 0).cpu().numpy()
                render_img = (render_img * 255).astype(np.uint8)
                Image.fromarray(render_img).save(
                    os.path.join(sample_dir, f"view{view_idx}_render.png")
                )

            print(f"    ✓ 渲染对比: {rendered_images.shape[1]}组")

            # 渲染 360 度视频
            video_path = os.path.join(sample_dir, "360_video.mp4")
            print(f"    渲染360度视频...")
            evaluator.render_360_video(gaussians, video_path)
            print(f"    ✓ 360视频: 360_video.mp4")

    # 计算统计结果
    print("\n" + "=" * 80)
    print("Zero-Shot 测试结果")
    print("=" * 80)

    # 总体统计
    avg_psnr = np.mean([m['psnr'] for m in all_metrics])
    avg_lpips = np.mean([m['lpips'] for m in all_metrics])
    avg_mse = np.mean([m['mse'] for m in all_metrics])

    print(f"\n总体性能（{len(all_metrics)}个样本）：")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  LPIPS: {avg_lpips:.4f}")
    print(f"  MSE: {avg_mse:.6f}")

    # 按类别统计
    print(f"\n按类别统计：")
    for category in sorted(results_by_category.keys()):
        cat_metrics = results_by_category[category]
        cat_psnr = np.mean([m['psnr'] for m in cat_metrics])
        cat_lpips = np.mean([m['lpips'] for m in cat_metrics])
        print(f"  {category} ({len(cat_metrics)}样本): "
              f"PSNR={cat_psnr:.2f}dB, LPIPS={cat_lpips:.4f}")

    # 保存结果到 JSON
    results = {
        'config': {
            'categories': args.categories,
            'num_objects': args.num_objects,
            'samples_per_object': args.samples_per_object,
            'num_supervision_views': args.num_supervision_views,
        },
        'overall': {
            'psnr': float(avg_psnr),
            'lpips': float(avg_lpips),
            'mse': float(avg_mse),
            'num_samples': len(all_metrics),
        },
        'by_category': {
            cat: {
                'psnr': float(np.mean([m['psnr'] for m in results_by_category[cat]])),
                'lpips': float(np.mean([m['lpips'] for m in results_by_category[cat]])),
                'num_samples': len(results_by_category[cat]),
            }
            for cat in results_by_category
        },
        'all_samples': all_metrics,
    }

    results_path = os.path.join(args.output_dir, 'zero_shot_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {results_path}")
    if args.save_visualizations > 0:
        print(f"可视化结果（前{args.save_visualizations}个样本）: {args.output_dir}/sample_*/")
        print(f"  - PLY文件: output.ply")
        print(f"  - 渲染对比: view*_gt.png / view*_render.png")
        print(f"  - 360度视频: 360_video.mp4")


if __name__ == '__main__':
    main()

