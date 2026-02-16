"""
Zero-Shot测试脚本 - 测试原始LGM模型的生成能力
"""

import sys
sys.path.append('/mnt/huangjiaxin/3d-defense/LGM')
sys.path.append('/mnt/huangjiaxin/3d-defense')

import os
import yaml
import argparse

# 设置HuggingFace镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 解析命令行参数（在导入torch之前）
def parse_args_early():
    """早期解析参数以设置GPU"""
    parser = argparse.ArgumentParser(description='LGM Zero-Shot测试')
    parser.add_argument('--config', type=str,
                       default='configs/zero_shot_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--gpu', type=str, default=None,
                       help='指定GPU ID，如 "0" 或 "0,1,2,3"（优先级高于配置文件）')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='测试样本数量')
    parser.add_argument('--save_ply', action='store_true',
                       help='是否保存PLY文件')
    parser.add_argument('--save_video', action='store_true',
                       help='是否保存360度视频')
    args, _ = parser.parse_known_args()
    return args

# 早期解析参数
early_args = parse_args_early()

# 设置GPU（优先级：命令行 > 环境变量 > 配置文件）
if early_args.gpu is not None:
    # 命令行指定的GPU（最高优先级）
    os.environ["CUDA_VISIBLE_DEVICES"] = early_args.gpu
    print(f"[INFO] 使用命令行指定的GPU: {early_args.gpu}")
elif "CUDA_VISIBLE_DEVICES" not in os.environ:
    # 如果环境变量没有设置，尝试从配置文件读取
    try:
        with open(early_args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config['model'].get('gpu_ids') is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config['model']['gpu_ids'])
            print(f"[INFO] 使用配置文件中的GPU: {config['model']['gpu_ids']}")
    except:
        pass
else:
    print(f"[INFO] 使用环境变量中的GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from core.options import config_defaults
from methods.model_loader import load_lgm_model
from methods.data_loader import create_dataloader
from methods.evaluator import Evaluator


def load_config(config_path):
    """加载yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='LGM Zero-Shot测试')
    parser.add_argument('--config', type=str,
                       default='configs/zero_shot_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--gpu', type=str, default=None,
                       help='指定GPU ID，如 "0" 或 "0,1,2,3"（优先级高于配置文件）')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='测试样本数量')
    parser.add_argument('--save_ply', action='store_true',
                       help='是否保存PLY文件')
    parser.add_argument('--save_video', action='store_true',
                       help='是否保存360度视频')
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate_model(model, dataloader, device, num_samples=10):
    """
    评估模型的zero-shot性能

    Args:
        model: LGM模型
        dataloader: 数据加载器
        device: 设备
        num_samples: 测试样本数量

    Returns:
        metrics: 评估指标字典
    """
    model.eval()

    total_loss = 0
    total_loss_lpips = 0
    total_psnr = 0
    num_batches = 0

    print(f"\n[INFO] 开始评估，共 {num_samples} 个样本...")

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Zero-Shot评估", total=min(num_samples, len(dataloader)))
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= num_samples:
                break

            # 准备数据
            input_images = batch['input_images'].to(device)
            supervision_images = batch['supervision_images'].to(device)
            supervision_transforms = batch['supervision_transforms'].to(device)

            B, V_sup, _, H, W = supervision_images.shape

            # 调整监督图像尺寸
            if H != model.opt.output_size or W != model.opt.output_size:
                supervision_images = torch.nn.functional.interpolate(
                    supervision_images.view(B * V_sup, 3, H, W),
                    size=(model.opt.output_size, model.opt.output_size),
                    mode='bilinear',
                    align_corners=False
                ).view(B, V_sup, 3, model.opt.output_size, model.opt.output_size)

            # 准备相机参数
            cam_poses = supervision_transforms
            cam_view = torch.inverse(cam_poses).transpose(-2, -1)

            tan_half_fov = torch.tan(torch.tensor(0.5 * torch.pi * model.opt.fovy / 180.0))
            proj_matrix = torch.zeros(4, 4, dtype=input_images.dtype, device=device)
            proj_matrix[0, 0] = 1 / tan_half_fov
            proj_matrix[1, 1] = 1 / tan_half_fov
            proj_matrix[2, 2] = (model.opt.zfar + model.opt.znear) / (model.opt.zfar - model.opt.znear)
            proj_matrix[3, 2] = -(model.opt.zfar * model.opt.znear) / (model.opt.zfar - model.opt.znear)
            proj_matrix[2, 3] = 1

            cam_view_proj = cam_view @ proj_matrix.unsqueeze(0).unsqueeze(0)
            cam_pos = -cam_poses[..., :3, 3]

            masks_output = torch.ones(B, V_sup, 1, model.opt.output_size, model.opt.output_size,
                                      dtype=input_images.dtype, device=device)

            data = {
                'input': input_images,
                'images_output': supervision_images,
                'masks_output': masks_output,
                'cam_view': cam_view,
                'cam_view_proj': cam_view_proj,
                'cam_pos': cam_pos,
            }

            # 前向传播
            results = model.forward(data, step_ratio=1.0)

            # 累积指标
            total_loss += results['loss'].item()
            total_loss_lpips += results.get('loss_lpips', torch.tensor(0.0)).item()
            total_psnr += results.get('psnr', torch.tensor(0.0)).item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{results['loss'].item():.4f}",
                'lpips': f"{results.get('loss_lpips', 0):.4f}",
                'psnr': f"{results.get('psnr', 0):.2f}",
            })

    # 计算平均指标
    metrics = {
        'avg_loss': total_loss / num_batches,
        'avg_loss_lpips': total_loss_lpips / num_batches,
        'avg_psnr': total_psnr / num_batches,
        'num_samples': num_batches,
    }

    return metrics


def main():
    args = parse_args()

    # 加载配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    print(f"[INFO] 加载配置文件: {config_path}")
    config = load_config(config_path)

    # 设置随机种子
    set_seed(config['misc']['seed'])

    # 创建工作目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    workspace = os.path.join(config['misc']['workspace'], f"zero_shot_{timestamp}")
    os.makedirs(workspace, exist_ok=True)
    print(f"[INFO] 工作目录: {workspace}")

    # 创建LGM配置
    model_size = config['model']['size']
    if model_size not in config_defaults:
        raise ValueError(f"不支持的模型大小: {model_size}")

    opt = config_defaults[model_size]
    print(f"\n[INFO] 使用 {model_size} 配置")

    print("\n" + "=" * 80)
    print("步骤1: 加载预训练LGM模型")
    print("=" * 80)

    # 加载模型
    model = load_lgm_model(
        opt=opt,
        resume_path=config['model']['resume'],
        device=config['model']['device'],
        dtype=torch.float32,
    )

    print(f"[INFO] 模型加载完成")

    print("\n" + "=" * 80)
    print("步骤2: 准备测试数据")
    print("=" * 80)

    # 创建数据加载器
    dataloader = create_dataloader(
        data_root=config['data']['root'],
        categories=config['data']['categories'],
        batch_size=1,  # Zero-shot测试使用batch_size=1
        num_workers=config['data']['num_workers'],
        shuffle=False,  # 不打乱，保证可重复性
        max_samples=args.num_samples,
        num_input_views=4,
        input_size=opt.input_size,
        fovy=opt.fovy,
        view_selector=config['data']['view_selector'],
        angle_offset=config['data']['angle_offset'],
    )

    print(f"[INFO] 数据加载器创建完成")

    print("\n" + "=" * 80)
    print("步骤3: Zero-Shot评估")
    print("=" * 80)

    # 评估模型
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=config['model']['device'],
        num_samples=args.num_samples,
    )

    print("\n" + "=" * 80)
    print("Zero-Shot评估结果")
    print("=" * 80)
    print(f"测试样本数: {metrics['num_samples']}")
    print(f"平均损失 (loss): {metrics['avg_loss']:.4f}")
    print(f"平均LPIPS损失: {metrics['avg_loss_lpips']:.4f}")
    print(f"平均PSNR: {metrics['avg_psnr']:.2f} dB")

    # 保存结果
    results_file = os.path.join(workspace, 'zero_shot_results.txt')
    with open(results_file, 'w') as f:
        f.write("Zero-Shot评估结果\n")
        f.write("=" * 80 + "\n")
        f.write(f"测试样本数: {metrics['num_samples']}\n")
        f.write(f"平均损失 (loss): {metrics['avg_loss']:.4f}\n")
        f.write(f"平均LPIPS损失: {metrics['avg_loss_lpips']:.4f}\n")
        f.write(f"平均PSNR: {metrics['avg_psnr']:.2f} dB\n")

    print(f"\n[INFO] 结果已保存到: {results_file}")

    # 可选：生成样本
    if args.save_ply or args.save_video:
        print("\n" + "=" * 80)
        print("步骤4: 生成样本")
        print("=" * 80)

        evaluator = Evaluator(model=model, device=config['model']['device'])

        # 获取第一个batch
        test_batch = next(iter(dataloader))
        test_images = test_batch['input_images'].to(config['model']['device'])

        # 生成Gaussian
        print("[INFO] 生成Gaussian...")
        gaussians = evaluator.generate_gaussians(test_images)

        if args.save_ply:
            ply_path = os.path.join(workspace, "zero_shot_sample.ply")
            evaluator.save_ply(gaussians, ply_path)
            print(f"[INFO] PLY文件已保存: {ply_path}")

        if args.save_video:
            video_path = os.path.join(workspace, "zero_shot_sample_360.mp4")
            evaluator.render_360_video(gaussians, video_path)
            print(f"[INFO] 360度视频已保存: {video_path}")

    print("\n" + "=" * 80)
    print("Zero-Shot测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

