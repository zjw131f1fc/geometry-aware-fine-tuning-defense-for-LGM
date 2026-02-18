#!/usr/bin/env python3
"""
LGM 推理测试脚本 - 使用 Objaverse 渲染的 4 视图数据

输入：4 个正交视图（0°, 90°, 180°, 270°）
- 0° = 正视图（Front）
- 90° = 左视图（Left）
- 180° = 背视图（Back）
- 270° = 右视图（Right）

支持整体旋转：通过 angle_offset 参数给所有视角加同一个偏移
- angle_offset=0: 标准正交视图（0°, 90°, 180°, 270°）
- angle_offset=15: 偏移正交视图（15°, 105°, 195°, 285°）
- angle_offset=45: 对角视图（45°, 135°, 225°, 315°）

使用方法：
    # 标准正交视图
    python test_inference.py --data_root datas/objaverse_rendered --num_samples 5

    # 偏移 45 度
    python test_inference.py --data_root datas/objaverse_rendered --angle_offset 45

    # 指定输出目录
    python test_inference.py --data_root datas/objaverse_rendered --output_dir output/test
"""

import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_core import ConfigManager
from models import ModelManager
from data import ObjaverseRenderedDataset
from evaluation import Evaluator
from torch.utils.data import DataLoader
from kiui.cam import orbit_camera


def parse_args():
    parser = argparse.ArgumentParser(description='LGM 推理测试 - 4 视图输入')
    parser.add_argument('--config', type=str, default='configs/attack_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--data_root', type=str, default='datas/objaverse_rendered',
                        help='Objaverse 渲染数据根目录')
    parser.add_argument('--gpu', type=int, default=0,
                        help='使用的 GPU 编号')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='测试样本数')
    parser.add_argument('--output_dir', type=str, default='output/inference_test',
                        help='输出目录')
    parser.add_argument('--angle_offset', type=float, default=0.0,
                        help='视角偏移量（度）。0=标准正交视图（0°,90°,180°,270°），'
                             '15=偏移视图（15°,105°,195°,285°）')
    parser.add_argument('--no_video', action='store_true',
                        help='不渲染 360 度视频（默认会渲染）')
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda'

    print("=" * 80)
    print("LGM 推理测试 - 4 视图输入")
    print("=" * 80)
    print(f"\n视角配置：")
    print(f"  基准角度：0°, 90°, 180°, 270°（正交视图）")
    print(f"  角度偏移：{args.angle_offset}°")
    if args.angle_offset != 0:
        angles = [(i * 90 + args.angle_offset) % 360 for i in range(4)]
        print(f"  实际角度：{angles[0]:.1f}°, {angles[1]:.1f}°, {angles[2]:.1f}°, {angles[3]:.1f}°")
    print()

    # 1. 加载配置
    print("[1/5] 加载配置...")
    config_mgr = ConfigManager(args.config)
    config = config_mgr.config
    print(f"  模型大小: {config['model']['size']}")
    print(f"  权重路径: {config['model'].get('resume', 'None')}")

    # 2. 加载模型
    print("\n[2/5] 加载模型...")
    model_mgr = ModelManager(config)
    # 推理时不应用 LoRA（除非 LoRA 已经训练好）
    model_mgr.setup(device=device, apply_lora=False)
    model = model_mgr.model
    model.eval()
    print(f"  ✓ 模型加载完成")

    # 3. 加载数据集
    print("\n[3/5] 加载数据集...")
    print(f"  数据根目录: {args.data_root}")
    print(f"  视角选择器: orthogonal")
    print(f"  角度偏移: {args.angle_offset}°")

    dataset = ObjaverseRenderedDataset(
        data_root=args.data_root,
        num_input_views=4,
        num_supervision_views=4,
        input_size=256,
        view_selector='orthogonal',
        angle_offset=args.angle_offset,  # 整体旋转偏移
        max_samples=args.num_samples,
        samples_per_object=1,  # 不使用随机偏移
    )
    print(f"  ✓ 数据集大小: {len(dataset)}")

    # 创建 DataLoader
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

    # 5. 推理并保存结果
    print("\n[5/5] 开始推理...")
    os.makedirs(args.output_dir, exist_ok=True)

    for i, batch in enumerate(dataloader):
        uuid = batch['uuid'][0]
        print(f"\n样本 {i+1}/{len(dataset)}: {uuid}")

        # 移动数据到设备
        input_images = batch['input_images'].to(device)  # [1, 4, 9, 256, 256]

        # 保存输入图像（前3通道是RGB，已经ImageNet归一化）
        print(f"  保存输入图像...")
        input_dir = os.path.join(args.output_dir, f"{uuid}_inputs")
        os.makedirs(input_dir, exist_ok=True)

        # 反归一化并保存
        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        view_names = ['0deg_front', '90deg_left', '180deg_back', '270deg_right']
        for view_idx in range(4):
            # 提取RGB通道并反归一化
            img = input_images[0, view_idx, :3, :, :]  # [3, 256, 256]
            img = img * IMAGENET_STD[0] + IMAGENET_MEAN[0]  # 反归一化
            img = torch.clamp(img, 0, 1)

            # 转换为PIL图像并保存
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            # 计算实际角度
            actual_angle = (view_idx * 90 + args.angle_offset) % 360
            filename = f"view_{view_idx}_{view_names[view_idx]}_actual_{actual_angle:.0f}deg.png"
            img_pil.save(os.path.join(input_dir, filename))

        print(f"  ✓ 输入图像已保存: {input_dir}")

        # 生成 Gaussian 参数
        print(f"  生成 Gaussian 参数...")
        with torch.no_grad():
            gaussians = evaluator.generate_gaussians(input_images)

        print(f"  Gaussian shape: {gaussians.shape}")

        # 保存 PLY 文件
        ply_path = os.path.join(args.output_dir, f"{uuid}.ply")
        evaluator.save_ply(gaussians, ply_path)

        # 渲染正交视图
        print(f"  渲染正交视图...")
        render_dir = os.path.join(args.output_dir, f"{uuid}_renders")
        os.makedirs(render_dir, exist_ok=True)

        # 渲染4个正交视角
        opt = model.opt
        tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
        proj_matrix[0, 0] = 1 / tan_half_fov
        proj_matrix[1, 1] = 1 / tan_half_fov
        proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[2, 3] = 1

        for angle_idx, azimuth in enumerate([0, 90, 180, 270]):
            cam_pose = orbit_camera(0, azimuth, radius=1.5, opengl=True)
            cam_pose = torch.from_numpy(cam_pose).unsqueeze(0).to(device)
            cam_pose[:, :3, 1:3] *= -1

            cam_view = torch.inverse(cam_pose).transpose(1, 2)
            cam_view_proj = cam_view @ proj_matrix
            cam_pos = -cam_pose[:, :3, 3]

            result = model.gs.render(
                gaussians,
                cam_view.unsqueeze(0),
                cam_view_proj.unsqueeze(0),
                cam_pos.unsqueeze(0),
            )

            image = result['image'].squeeze(1).permute(0, 2, 3, 1)
            image = (image.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(image[0])
            img_pil.save(os.path.join(render_dir, f"render_{azimuth:03d}.png"))

        print(f"  ✓ 正交视图已保存: {render_dir}")

        # 渲染 360 度视频（默认启用）
        if not args.no_video:
            video_path = os.path.join(args.output_dir, f"{uuid}_360.mp4")
            print(f"  渲染 360 度视频...")
            evaluator.render_360_video(gaussians, video_path)
            print(f"  ✓ 视频已保存: {video_path}")

        # 计算 Gaussian 统计信息
        print(f"  Gaussian 统计:")
        print(f"    Position: mean={gaussians[0, :, 0:3].mean().item():.4f}, "
              f"std={gaussians[0, :, 0:3].std().item():.4f}")
        print(f"    Opacity: mean={gaussians[0, :, 3].mean().item():.4f}, "
              f"std={gaussians[0, :, 3].std().item():.4f}")
        print(f"    Scale: mean={gaussians[0, :, 4:7].mean().item():.4f}, "
              f"std={gaussians[0, :, 4:7].std().item():.4f}")

    print("\n" + "=" * 80)
    print("推理测试完成！")
    print("=" * 80)
    print(f"\n结果保存在: {args.output_dir}")
    print(f"  - PLY 文件: {len(dataset)} 个")
    if not args.no_video:
        print(f"  - 360 度视频: {len(dataset)} 个")
    print(f"  - 输入图像: {len(dataset)} 组（每组4张，标注了实际角度）")
    print(f"  - 渲染视图: {len(dataset)} 组")


if __name__ == '__main__':
    main()

