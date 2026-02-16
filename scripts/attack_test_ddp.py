"""
攻击测试主脚本 - DDP版本（支持多GPU并行训练）
"""

import sys
sys.path.append('/mnt/huangjiaxin/3d-defense/LGM')
sys.path.append('/mnt/huangjiaxin/3d-defense')

import os
import yaml
import argparse

# 设置HuggingFace镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.distributed as dist
import numpy as np
from datetime import datetime
from pathlib import Path
from accelerate import Accelerator
from tqdm import tqdm

from core.options import config_defaults
from methods.model_loader import load_lgm_model, apply_lora
from methods.data_loader import create_dataloader
from methods.auto_finetune import AutoFineTuner
from methods.evaluator import Evaluator
from methods.attack_scenarios import create_attack_scenario


def load_config(config_path):
    """加载yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='LGM攻击测试 - DDP版本')
    parser.add_argument('--config', type=str,
                       default='configs/attack_config.yaml',
                       help='配置文件路径')
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    args = parse_args()

    # 初始化Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,  # 我们在AutoFineTuner中处理梯度累计
        mixed_precision='no',  # 使用FP32
    )

    # 只在主进程打印
    if accelerator.is_main_process:
        print("=" * 80)
        print("DDP并行训练")
        print("=" * 80)
        print(f"[INFO] 进程数: {accelerator.num_processes}")
        print(f"[INFO] 当前进程: {accelerator.process_index}")

    # 加载配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    config = load_config(config_path)

    # 设置随机种子
    set_seed(config['misc']['seed'])

    # 创建工作目录（只在主进程）
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        attack_scenario = config['attack']['scenario']
        workspace = os.path.join(config['misc']['workspace'], f"{attack_scenario}_{timestamp}_ddp")
        os.makedirs(workspace, exist_ok=True)
        print(f"[INFO] 工作目录: {workspace}")
    else:
        workspace = None

    # 广播workspace到所有进程
    if dist.is_initialized():
        workspace_list = [workspace]
        dist.broadcast_object_list(workspace_list, src=0)
        workspace = workspace_list[0]
    elif workspace is None:
        raise RuntimeError("非主进程无法获取workspace路径")

    # 创建LGM配置
    model_size = config['model']['size']
    if model_size not in config_defaults:
        raise ValueError(f"不支持的模型大小: {model_size}")

    opt = config_defaults[model_size]

    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("步骤1: 加载LGM模型")
        print("=" * 80)

    # 加载模型
    model = load_lgm_model(
        opt=opt,
        resume_path=config['model']['resume'],
        device='cuda',
        dtype=torch.float32,
    )

    # 应用LoRA或全量微调
    training_mode = config['training']['mode']
    if training_mode == 'lora':
        if accelerator.is_main_process:
            print("[INFO] 应用 LoRA 微调")
        model = apply_lora(
            model=model,
            target_modules=config['lora']['target_modules'],
            r=config['lora']['r'],
            lora_alpha=config['lora']['alpha'],
            lora_dropout=config['lora']['dropout'],
        )
    elif training_mode == 'full':
        if accelerator.is_main_process:
            print("[INFO] 使用全量微调")

    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("步骤2: 准备数据")
        print("=" * 80)

    # 创建数据加载器
    dataloader = create_dataloader(
        data_root=config['data']['root'],
        categories=config['data']['categories'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        shuffle=True,
        max_samples=config['data']['max_samples'],
        num_input_views=4,
        num_supervision_views=config['data'].get('num_supervision_views', 4),
        input_size=opt.input_size,
        fovy=opt.fovy,
        view_selector=config['data']['view_selector'],
        angle_offset=config['data']['angle_offset'],
        samples_per_object=config['data'].get('samples_per_object', 1),
    )

    if accelerator.is_main_process:
        print(f"[INFO] 数据加载器创建完成，共 {len(dataloader)} 个批次")

    # 创建优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.95),
    )

    # 使用Accelerator准备模型、优化器和数据加载器
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("步骤3: 开始训练")
        print("=" * 80)

    # 训练循环
    num_epochs = config['training']['num_epochs']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    gradient_clip = config['training']['gradient_clip']

    for epoch in range(1, num_epochs + 1):
        if accelerator.is_main_process:
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")

        model.train()
        total_loss = 0
        total_loss_lpips = 0
        total_psnr = 0
        num_batches = 0

        # 创建进度条（只在主进程显示）
        if accelerator.is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", ncols=100)
        else:
            pbar = dataloader

        for batch_idx, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                # 准备数据（简化版，直接使用batch）
                input_images = batch['input_images']
                supervision_images = batch['supervision_images']
                supervision_masks = batch['supervision_masks']  # 使用真实的alpha mask
                supervision_transforms = batch['supervision_transforms']

                B, V_sup, _, H, W = supervision_images.shape

                # 调整监督图像尺寸
                if H != model.module.opt.output_size or W != model.module.opt.output_size:
                    supervision_images = torch.nn.functional.interpolate(
                        supervision_images.view(B * V_sup, 3, H, W),
                        size=(model.module.opt.output_size, model.module.opt.output_size),
                        mode='bilinear',
                        align_corners=False
                    ).view(B, V_sup, 3, model.module.opt.output_size, model.module.opt.output_size)

                    # 同时调整mask尺寸
                    supervision_masks = torch.nn.functional.interpolate(
                        supervision_masks.view(B * V_sup, 1, H, W),
                        size=(model.module.opt.output_size, model.module.opt.output_size),
                        mode='bilinear',
                        align_corners=False
                    ).view(B, V_sup, 1, model.module.opt.output_size, model.module.opt.output_size)

                # 准备相机参数
                cam_poses = supervision_transforms.clone()
                cam_poses[:, :, :3, 1:3] *= -1

                cam_view = torch.inverse(cam_poses).transpose(-2, -1)

                tan_half_fov = torch.tan(torch.tensor(0.5 * torch.pi * model.module.opt.fovy / 180.0))
                proj_matrix = torch.zeros(4, 4, device=input_images.device)
                proj_matrix[0, 0] = 1 / tan_half_fov
                proj_matrix[1, 1] = 1 / tan_half_fov
                proj_matrix[2, 2] = (model.module.opt.zfar + model.module.opt.znear) / (model.module.opt.zfar - model.module.opt.znear)
                proj_matrix[3, 2] = -(model.module.opt.zfar * model.module.opt.znear) / (model.module.opt.zfar - model.module.opt.znear)
                proj_matrix[2, 3] = 1

                cam_view_proj = cam_view @ proj_matrix.unsqueeze(0).unsqueeze(0)
                cam_pos = -cam_poses[..., :3, 3]

                # 使用真实的alpha mask，而不是全1
                masks_output = supervision_masks

                data = {
                    'input': input_images,
                    'images_output': supervision_images,
                    'masks_output': masks_output,
                    'cam_view': cam_view,
                    'cam_view_proj': cam_view_proj,
                    'cam_pos': cam_pos,
                }

                # 前向传播
                results = model(data, step_ratio=1.0)
                loss = results['loss'] / gradient_accumulation_steps

                # 反向传播
                accelerator.backward(loss)

                # 梯度裁剪和优化器步进
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), gradient_clip)

                optimizer.step()
                optimizer.zero_grad()

                # 累积指标
                total_loss += results['loss'].item()
                total_loss_lpips += results.get('loss_lpips', torch.tensor(0.0)).item()
                total_psnr += results.get('psnr', torch.tensor(0.0)).item()
                num_batches += 1

                # 更新进度条（只在主进程）
                if accelerator.is_main_process:
                    pbar.set_postfix({
                        'loss': f"{results['loss'].item():.4f}",
                        'lpips': f"{results.get('loss_lpips', torch.tensor(0.0)).item():.4f}",
                        'psnr': f"{results.get('psnr', torch.tensor(0.0)).item():.2f}"
                    })

        # 计算平均指标
        avg_loss = total_loss / num_batches
        avg_loss_lpips = total_loss_lpips / num_batches
        avg_psnr = total_psnr / num_batches

        if accelerator.is_main_process:
            print(f"[INFO] Epoch {epoch} 平均损失:")
            print(f"  loss: {avg_loss:.4f}")
            print(f"  loss_lpips: {avg_loss_lpips:.4f}")
            print(f"  psnr: {avg_psnr:.2f}")

        # 保存检查点（只在主进程）
        if epoch % 5 == 0 or epoch == num_epochs:
            # 所有进程都需要等待同步
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                checkpoint_path = os.path.join(workspace, f"checkpoint_epoch{epoch}.pt")
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f"[INFO] 检查点已保存到: {checkpoint_path}")

    # 等待所有进程完成训练
    accelerator.wait_for_everyone()

    # 评估攻击效果（只在主进程）
    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("步骤4: 评估攻击效果")
        print("=" * 80)

        # 创建评估器
        unwrapped_model = accelerator.unwrap_model(model)
        evaluator = Evaluator(model=unwrapped_model, device=accelerator.device)

        # 生成测试样本
        print("[INFO] 生成测试样本...")
        test_batch = next(iter(dataloader))
        test_images = test_batch['input_images'].to(accelerator.device)

        # 调试：检查输入图像和alpha通道
        print(f"[DEBUG] Input images shape: {test_images.shape}")
        print(f"[DEBUG] Input RGB (前3通道) range: [{test_images[:, :, :3].min():.4f}, {test_images[:, :, :3].max():.4f}]")
        print(f"[DEBUG] Input RGB mean: {test_images[:, :, :3].mean():.4f}")

        # 检查监督图像
        if 'supervision_images' in test_batch:
            sup_images = test_batch['supervision_images']
            print(f"[DEBUG] Supervision images shape: {sup_images.shape}")
            print(f"[DEBUG] Supervision RGB range: [{sup_images.min():.4f}, {sup_images.max():.4f}]")
            print(f"[DEBUG] Supervision RGB mean: {sup_images.mean():.4f}")

            # 保存一张监督图像看看
            import torchvision
            sample_img = sup_images[0, 0]  # 第一个batch的第一个视图
            torchvision.utils.save_image(sample_img, os.path.join(workspace, "debug_supervision.png"))
            print(f"[DEBUG] 已保存监督图像样本到: {os.path.join(workspace, 'debug_supervision.png')}")

        # 生成Gaussian
        gaussians = evaluator.generate_gaussians(test_images)

        # 调试：检查Gaussian参数
        print(f"[DEBUG] Gaussians shape: {gaussians.shape}")
        print(f"[DEBUG] Opacity range: [{gaussians[:, :, 3].min():.4f}, {gaussians[:, :, 3].max():.4f}]")
        print(f"[DEBUG] Opacity mean: {gaussians[:, :, 3].mean():.4f}")
        print(f"[DEBUG] RGB range: [{gaussians[:, :, 11:].min():.4f}, {gaussians[:, :, 11:].max():.4f}]")
        print(f"[DEBUG] Position range: [{gaussians[:, :, 0:3].min():.4f}, {gaussians[:, :, 0:3].max():.4f}]")

        # 保存PLY文件和渲染视频
        num_samples = min(3, test_images.shape[0])
        for i in range(num_samples):
            ply_path = os.path.join(workspace, f"sample_{i}.ply")
            evaluator.save_ply(gaussians[i:i+1], ply_path)
            print(f"[INFO] PLY文件已保存: {ply_path}")

        # 渲染360度视频
        for i in range(num_samples):
            video_path = os.path.join(workspace, f"sample_{i}_360.mp4")
            evaluator.render_360_video(gaussians[i:i+1], video_path)
            print(f"[INFO] 视频已保存: {video_path}")

    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("训练完成！")
        print("=" * 80)
        print(f"结果保存在: {workspace}")


if __name__ == '__main__':
    main()
