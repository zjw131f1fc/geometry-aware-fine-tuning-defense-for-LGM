"""
攻击训练脚本 - DDP 版本（使用新架构）

支持多 GPU 并行训练，使用 Accelerate 库
每个 epoch 结束后渲染对比图，直观展示训练变化
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.distributed as dist
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from accelerate import Accelerator
from tqdm import tqdm

from project_core import ConfigManager
from models import ModelManager
from data import DataManager
from training import AutoFineTuner
from evaluation import Evaluator
from tools import set_seed, get_base_model


def parse_args():
    parser = argparse.ArgumentParser(description='攻击训练 - DDP 版本')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    return parser.parse_args()


@torch.no_grad()
def render_epoch_samples(evaluator, val_loader, workspace, epoch, num_samples=3):
    """每个 epoch 结束后渲染对比图"""
    render_dir = os.path.join(workspace, 'renders')

    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break

        input_images = batch['input_images'].to('cuda')
        gt_images = batch.get('supervision_images')
        if gt_images is not None:
            gt_images = gt_images.to('cuda')

        # 获取监督视角的相机姿态，用于从相同视角渲染（便于与 GT 对比）
        cam_poses = batch.get('supervision_transforms')
        if cam_poses is not None:
            cam_poses = cam_poses.to('cuda')

        gaussians = evaluator.generate_gaussians(input_images)

        evaluator.render_and_save(
            gaussians,
            save_dir=render_dir,
            prefix=f"epoch{epoch:02d}_",
            gt_images=gt_images,
            cam_poses=cam_poses,
        )

        # 第一个样本额外保存 PLY
        if i == 0:
            ply_path = os.path.join(workspace, f"epoch{epoch:02d}.ply")
            evaluator.save_ply(gaussians[0:1], ply_path)


@torch.no_grad()
def eval_source_samples(model, evaluator, source_val_loader, workspace, epoch,
                        device, num_samples=3):
    """每个 epoch 结束后评估 source 质量（PSNR/LPIPS）并渲染对比图"""
    model.eval()

    finetuner = AutoFineTuner(
        model=model, device=device,
        lr=1e-4, weight_decay=0, gradient_clip=1.0,
        mixed_precision='no', gradient_accumulation_steps=1,
    )

    total_psnr = 0
    total_lpips = 0
    num_batches = 0
    render_dir = os.path.join(workspace, 'source_renders')

    for i, batch in enumerate(source_val_loader):
        data = finetuner._prepare_data(batch)
        results = model.forward(data, step_ratio=1.0)
        total_psnr += results.get('psnr', torch.tensor(0.0)).item()
        total_lpips += results.get('loss_lpips', torch.tensor(0.0)).item()
        num_batches += 1

        if i < num_samples:
            input_images = batch['input_images'].to(device)
            gt_images = batch.get('supervision_images')
            if gt_images is not None:
                gt_images = gt_images.to(device)
            cam_poses = batch.get('supervision_transforms')
            if cam_poses is not None:
                cam_poses = cam_poses.to(device)

            gaussians = evaluator.generate_gaussians(input_images)
            evaluator.render_and_save(
                gaussians, save_dir=render_dir,
                prefix=f"epoch{epoch:02d}_source_{i}_",
                gt_images=gt_images, cam_poses=cam_poses,
            )

    del finetuner

    avg_psnr = total_psnr / max(num_batches, 1)
    avg_lpips = total_lpips / max(num_batches, 1)
    print(f"  [Source] PSNR={avg_psnr:.2f}, LPIPS={avg_lpips:.4f}")
    return {'source_psnr': avg_psnr, 'source_lpips': avg_lpips}


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='bf16',
    )

    is_main = accelerator.is_main_process

    if is_main:
        print("=" * 80)
        print("攻击训练 - DDP 版本")
        print("=" * 80)
        print(f"进程数: {accelerator.num_processes}")
        print(f"当前进程: {accelerator.process_index}")

    # 1. 加载配置
    config = ConfigManager(args.config).config
    set_seed(config['misc']['seed'])

    if is_main:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        workspace = os.path.join(config['misc']['workspace'], f"attack_{timestamp}_ddp")
        os.makedirs(workspace, exist_ok=True)
        print(f"工作目录: {workspace}")
    else:
        workspace = None

    if dist.is_initialized():
        workspace_list = [workspace]
        dist.broadcast_object_list(workspace_list, src=0)
        workspace = workspace_list[0]

    # 2. 加载模型
    if is_main:
        print("\n" + "=" * 80)
        print("加载模型")
        print("=" * 80)

    model_mgr = ModelManager(config)
    model_mgr.setup(device='cuda')
    model = model_mgr.model

    # 3. 加载数据
    if is_main:
        print("\n" + "=" * 80)
        print("加载数据")
        print("=" * 80)

    data_mgr = DataManager(config, model_mgr.opt)
    data_mgr.setup_dataloaders(train=True, val=True, subset='target')

    # Source 数据加载器（用于评估攻击对 source 质量的影响）
    source_data_mgr = DataManager(config, model_mgr.opt)
    source_data_mgr.setup_dataloaders(train=False, val=True, subset='source')
    source_val_loader = source_data_mgr.val_loader
    if is_main:
        print(f"  Source 验证数据: {len(source_val_loader.dataset)} 样本")

    # 4. 创建微调器
    training_config = config['training']
    finetuner = AutoFineTuner(
        model=model,
        device='cuda',
        lr=training_config['lr'],
        weight_decay=training_config['weight_decay'],
        gradient_clip=training_config['gradient_clip'],
        mixed_precision='no',  # Accelerator 已处理 bf16
        lambda_lpips=training_config.get('lambda_lpips', 1.0),
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
    )

    # 5. Accelerator 准备
    model, finetuner.optimizer, data_mgr.train_loader, data_mgr.val_loader = accelerator.prepare(
        model, finetuner.optimizer, data_mgr.train_loader, data_mgr.val_loader
    )
    finetuner.model = model

    # 6. 训练前渲染 baseline（epoch 0）
    if is_main:
        print("\n渲染 baseline（训练前）...")
        unwrapped = accelerator.unwrap_model(model)
        evaluator = Evaluator(unwrapped)
        render_epoch_samples(evaluator, data_mgr.val_loader, workspace, epoch=0)

        # Baseline source 质量
        print("  评估 baseline source 质量...")
        eval_source_samples(
            unwrapped, evaluator, source_val_loader, workspace,
            epoch=0, device='cuda',
        )

    # 7. 训练循环
    if is_main:
        print("\n" + "=" * 80)
        print("开始训练")
        print("=" * 80)

    num_epochs = training_config.get('attack_epochs', training_config.get('num_epochs', 5))
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()

        total_loss = 0
        total_psnr = 0
        num_batches = 0

        pbar = tqdm(data_mgr.train_loader, desc=f"Epoch {epoch}", disable=not is_main)
        for batch in pbar:
            loss_dict, updated = finetuner.train_step(batch)
            global_step += 1

            if updated:
                total_loss += loss_dict['loss']
                total_psnr += loss_dict.get('psnr', 0)
                num_batches += 1

                if is_main:
                    pbar.set_postfix({
                        'loss': f"{loss_dict['loss']:.4f}",
                        'lpips': f"{loss_dict.get('loss_lpips', 0):.4f}",
                        'psnr': f"{loss_dict.get('psnr', 0):.2f}",
                    })

            # Debug: 每 20 步打印一次详细信息，前 5 步每步都打印
            if is_main and '_debug' in loss_dict and (global_step <= 5 or global_step % 20 == 0):
                d = loss_dict['_debug']
                print(f"\n[DEBUG step={global_step}] "
                      f"pred_img=[{d['pred_img_min']:.3f}, {d['pred_img_max']:.3f}] mean={d['pred_img_mean']:.3f} | "
                      f"pred_alpha=[{d['pred_alpha_min']:.3f}, {d['pred_alpha_max']:.3f}] mean={d['pred_alpha_mean']:.3f} | "
                      f"gt_img=[{d['gt_img_min']:.3f}, {d['gt_img_max']:.3f}] mean={d['gt_img_mean']:.3f} | "
                      f"gt_mask_mean={d['gt_mask_mean']:.3f}")

            # Debug: 每 50 步保存一次 GT vs Pred 图像
            if is_main and '_debug' in loss_dict and (global_step == 1 or global_step % 50 == 0):
                d = loss_dict['_debug']
                debug_dir = os.path.join(workspace, 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                pred_np = d['pred_images'][0].clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()  # [V, H, W, 3]
                gt_np = d['gt_images_masked'][0].clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
                row_pred = np.concatenate(list(pred_np), axis=1)
                row_gt = np.concatenate(list(gt_np), axis=1)
                grid = np.concatenate([row_gt, row_pred], axis=0)
                from PIL import Image as PILImage
                PILImage.fromarray((grid * 255).astype(np.uint8)).save(
                    os.path.join(debug_dir, f"step{global_step:05d}_gt_vs_pred.png"))

            # 释放 debug tensor 避免显存泄漏
            loss_dict.pop('_debug', None)

        accelerator.wait_for_everyone()

        if is_main:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            avg_psnr = total_psnr / num_batches if num_batches > 0 else 0
            print(f"\nEpoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}")

            # 每个 epoch 渲染对比图
            unwrapped = accelerator.unwrap_model(model)
            evaluator = Evaluator(unwrapped)
            render_epoch_samples(evaluator, data_mgr.val_loader, workspace, epoch=epoch)

            # Source 质量评估 + 渲染
            print(f"\n  评估 source 质量...")
            eval_source_samples(
                unwrapped, evaluator, source_val_loader, workspace,
                epoch=epoch, device='cuda',
            )

        # 保存检查点
        if is_main and (epoch % 5 == 0 or epoch == num_epochs):
            checkpoint_path = os.path.join(workspace, f"checkpoint_epoch_{epoch}.pth")
            unwrapped = accelerator.unwrap_model(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped.state_dict(),
                'optimizer_state_dict': finetuner.optimizer.state_dict(),
                'config': config,
            }, checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")

    if is_main:
        print("\n" + "=" * 80)
        print("全部完成！")
        print(f"渲染结果: {workspace}/renders/")
        print("=" * 80)


if __name__ == '__main__':
    main()
