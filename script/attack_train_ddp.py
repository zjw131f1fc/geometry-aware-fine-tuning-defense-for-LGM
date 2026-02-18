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


def parse_args():
    parser = argparse.ArgumentParser(description='攻击训练 - DDP 版本')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_base_model(model):
    """获取底层模型（处理 LoRA/DDP 包装）"""
    if hasattr(model, 'base_model'):
        model = model.base_model
    if hasattr(model, 'model'):
        model = model.model
    if hasattr(model, 'module'):
        model = model.module
    return model


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

        gaussians = evaluator.generate_gaussians(input_images)

        evaluator.render_and_save(
            gaussians,
            save_dir=render_dir,
            prefix=f"epoch{epoch:02d}_",
            gt_images=gt_images,
        )

        # 第一个样本额外保存 PLY
        if i == 0:
            ply_path = os.path.join(workspace, f"epoch{epoch:02d}.ply")
            evaluator.save_ply(gaussians[0:1], ply_path)


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='no',
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

    # 4. 创建微调器
    training_config = config['training']
    finetuner = AutoFineTuner(
        model=model,
        device='cuda',
        lr=training_config['lr'],
        weight_decay=training_config['weight_decay'],
        gradient_clip=training_config['gradient_clip'],
        mixed_precision=False,
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

    # 7. 训练循环
    if is_main:
        print("\n" + "=" * 80)
        print("开始训练")
        print("=" * 80)

    num_epochs = training_config['num_epochs']

    for epoch in range(1, num_epochs + 1):
        model.train()

        total_loss = 0
        total_psnr = 0
        num_batches = 0

        pbar = tqdm(data_mgr.train_loader, desc=f"Epoch {epoch}", disable=not is_main)
        for batch in pbar:
            loss_dict, updated = finetuner.train_step(batch)

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

        accelerator.wait_for_everyone()

        if is_main:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            avg_psnr = total_psnr / num_batches if num_batches > 0 else 0
            print(f"\nEpoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}")

            # 每个 epoch 渲染对比图
            unwrapped = accelerator.unwrap_model(model)
            evaluator = Evaluator(unwrapped)
            render_epoch_samples(evaluator, data_mgr.val_loader, workspace, epoch=epoch)

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
