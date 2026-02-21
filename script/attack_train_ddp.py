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
import json
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
from methods.trap_losses import PositionCollapseLoss, ScaleAnisotropyLoss, OpacityCollapseLoss


def parse_args():
    parser = argparse.ArgumentParser(description='攻击训练 - DDP 版本')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    return parser.parse_args()


def compute_defense_metrics(model, evaluator, val_loader, device, num_samples=10, trap_combo='position+scale'):
    """
    计算防御相关指标，追踪陷阱的强度变化

    返回：
    - position_static: 位置塌缩指标（越负越强）
    - scale_static: 尺度各向异性指标（越负越强）
    - coupling_value: 乘法耦合强度（越负越强）
    - grad_cosine_sim: 梯度余弦相似度（交替反向模式下应趋向-1，正交模式下应趋向0）
    - gaussian_stats: Gaussian 统计信息
    """
    position_loss_fn = PositionCollapseLoss()
    scale_loss_fn = ScaleAnisotropyLoss()
    opacity_loss_fn = OpacityCollapseLoss()

    # 解析 trap 组合
    trap_names = trap_combo.split('+')
    trap_losses = {}
    if 'position' in trap_names:
        trap_losses['position'] = position_loss_fn
    if 'scale' in trap_names:
        trap_losses['scale'] = scale_loss_fn
    if 'opacity' in trap_names:
        trap_losses['opacity'] = opacity_loss_fn

    all_position_losses = []
    all_scale_losses = []
    all_opacity_losses = []
    all_coupling_values = []
    all_grad_cosine_sims = []
    all_positions = []
    all_scales = []
    all_opacities = []

    model.eval()

    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break

        input_images = batch['input_images'].to(device)

        # 计算 Gaussian（不需要梯度）
        with torch.no_grad():
            gaussians = evaluator.generate_gaussians(input_images)  # [B, N, 14]

            # 计算陷阱损失（作为指标）
            position_loss = position_loss_fn(gaussians).item()
            scale_loss = scale_loss_fn(gaussians).item()
            opacity_loss = opacity_loss_fn(gaussians).item()

            all_position_losses.append(position_loss)
            all_scale_losses.append(scale_loss)
            all_opacity_losses.append(opacity_loss)

            # 收集 Gaussian 参数统计
            all_positions.append(gaussians[..., 0:3].cpu())
            all_scales.append(gaussians[..., 4:7].cpu())
            all_opacities.append(gaussians[..., 3:4].cpu())

        # 计算耦合和梯度冲突（需要梯度）
        model.zero_grad()
        gaussians_grad = model.forward_gaussians(input_images)

        # 计算各 trap loss（保留梯度图）
        static_loss_tensors = {}
        for name, loss_fn in trap_losses.items():
            static_loss_tensors[name] = loss_fn(gaussians_grad)

        # 计算乘法耦合
        if len(static_loss_tensors) > 1:
            product = torch.ones(1, device=device)
            for loss in static_loss_tensors.values():
                product = product * (1 - loss)
            coupling_value = -(product - 1)
            all_coupling_values.append(coupling_value.item())

            # 计算梯度余弦相似度
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if len(trainable_params) > 0:
                # 计算每个 trap 的梯度
                all_grads = {}
                for trap_name, loss in static_loss_tensors.items():
                    grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=trainable_params,
                        create_graph=False,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    # 拼接所有梯度为一个向量
                    grad_vec = []
                    for g in grads:
                        if g is not None:
                            grad_vec.append(g.reshape(-1))
                    if len(grad_vec) > 0:
                        all_grads[trap_name] = torch.cat(grad_vec)

                # 计算两两之间的余弦相似度
                if len(all_grads) >= 2:
                    names = list(all_grads.keys())
                    cos_sims = []
                    for i in range(len(names)):
                        for j in range(i + 1, len(names)):
                            g_i = all_grads[names[i]]
                            g_j = all_grads[names[j]]
                            cos_sim = torch.dot(g_i, g_j) / (g_i.norm() * g_j.norm() + 1e-8)
                            cos_sims.append(cos_sim.item())
                    all_grad_cosine_sims.append(np.mean(cos_sims))

    # 汇总统计
    all_positions = torch.cat(all_positions, dim=0)  # [B*num_samples, N, 3]
    all_scales = torch.cat(all_scales, dim=0)
    all_opacities = torch.cat(all_opacities, dim=0)

    metrics = {
        'position_static': np.mean(all_position_losses),
        'scale_static': np.mean(all_scale_losses),
        'opacity_static': np.mean(all_opacity_losses),
        'coupling_value': np.mean(all_coupling_values) if all_coupling_values else 0.0,
        'grad_cosine_sim': np.mean(all_grad_cosine_sims) if all_grad_cosine_sims else 0.0,
        'gaussian_stats': {
            'position_std': all_positions.std().item(),
            'position_range': (all_positions.min().item(), all_positions.max().item()),
            'scale_mean': all_scales.mean().item(),
            'scale_std': all_scales.std().item(),
            'scale_range': (all_scales.min().item(), all_scales.max().item()),
            'opacity_mean': all_opacities.mean().item(),
            'opacity_std': all_opacities.std().item(),
            'opacity_range': (all_opacities.min().item(), all_opacities.max().item()),
        }
    }

    model.train()
    return metrics


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
        elevations = batch.get('supervision_elevations')
        azimuths = batch.get('supervision_azimuths')
        if elevations is not None:
            elevations = elevations.to('cuda')
        if azimuths is not None:
            azimuths = azimuths.to('cuda')

        gaussians = evaluator.generate_gaussians(input_images)

        evaluator.render_and_save(
            gaussians,
            save_dir=render_dir,
            prefix=f"epoch{epoch:02d}_",
            gt_images=gt_images,
            elevations=elevations,
            azimuths=azimuths,
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
            elevations = batch.get('supervision_elevations')
            azimuths = batch.get('supervision_azimuths')
            if elevations is not None:
                elevations = elevations.to(device)
            if azimuths is not None:
                azimuths = azimuths.to(device)

            gaussians = evaluator.generate_gaussians(input_images)
            evaluator.render_and_save(
                gaussians, save_dir=render_dir,
                prefix=f"epoch{epoch:02d}_source_{i}_",
                gt_images=gt_images, elevations=elevations, azimuths=azimuths,
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

    # 获取 trap 组合（用于指标计算）
    trap_combo = config['defense'].get('trap_combo', 'position+scale')

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

        # 初始化日志文件
        log_file = os.path.join(workspace, 'defense_metrics_log.json')
        metrics_history = []

        # Baseline 防御指标
        print("  计算 baseline 防御指标...")
        defense_metrics = compute_defense_metrics(unwrapped, evaluator, data_mgr.val_loader, device='cuda', trap_combo=trap_combo)
        print(f"  [Baseline Defense Metrics]")
        print(f"    position_static: {defense_metrics['position_static']:.4f}")
        print(f"    scale_static: {defense_metrics['scale_static']:.4f}")
        print(f"    opacity_static: {defense_metrics['opacity_static']:.4f}")
        print(f"    coupling_value: {defense_metrics['coupling_value']:.4f} (越负越强)")
        print(f"    grad_cosine_sim: {defense_metrics['grad_cosine_sim']:.4f} (应趋向-1或0)")
        print(f"  [Baseline Gaussian Stats]")
        stats = defense_metrics['gaussian_stats']
        print(f"    position: std={stats['position_std']:.4f}, range=[{stats['position_range'][0]:.3f}, {stats['position_range'][1]:.3f}]")
        print(f"    scale: mean={stats['scale_mean']:.4f}, std={stats['scale_std']:.4f}")
        print(f"    opacity: mean={stats['opacity_mean']:.4f}, std={stats['opacity_std']:.4f}")

        # 记录 baseline
        metrics_history.append({
            'epoch': 0,
            'defense_metrics': defense_metrics
        })

        render_epoch_samples(evaluator, data_mgr.val_loader, workspace, epoch=0)

        # Baseline source 质量
        print("  评估 baseline source 质量...")
        source_metrics = eval_source_samples(
            unwrapped, evaluator, source_val_loader, workspace,
            epoch=0, device='cuda',
        )
        metrics_history[-1]['source_metrics'] = source_metrics
    else:
        log_file = None
        metrics_history = None

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

            # 计算防御指标
            print(f"\n  计算防御指标...")
            unwrapped = accelerator.unwrap_model(model)
            evaluator = Evaluator(unwrapped)
            defense_metrics = compute_defense_metrics(unwrapped, evaluator, data_mgr.val_loader, device='cuda', trap_combo=trap_combo)

            print(f"  [Defense Metrics]")
            print(f"    position_static: {defense_metrics['position_static']:.4f} (越负越强)")
            print(f"    scale_static: {defense_metrics['scale_static']:.4f} (越负越强)")
            print(f"    opacity_static: {defense_metrics['opacity_static']:.4f} (越负越强)")
            print(f"    coupling_value: {defense_metrics['coupling_value']:.4f} (越负越强)")
            print(f"    grad_cosine_sim: {defense_metrics['grad_cosine_sim']:.4f} (应趋向-1或0)")
            print(f"  [Gaussian Stats]")
            stats = defense_metrics['gaussian_stats']
            print(f"    position: std={stats['position_std']:.4f}, range=[{stats['position_range'][0]:.3f}, {stats['position_range'][1]:.3f}]")
            print(f"    scale: mean={stats['scale_mean']:.4f}, std={stats['scale_std']:.4f}, range=[{stats['scale_range'][0]:.3f}, {stats['scale_range'][1]:.3f}]")
            print(f"    opacity: mean={stats['opacity_mean']:.4f}, std={stats['opacity_std']:.4f}, range=[{stats['opacity_range'][0]:.3f}, {stats['opacity_range'][1]:.3f}]")

            # 记录到历史
            epoch_log = {
                'epoch': epoch,
                'train_loss': avg_loss,
                'train_psnr': avg_psnr,
                'defense_metrics': defense_metrics
            }

            # 每个 epoch 渲染对比图
            render_epoch_samples(evaluator, data_mgr.val_loader, workspace, epoch=epoch)

            # Source 质量评估 + 渲染
            print(f"\n  评估 source 质量...")
            source_metrics = eval_source_samples(
                unwrapped, evaluator, source_val_loader, workspace,
                epoch=epoch, device='cuda',
            )
            epoch_log['source_metrics'] = source_metrics

            # 保存日志
            metrics_history.append(epoch_log)
            with open(log_file, 'w') as f:
                json.dump(metrics_history, f, indent=2)
            print(f"  日志已保存: {log_file}")

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
