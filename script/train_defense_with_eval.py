#!/usr/bin/env python3
"""
GeoTrap 防御训练 + 攻击评估混合脚本

流程：
1. 保存原始参数 → 基线攻击测试（defense 前攻击能达到什么水平）
2. 防御训练循环：
   - 每 eval_every 步暂停防御
   - 保存防御参数 → 攻击测试 → 恢复防御参数 → 继续防御
3. 最终攻击测试

攻击测试 = 用当前模型状态跑 N 个 epoch 的攻击微调，测量 PSNR/LPIPS/Loss
"""

import os
import sys
import argparse

# 解析 GPU 参数（必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES）
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=int, default=0)
_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)

os.environ['XFORMERS_DISABLED'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
from datetime import datetime

from project_core import ConfigManager
from data import DataManager
from training import DefenseTrainer, AutoFineTuner
from evaluation import Evaluator
from tools import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='GeoTrap 防御训练 + 攻击评估')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认自动生成）')
    parser.add_argument('--target_layers', type=str, default=None,
                        help='敏感层（逗号分隔，覆盖配置文件）')
    return parser.parse_args()


@torch.no_grad()
def render_attack_samples(evaluator, val_loader, save_dir, num_samples=3):
    """攻击评估后渲染对比图（GT vs Pred）"""
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break
        input_images = batch['input_images'].to('cuda')
        gt_images = batch.get('supervision_images')
        if gt_images is not None:
            gt_images = gt_images.to('cuda')
        elevations = batch.get('supervision_elevations')
        azimuths = batch.get('supervision_azimuths')
        if elevations is not None:
            elevations = elevations.to('cuda')
        if azimuths is not None:
            azimuths = azimuths.to('cuda')

        gaussians = evaluator.generate_gaussians(input_images)
        evaluator.render_and_save(
            gaussians, save_dir=save_dir,
            gt_images=gt_images, elevations=elevations, azimuths=azimuths,
        )
        if i == 0:
            evaluator.save_ply(gaussians[0:1], os.path.join(save_dir, 'sample.ply'))


@torch.no_grad()
def run_source_eval(model, source_val_loader, device, baseline_source=None):
    """在 source 验证集上评估渲染质量（PSNR/LPIPS），检查防御是否导致退化"""
    model.eval()

    # 创建临时 finetuner 用于 _prepare_data（需要 opt 等信息）
    finetuner = AutoFineTuner(
        model=model, device=device,
        lr=1e-4, weight_decay=0, gradient_clip=1.0,
        mixed_precision='no', gradient_accumulation_steps=1,
    )

    total_psnr = 0
    total_lpips = 0
    total_masked_psnr = 0
    num_batches = 0

    for batch in source_val_loader:
        data = finetuner._prepare_data(batch)
        results = model.forward(data, step_ratio=1.0)
        total_psnr += results.get('psnr', torch.tensor(0.0)).item()
        total_lpips += results.get('loss_lpips', torch.tensor(0.0)).item()

        # Masked PSNR
        pred_images = results.get('images_pred')
        gt_masks = data['masks_output']
        gt_images = data['images_output']
        if pred_images is not None and gt_masks is not None:
            mask_flat = gt_masks.reshape(-1)
            pred_flat = pred_images.reshape(-1, 3)
            gt_flat = gt_images.reshape(-1, 3)
            mask_sum = mask_flat.sum().clamp(min=1.0)
            masked_mse = ((pred_flat - gt_flat) ** 2 * mask_flat.unsqueeze(-1)).sum() / (mask_sum * 3)
            total_masked_psnr += (-10 * torch.log10(masked_mse + 1e-8)).item()

        num_batches += 1

    del finetuner

    metrics = {
        'source_psnr': total_psnr / max(num_batches, 1),
        'source_lpips': total_lpips / max(num_batches, 1),
        'source_masked_psnr': total_masked_psnr / max(num_batches, 1),
    }

    msg = f"  Source 质量: PSNR={metrics['source_psnr']:.2f}, LPIPS={metrics['source_lpips']:.4f}"
    if baseline_source is not None:
        dp = metrics['source_psnr'] - baseline_source['source_psnr']
        dl = metrics['source_lpips'] - baseline_source['source_lpips']
        msg += f" (ΔPSNR={dp:+.2f}, ΔLPIPS={dl:+.4f})"
    print(msg)

    return metrics


@torch.no_grad()
def _eval_source_quality(model, evaluator, source_val_loader, device,
                         save_dir, num_render=3):
    """
    在 source 验证集上计算 PSNR/LPIPS 并渲染对比图

    Args:
        model: 当前模型
        evaluator: Evaluator 实例
        source_val_loader: source 验证数据加载器
        device: 设备
        save_dir: 渲染图片保存目录
        num_render: 渲染样本数

    Returns:
        metrics: {'psnr': float, 'lpips': float}
    """
    model.eval()

    finetuner = AutoFineTuner(
        model=model, device=device,
        lr=1e-4, weight_decay=0, gradient_clip=1.0,
        mixed_precision='no', gradient_accumulation_steps=1,
    )

    total_psnr = 0
    total_lpips = 0
    total_masked_psnr = 0
    num_batches = 0

    for i, batch in enumerate(source_val_loader):
        data = finetuner._prepare_data(batch)
        results = model.forward(data, step_ratio=1.0)
        total_psnr += results.get('psnr', torch.tensor(0.0)).item()
        total_lpips += results.get('loss_lpips', torch.tensor(0.0)).item()

        # Masked PSNR
        pred_images = results.get('images_pred')
        gt_masks = data['masks_output']
        gt_images_out = data['images_output']
        if pred_images is not None and gt_masks is not None:
            mask_flat = gt_masks.reshape(-1)
            pred_flat = pred_images.reshape(-1, 3)
            gt_flat = gt_images_out.reshape(-1, 3)
            mask_sum = mask_flat.sum().clamp(min=1.0)
            masked_mse = ((pred_flat - gt_flat) ** 2 * mask_flat.unsqueeze(-1)).sum() / (mask_sum * 3)
            total_masked_psnr += (-10 * torch.log10(masked_mse + 1e-8)).item()

        num_batches += 1

        # 渲染前 num_render 个样本的对比图
        if i < num_render:
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
                gaussians, save_dir=save_dir,
                prefix=f"source_{i}_",
                gt_images=gt_images, elevations=elevations, azimuths=azimuths,
            )

    del finetuner

    return {
        'psnr': total_psnr / max(num_batches, 1),
        'lpips': total_lpips / max(num_batches, 1),
        'masked_psnr': total_masked_psnr / max(num_batches, 1),
    }


def run_attack_eval(model, config, opt, attack_train_loader, attack_val_loader,
                    device, attack_epochs, save_dir, eval_tag,
                    baseline_metrics=None, source_val_loader=None):
    """
    在当前模型状态上跑攻击微调，测量攻击成功度。

    流程：保存模型 → 攻击训练 N epochs → 测量指标 + 渲染 → 恢复模型
    返回：metrics dict（包含每 epoch 的 loss/psnr/lpips + 最终 Gaussian stats + source 质量）
    """
    print(f"\n{'='*60}")
    print(f"  攻击评估: {eval_tag}")
    print(f"  攻击 epochs: {attack_epochs}")
    print(f"{'='*60}")

    # 保存当前模型状态到 CPU（节省 GPU 显存）
    saved_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    # 保存 requires_grad 状态（防御训练冻结了大部分层，攻击需要解冻）
    saved_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}

    # 攻击时解冻所有参数（模拟全量微调攻击）
    for p in model.parameters():
        p.requires_grad = True

    # 创建攻击微调器（独立优化器）
    training_config = config['training']
    attack_lr = config['defense'].get('eval', {}).get('attack_lr', training_config['lr'])
    finetuner = AutoFineTuner(
        model=model,
        device=device,
        lr=attack_lr,
        weight_decay=training_config['weight_decay'],
        gradient_clip=training_config['gradient_clip'],
        mixed_precision=training_config.get('mixed_precision', 'bf16'),
        lambda_lpips=training_config.get('lambda_lpips', 1.0),
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        optimizer_type=training_config.get('optimizer', 'adamw'),
        optimizer_betas=training_config.get('optimizer_betas', [0.9, 0.95]),
        optimizer_momentum=training_config.get('optimizer_momentum', 0.9),
    )

    # 攻击训练（每个 epoch 渲染对比图）
    evaluator = Evaluator(model, device=device)
    os.makedirs(save_dir, exist_ok=True)
    num_render = config['defense'].get('eval', {}).get('num_render_samples', 3)

    epoch_metrics = []
    for epoch in range(1, attack_epochs + 1):
        metrics = finetuner.train_epoch(attack_train_loader, epoch)
        epoch_metrics.append(metrics)
        print(f"  [Attack {eval_tag}] Epoch {epoch}/{attack_epochs} - "
              f"Loss: {metrics['loss']:.4f}, PSNR: {metrics['psnr']:.2f}, "
              f"LPIPS: {metrics.get('loss_lpips', 0):.4f}")

        # 每个 epoch 渲染对比图
        render_dir = os.path.join(save_dir, f'renders_epoch{epoch}')
        render_attack_samples(evaluator, attack_val_loader, render_dir, num_samples=num_render)

    # 在 val 数据上计算 Gaussian 统计
    gaussian_stats_list = []
    for i, batch in enumerate(attack_val_loader):
        if i >= num_render:
            break
        input_images = batch['input_images'].to(device)
        gaussians = evaluator.generate_gaussians(input_images)
        stats = evaluator.compute_gaussian_stats(gaussians)
        gaussian_stats_list.append(stats)

    # 汇总结果
    final_metrics = epoch_metrics[-1] if epoch_metrics else {}
    result = {
        'eval_tag': eval_tag,
        'attack_epochs': attack_epochs,
        'final_loss': final_metrics.get('loss', 0),
        'final_psnr': final_metrics.get('psnr', 0),
        'final_lpips': final_metrics.get('loss_lpips', 0),
        'epoch_history': epoch_metrics,
    }
    if gaussian_stats_list:
        result['gaussian_stats'] = {
            'opacity_mean': np.mean([s['opacity_mean'] for s in gaussian_stats_list]),
            'scale_mean': np.mean([s['scale_mean'] for s in gaussian_stats_list]),
            'rgb_mean': np.mean([s['rgb_mean'] for s in gaussian_stats_list]),
        }

    # 攻击后评估 source 质量：PSNR/LPIPS + 渲染图片
    if source_val_loader is not None:
        print(f"  [Source Eval] 攻击后 source 质量评估...")
        source_metrics = _eval_source_quality(
            model, evaluator, source_val_loader, device,
            save_dir=os.path.join(save_dir, 'source_renders'),
            num_render=num_render,
        )
        result['source_after_attack_psnr'] = source_metrics['psnr']
        result['source_after_attack_lpips'] = source_metrics['lpips']
        print(f"  [Source Eval] PSNR={source_metrics['psnr']:.2f}, "
              f"LPIPS={source_metrics['lpips']:.4f}")

    # 保存指标到 JSON
    json_path = os.path.join(save_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump({k: v for k, v in result.items()
                   if k != 'epoch_history'}, f, indent=2, default=str)

    print(f"  结果: PSNR={result['final_psnr']:.2f}, "
          f"LPIPS={result['final_lpips']:.4f}, "
          f"Loss={result['final_loss']:.4f}")
    if baseline_metrics is not None:
        dp = result['final_psnr'] - baseline_metrics['final_psnr']
        dl = result['final_lpips'] - baseline_metrics['final_lpips']
        print(f"  vs baseline: ΔPSNR={dp:+.2f}, ΔLPIPS={dl:+.4f}")
    print(f"  渲染保存: {render_dir}")

    # 恢复模型状态 + requires_grad 状态
    # strict=False: LPIPS 网络权重是懒加载的，saved_state 可能不含 lpips_loss 的 key
    model.load_state_dict(saved_state, strict=False)
    for name, p in model.named_parameters():
        p.requires_grad = saved_requires_grad.get(name, False)
    del saved_state, saved_requires_grad
    torch.cuda.empty_cache()

    return result


def main():
    args = parse_args()
    device = 'cuda'

    # 加载配置
    config = ConfigManager(args.config).config
    set_seed(config['misc']['seed'])

    eval_config = config['defense'].get('eval', {})
    eval_every = eval_config.get('eval_every_steps', 50)
    attack_epochs = eval_config.get('attack_epochs', 3)

    # 工作目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    workspace = args.output_dir or os.path.join(
        config['misc']['workspace'], f"defense_eval_{timestamp}")
    os.makedirs(workspace, exist_ok=True)

    print("=" * 80)
    print("GeoTrap 防御训练 + 攻击评估")
    print("=" * 80)
    print(f"工作目录: {workspace}")
    print(f"攻击评估间隔: 每 {eval_every} 步")
    print(f"攻击评估 epochs: {attack_epochs}")

    # 解析敏感层
    target_layers = None
    if args.target_layers:
        target_layers = [l.strip() for l in args.target_layers.split(',')]
    elif 'target_layers' in config.get('defense', {}):
        target_layers = config['defense']['target_layers']

    # ========== 1. 创建防御训练器 ==========
    trainer = DefenseTrainer(config)
    trainer.setup(device=device, target_layers=target_layers)
    model = trainer.model_mgr.model

    # ========== 2. 创建攻击评估用数据加载器 ==========
    # 攻击用原始 target 配置（offset=0），与防御用的 target（offset=15）不重叠
    print("\n创建攻击评估数据加载器...")
    attack_data_mgr = DataManager(config, trainer.model_mgr.opt)
    attack_data_mgr.setup_dataloaders(train=True, val=True, subset='target')
    attack_train_loader = attack_data_mgr.train_loader
    attack_val_loader = attack_data_mgr.val_loader
    print(f"  攻击训练数据: {len(attack_train_loader.dataset)} 样本")
    print(f"  攻击验证数据: {len(attack_val_loader.dataset)} 样本")

    # 收集所有评估结果
    all_eval_results = []

    # ========== 3. 基线评估（防御前） ==========
    # 3a. 基线 source 质量
    print("\n--- 基线 Source 质量评估 ---")
    baseline_source = run_source_eval(model, trainer.source_val_loader, device)

    # 3b. 基线攻击评估
    baseline_result = run_attack_eval(
        model, config, trainer.model_mgr.opt,
        attack_train_loader, attack_val_loader,
        device, attack_epochs,
        save_dir=os.path.join(workspace, 'eval_baseline'),
        eval_tag='baseline (defense_step=0)',
        source_val_loader=trainer.source_val_loader,
    )
    baseline_result['source_psnr'] = baseline_source['source_psnr']
    baseline_result['source_lpips'] = baseline_source['source_lpips']
    all_eval_results.append(('baseline', 0, baseline_result))

    # ========== 4. 防御训练 + 周期性攻击评估 ==========
    num_epochs = config['training'].get('defense_epochs', config['training'].get('num_epochs', 10))

    # 构建步回调：每 eval_every 步触发攻击评估 + source 质量检查
    def eval_step_callback(global_step, loss_dict):
        if global_step % eval_every == 0:
            # Source 质量检查
            source_metrics = run_source_eval(
                model, trainer.source_val_loader, device, baseline_source)

            # 攻击评估
            eval_result = run_attack_eval(
                model, config, trainer.model_mgr.opt,
                attack_train_loader, attack_val_loader,
                device, attack_epochs,
                save_dir=os.path.join(workspace, f'eval_step{global_step:05d}'),
                eval_tag=f'defense_step={global_step}',
                baseline_metrics=baseline_result,
                source_val_loader=trainer.source_val_loader,
            )
            eval_result['source_psnr'] = source_metrics['source_psnr']
            eval_result['source_lpips'] = source_metrics['source_lpips']
            all_eval_results.append((f'step_{global_step}', global_step, eval_result))

    print(f"\n{'='*80}")
    print(f"开始防御训练: {num_epochs} epochs, 每 {eval_every} 步评估一次")
    print(f"{'='*80}")

    trainer.train(
        num_epochs=num_epochs,
        save_dir=workspace,
        validate_every=1,
        step_callback=eval_step_callback,
    )

    # ========== 5. 最终评估 ==========
    final_step = all_eval_results[-1][1] if all_eval_results else 0

    # Source 质量
    final_source = run_source_eval(model, trainer.source_val_loader, device, baseline_source)

    # 攻击评估
    final_result = run_attack_eval(
        model, config, trainer.model_mgr.opt,
        attack_train_loader, attack_val_loader,
        device, attack_epochs,
        save_dir=os.path.join(workspace, 'eval_final'),
        eval_tag='final',
        baseline_metrics=baseline_result,
        source_val_loader=trainer.source_val_loader,
    )
    final_result['source_psnr'] = final_source['source_psnr']
    final_result['source_lpips'] = final_source['source_lpips']
    final_step = final_step + 1  # just a tag, doesn't matter
    all_eval_results.append(('final', final_step, final_result))

    # ========== 6. 汇总报告 ==========
    print(f"\n{'='*80}")
    print("评估汇总")
    print(f"{'='*80}")
    baseline_psnr = all_eval_results[0][2]['final_psnr']
    baseline_lpips = all_eval_results[0][2]['final_lpips']
    baseline_src_psnr = all_eval_results[0][2].get('source_psnr', 0)
    baseline_src_atk_psnr = all_eval_results[0][2].get('source_after_attack_psnr', 0)
    print(f"{'阶段':<25} {'Step':>6} {'Atk PSNR':>9} {'ΔAtk':>7} {'Atk LPIPS':>10} "
          f"{'Src PSNR':>9} {'ΔSrc':>7} {'SrcAtk PSNR':>11}")
    print("-" * 105)
    for tag, step, result in all_eval_results:
        dp = result['final_psnr'] - baseline_psnr
        src_psnr = result.get('source_psnr', 0)
        ds = src_psnr - baseline_src_psnr
        src_atk_psnr = result.get('source_after_attack_psnr', 0)
        print(f"{tag:<25} {step:>6d} {result['final_psnr']:>9.2f} {dp:>+7.2f} "
              f"{result['final_lpips']:>10.4f} {src_psnr:>9.2f} {ds:>+7.2f} "
              f"{src_atk_psnr:>11.2f}")
    print("-" * 105)

    # 攻击 PSNR 下降 = 防御有效；Source PSNR 下降 = 退化
    if len(all_eval_results) >= 2:
        final_r = all_eval_results[-1][2]
        atk_delta = final_r['final_psnr'] - baseline_psnr
        src_delta = final_r.get('source_psnr', 0) - baseline_src_psnr
        print(f"\n攻击 PSNR: {baseline_psnr:.2f} → {final_r['final_psnr']:.2f} (Δ={atk_delta:+.2f})")
        print(f"Source PSNR: {baseline_src_psnr:.2f} → {final_r.get('source_psnr', 0):.2f} (Δ={src_delta:+.2f})")
        if atk_delta < -1.0 and src_delta > -1.0:
            print("→ 防御有效：攻击 PSNR 下降，Source 质量保持")
        elif atk_delta < -1.0:
            print("→ 防御有效但有退化：攻击 PSNR 下降，Source 质量也下降")
        elif atk_delta < 0:
            print("→ 防御有一定效果")
        else:
            print("→ 防御无效：攻击 PSNR 未下降")

    # 保存完整结果
    summary_path = os.path.join(workspace, 'eval_summary.json')
    summary = []
    for tag, step, result in all_eval_results:
        summary.append({
            'tag': tag,
            'defense_step': step,
            'attack_psnr': result['final_psnr'],
            'attack_lpips': result['final_lpips'],
            'attack_loss': result['final_loss'],
            'delta_psnr': result['final_psnr'] - baseline_psnr,
            'delta_lpips': result['final_lpips'] - baseline_lpips,
            'source_psnr': result.get('source_psnr', 0),
            'source_lpips': result.get('source_lpips', 0),
            'delta_source_psnr': result.get('source_psnr', 0) - baseline_src_psnr,
            'source_after_attack_psnr': result.get('source_after_attack_psnr', 0),
            'source_after_attack_lpips': result.get('source_after_attack_lpips', 0),
        })
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n完整结果: {summary_path}")
    print(f"工作目录: {workspace}")


if __name__ == '__main__':
    main()
