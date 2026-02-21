#!/usr/bin/env python3
"""
端到端 Pipeline：Baseline Attack → Defense Training → Post-Defense Attack

在单 GPU 上依次运行三个阶段，每隔 N 个 step 收集指标，最后绘制对比图（横轴为 step）。

用法：
    python script/run_pipeline.py --gpu 0 --config configs/config.yaml \
        --trap_losses position,scale --tag geotrap_v1 \
        --attack_epochs 5 --defense_epochs 25 --eval_every_steps 10
"""

import os
import sys
import argparse

# 解析 GPU 参数（必须在 import torch 之前）
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=int, default=0)
_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
os.environ['XFORMERS_DISABLED'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import copy
# import hashlib  # 延迟加载：只在 compute_config_hash() 中使用
# import shutil  # 延迟加载：只在 save_defense_model() 中使用
import torch
import numpy as np
from datetime import datetime

from project_core import ConfigManager
from models import ModelManager
from data import DataManager
from training import DefenseTrainer, AutoFineTuner
from evaluation import Evaluator
# from tools import set_seed  # 延迟加载：只在 main() 中使用一次


def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline: Attack → Defense → Attack')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--trap_losses', type=str, default=None,
                        help='启用的 trap loss 组合，逗号分隔（如 position,scale）')
    parser.add_argument('--trap_combo', type=str, default=None,
                        help='层选择用的 trap 组合（如 position+scale），默认从 --trap_losses 推断')
    parser.add_argument('--num_target_layers', type=int, default=None,
                        help='自动选择的防御层数（覆盖 config）')
    parser.add_argument('--tag', type=str, default=None,
                        help='防御模型标签名（覆盖 config 中的 defense.tag）')
    parser.add_argument('--attack_epochs', type=int, default=None,
                        help='攻击训练 epoch 数')
    parser.add_argument('--defense_epochs', type=int, default=None,
                        help='防御训练 epoch 数')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认自动生成）')
    parser.add_argument('--num_render', type=int, default=3,
                        help='每阶段渲染样本数')
    parser.add_argument('--eval_every_steps', type=int, default=10,
                        help='每隔多少 step 评估一次指标')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_source(model, source_val_loader, device):
    """评估 source 质量，返回 {psnr, lpips, masked_psnr}。

    LoRA 模式下自动禁用 adapter，只测底座模型的 source 能力。
    """
    model.eval()

    # 先创建 finetuner（需要可训练参数来初始化 optimizer）
    finetuner = AutoFineTuner(
        model=model, device=device,
        lr=1e-4, weight_decay=0, gradient_clip=1.0,
        mixed_precision='no', gradient_accumulation_steps=1,
    )

    # LoRA 模式：创建完 finetuner 后再禁用 adapter，只测底座模型
    has_lora = hasattr(model, 'disable_adapter_layers')
    if has_lora:
        model.disable_adapter_layers()

    total_psnr, total_lpips, total_masked_psnr, n = 0, 0, 0, 0
    for batch in source_val_loader:
        data = finetuner._prepare_data(batch)
        results = model.forward(data, step_ratio=1.0)
        total_psnr += results.get('psnr', torch.tensor(0.0)).item()
        total_lpips += results.get('loss_lpips', torch.tensor(0.0)).item()

        # Masked PSNR：只在物体区域计算
        pred_images = results.get('images_pred')
        gt_images = data['images_output']
        gt_masks = data['masks_output']
        if pred_images is not None and gt_masks is not None:
            mask_flat = gt_masks.reshape(-1)
            pred_flat = pred_images.reshape(-1, 3)
            gt_flat = gt_images.reshape(-1, 3)
            mask_sum = mask_flat.sum().clamp(min=1.0)
            masked_mse = ((pred_flat - gt_flat) ** 2 * mask_flat.unsqueeze(-1)).sum() / (mask_sum * 3)
            masked_psnr = -10 * torch.log10(masked_mse + 1e-8)
            total_masked_psnr += masked_psnr.item()

        n += 1
    del finetuner

    # 恢复 LoRA adapter
    if has_lora:
        model.enable_adapter_layers()

    return {
        'source_psnr': total_psnr / max(n, 1),
        'source_lpips': total_lpips / max(n, 1),
        'source_masked_psnr': total_masked_psnr / max(n, 1),
    }


@torch.no_grad()
def render_samples(model, evaluator, loader, save_dir, prefix='', num_samples=3):
    """渲染 GT vs Pred 对比图"""
    model.eval()
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
        input_images = batch['input_images'].to('cuda')

        # 优先使用 transforms（归一化后的相机姿态）
        input_transforms = batch.get('input_transforms')
        supervision_transforms = batch.get('supervision_transforms')

        if input_transforms is not None and supervision_transforms is not None:
            # 合并输入和监督视图的 transforms（8个视角）
            all_transforms = torch.cat([
                input_transforms.to('cuda'),
                supervision_transforms.to('cuda')
            ], dim=1)  # [B, 8, 4, 4]

            # 提取输入视图的 GT（从 input_images 前3通道反归一化）
            input_rgb = input_images[:, :, :3, :, :]  # [B, 4, 3, H, W]
            IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 1, 3, 1, 1)
            IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 1, 3, 1, 1)
            input_rgb = input_rgb * IMAGENET_STD + IMAGENET_MEAN

            # 合并输入 GT 和监督 GT
            supervision_images = batch.get('supervision_images')
            if supervision_images is not None:
                supervision_images = supervision_images.to('cuda')
                all_gt_images = torch.cat([input_rgb, supervision_images], dim=1)  # [B, 8, 3, H, W]
            else:
                all_gt_images = input_rgb

            gaussians = evaluator.generate_gaussians(input_images)
            evaluator.render_and_save(
                gaussians, save_dir=save_dir, prefix=f"{prefix}{i}_",
                gt_images=all_gt_images, transforms=all_transforms,
            )
        else:
            # 降级方案：使用 elevations/azimuths
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
                gaussians, save_dir=save_dir, prefix=f"{prefix}{i}_",
                gt_images=gt_images, elevations=elevations, azimuths=azimuths,
            )


def run_attack_phase(config, phase_name, target_train_loader, target_val_loader,
                     source_val_loader, save_dir, attack_epochs, num_render,
                     eval_every_steps=10, model_resume_override=None):
    """
    运行一个攻击阶段，返回 per-step 指标列表。

    每隔 eval_every_steps 个 step 收集一次指标（训练区间平均 + source 评估）。

    Returns:
        step_history: [{step, epoch, loss, psnr, lpips, source_psnr, ...}, ...]
    """
    print(f"\n{'='*80}")
    print(f"  {phase_name}")
    print(f"  Attack epochs: {attack_epochs}, eval_every_steps: {eval_every_steps}")
    print(f"{'='*80}")

    # 加载模型
    attack_config = copy.deepcopy(config)
    if model_resume_override:
        attack_config['model']['resume'] = model_resume_override
    model_mgr = ModelManager(attack_config)
    model_mgr.setup(device='cuda')
    model = model_mgr.model

    training_cfg = config['training']
    finetuner = AutoFineTuner(
        model=model, device='cuda',
        lr=training_cfg['lr'],
        weight_decay=training_cfg['weight_decay'],
        gradient_clip=training_cfg['gradient_clip'],
        mixed_precision='no',
        lambda_lpips=training_cfg.get('lambda_lpips', 1.0),
        gradient_accumulation_steps=training_cfg['gradient_accumulation_steps'],
    )

    evaluator = Evaluator(model, device='cuda')
    os.makedirs(save_dir, exist_ok=True)

    # 攻击前渲染 source（展示模型原始 source 能力）
    render_samples(model, evaluator, source_val_loader,
                   os.path.join(save_dir, 'source_renders'),
                   prefix='source_', num_samples=num_render)

    step_history = []
    global_step = 0
    interval_loss, interval_lpips, interval_psnr, interval_masked_psnr = 0, 0, 0, 0
    interval_count = 0
    total_steps = attack_epochs * len(target_train_loader)

    for epoch in range(1, attack_epochs + 1):
        model.train()
        for batch in target_train_loader:
            loss_dict, updated = finetuner.train_step(batch)
            global_step += 1

            interval_loss += loss_dict['loss']
            interval_lpips += loss_dict.get('loss_lpips', 0)
            interval_psnr += loss_dict.get('psnr', 0)
            interval_masked_psnr += loss_dict.get('masked_psnr', 0)
            interval_count += 1

            if global_step % eval_every_steps == 0 or global_step == total_steps:
                metrics = {
                    'step': global_step,
                    'epoch': epoch,
                    'loss': interval_loss / interval_count,
                    'loss_lpips': interval_lpips / interval_count,
                    'psnr': interval_psnr / interval_count,
                    'masked_psnr': interval_masked_psnr / interval_count,
                }
                src = eval_source(model, source_val_loader, 'cuda')
                metrics.update(src)
                step_history.append(metrics)

                print(f"  [{phase_name}] Step {global_step}/{total_steps} (Ep{epoch}) - "
                      f"Loss: {metrics['loss']:.4f}, LPIPS: {metrics['loss_lpips']:.4f}, "
                      f"MaskedPSNR: {metrics['masked_psnr']:.2f}, "
                      f"SrcPSNR: {src['source_psnr']:.2f}")

                interval_loss, interval_lpips, interval_psnr, interval_masked_psnr = 0, 0, 0, 0
                interval_count = 0

        # epoch 结束，flush 残余梯度
        if finetuner._accumulation_counter % finetuner.gradient_accumulation_steps != 0:
            if finetuner.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), finetuner.gradient_clip)
            finetuner.optimizer.step()
            finetuner.optimizer.zero_grad()

    # 最后渲染 target
    render_samples(model, evaluator, target_val_loader,
                   os.path.join(save_dir, 'target_renders'),
                   prefix='target_', num_samples=num_render)

    del finetuner, evaluator, model, model_mgr
    torch.cuda.empty_cache()

    return step_history


def run_defense_phase(config, save_dir, defense_epochs, target_layers):
    """
    运行防御训练阶段，返回 per-epoch 指标列表。

    Returns:
        epoch_history: [{train_metrics..., val_metrics...}, ...]
    """
    print(f"\n{'='*80}")
    print(f"  Defense Training")
    print(f"  Defense epochs: {defense_epochs}")
    print(f"{'='*80}")

    trainer = DefenseTrainer(config)
    trainer.setup(device='cuda', target_layers=target_layers)

    epoch_history = []
    global_step = 0

    validate_every = 5
    for epoch in range(1, defense_epochs + 1):
        train_metrics, global_step = trainer.train_epoch(epoch, global_step)

        combined = {f"train_{k}": v for k, v in train_metrics.items()}
        combined['epoch'] = epoch

        do_val = (epoch % validate_every == 0) or (epoch == defense_epochs)
        if do_val:
            val_metrics = trainer.validate()
            combined.update({f"val_{k}": v for k, v in val_metrics.items()})

            print(f"  [Defense] Epoch {epoch}/{defense_epochs} - "
                  f"Loss: {train_metrics['loss']:.4f}, "
                  f"DistillMSE: {val_metrics.get('source_distill_mse', 0):.6f}")
            for k in ('position_static', 'scale_static', 'opacity_static',
                      'coupling_value', 'grad_cosine_sim'):
                if k in val_metrics:
                    print(f"    {k}: {val_metrics[k]:.4f}")
        else:
            print(f"  [Defense] Epoch {epoch}/{defense_epochs} - "
                  f"Loss: {train_metrics['loss']:.4f}")

        epoch_history.append(combined)

        if epoch % 5 == 0 or epoch == defense_epochs:
            trainer.save_checkpoint(save_dir, epoch)

    del trainer
    torch.cuda.empty_cache()

    return epoch_history


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_results(baseline_history, postdef_history, defense_history, save_path):
    """绘制 2×2 对比图，攻击阶段横轴为 step"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps_b = [m['step'] for m in baseline_history]
    steps_p = [m['step'] for m in postdef_history]
    epochs_d = list(range(1, len(defense_history) + 1))

    # (0,0) Target LPIPS
    ax = axes[0, 0]
    ax.plot(steps_b, [m.get('loss_lpips', 0) for m in baseline_history],
            'b-o', label='Baseline Attack', markersize=3)
    ax.plot(steps_p, [m.get('loss_lpips', 0) for m in postdef_history],
            'r-s', label='Post-Defense Attack', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('LPIPS')
    ax.set_title('Target LPIPS (↑ = defense effective)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Target Masked PSNR
    ax = axes[0, 1]
    ax.plot(steps_b, [m.get('masked_psnr', 0) for m in baseline_history],
            'b-o', label='Baseline Attack', markersize=3)
    ax.plot(steps_p, [m.get('masked_psnr', 0) for m in postdef_history],
            'r-s', label='Post-Defense Attack', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Masked PSNR (dB)')
    ax.set_title('Target Masked PSNR (↓ = defense effective)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Source PSNR
    ax = axes[1, 0]
    ax.plot(steps_b, [m.get('source_psnr', 0) for m in baseline_history],
            'b-o', label='Baseline Attack', markersize=3)
    ax.plot(steps_p, [m.get('source_psnr', 0) for m in postdef_history],
            'r-s', label='Post-Defense Attack', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Source PSNR (should stay similar)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Defense Training Metrics
    ax = axes[1, 1]
    trap_keys = [k for k in defense_history[0] if k.startswith('val_') and 'static' in k]
    for k in trap_keys:
        label = k.replace('val_', '')
        ax.plot(epochs_d, [m.get(k, 0) for m in defense_history],
                '-o', label=label, markersize=3)
    ax.set_xlabel('Defense Epoch')
    ax.set_ylabel('Trap Loss (log scale)')
    ax.set_title('Defense Training')
    ax2 = ax.twinx()
    distill_vals = [m.get('val_source_distill_mse', 0) for m in defense_history]
    ax2.plot(epochs_d, distill_vals, 'k--', label='distill_mse', alpha=0.7)
    ax2.set_ylabel('Distill MSE')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Pipeline] 对比图已保存: {save_path}")


# ---------------------------------------------------------------------------
# Baseline 缓存
# ---------------------------------------------------------------------------

BASELINE_CACHE_DIR = 'output/baseline_cache'


def compute_baseline_hash(config, attack_epochs, num_render):
    """根据影响 baseline 结果的所有配置计算 SHA256 哈希。"""
    import hashlib  # 延迟加载

    key_parts = {
        'model_resume': config['model']['resume'],
        'model_size': config['model']['size'],
        'lora': config.get('lora', {}),
        'training': {
            'lr': config['training']['lr'],
            'weight_decay': config['training']['weight_decay'],
            'gradient_clip': config['training']['gradient_clip'],
            'gradient_accumulation_steps': config['training']['gradient_accumulation_steps'],
            'lambda_lpips': config['training'].get('lambda_lpips', 1.0),
        },
        'attack_epochs': attack_epochs,
        'data_target': config['data']['target'],
        'data_shared': {
            'root': config['data']['root'],
            'view_selector': config['data'].get('view_selector'),
            'angle_offset': config['data'].get('angle_offset'),
            'num_supervision_views': config['data'].get('num_supervision_views'),
            'samples_per_object': config['data'].get('samples_per_object'),
        },
        'data_source': config['data'].get('source', {}),
        'seed': config['misc']['seed'],
        'num_render': num_render,
    }
    raw = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_baseline_cache(cache_dir):
    """尝试加载缓存的 baseline 结果。返回 (history, True) 或 (None, False)。"""
    meta_path = os.path.join(cache_dir, 'baseline_meta.json')
    if not os.path.exists(meta_path):
        return None, False
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    print(f"[Cache] 命中 baseline 缓存: {cache_dir}")
    return meta['baseline_history'], True


def save_baseline_cache(cache_dir, history):
    """保存 baseline 结果到缓存目录。"""
    os.makedirs(cache_dir, exist_ok=True)
    meta = {'baseline_history': history}
    with open(os.path.join(cache_dir, 'baseline_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[Cache] baseline 结果已缓存: {cache_dir}")


def copy_cached_renders(cache_dir, dest_dir):
    """从缓存复制渲染图片到当前 pipeline 工作目录。"""
    import shutil  # 延迟加载

    for sub in ('source_renders', 'target_renders'):
        src = os.path.join(cache_dir, sub)
        dst = os.path.join(dest_dir, sub)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from tools import set_seed  # 延迟加载

    args = parse_args()
    device = 'cuda'

    # 加载配置
    config = ConfigManager(args.config).config
    set_seed(config['misc']['seed'])

    # CLI 覆盖
    if args.attack_epochs is not None:
        config['training']['attack_epochs'] = args.attack_epochs
    if args.defense_epochs is not None:
        config['training']['defense_epochs'] = args.defense_epochs

    tag = args.tag or config['defense'].get('tag', 'pipeline_default')
    config['defense']['tag'] = tag

    # 覆盖 trap_losses 开关 + 自动更新 trap_combo
    if args.trap_losses:
        enabled = sorted(args.trap_losses.split(','))
        for loss_name in ('position', 'scale', 'opacity', 'rotation'):
            config['defense']['trap_losses'][loss_name]['static'] = (loss_name in enabled)
        print(f"[Pipeline] Trap losses 覆盖: {enabled}")

        # 自动推断 trap_combo（如果 CLI 没有显式指定）
        if args.trap_combo is None and len(enabled) == 2:
            config['defense']['trap_combo'] = '+'.join(enabled)
            print(f"[Pipeline] trap_combo 自动推断: {config['defense']['trap_combo']}")

    if args.trap_combo:
        config['defense']['trap_combo'] = args.trap_combo
    if args.num_target_layers is not None:
        config['defense']['num_target_layers'] = args.num_target_layers

    # CLI 覆盖了 trap_combo/num_target_layers 时，重新解析 target_layers
    combo = config['defense'].get('trap_combo')
    num_layers = config['defense'].get('num_target_layers')
    if combo and num_layers:
        from project_core.config import resolve_target_layers
        config['defense']['target_layers'] = resolve_target_layers(combo, num_layers)
        print(f"[Pipeline] target_layers 已解析: {combo} top-{num_layers}")

    attack_epochs = config['training'].get('attack_epochs', 5)
    defense_epochs = config['training'].get('defense_epochs', 25)

    # 解析 target_layers（ConfigManager 已自动解析，这里取最终值）
    target_layers = config.get('defense', {}).get('target_layers')

    # 工作目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    workspace = args.output_dir or os.path.join(
        config['misc']['workspace'], f"pipeline_{tag}_{timestamp}")
    os.makedirs(workspace, exist_ok=True)

    print("=" * 80)
    print("Pipeline: Baseline Attack → Defense → Post-Defense Attack")
    print("=" * 80)
    print(f"Tag: {tag}")
    print(f"Attack epochs: {attack_epochs}")
    print(f"Defense epochs: {defense_epochs}")
    print(f"Trap losses: {[k for k, v in config['defense']['trap_losses'].items() if v.get('static')]}")
    trap_combo = config['defense'].get('trap_combo')
    num_tl = config['defense'].get('num_target_layers')
    if trap_combo and num_tl:
        print(f"Trap combo: {trap_combo}, num_target_layers: {num_tl}")
    if target_layers:
        print(f"Target layers ({len(target_layers)}):")
        for i, l in enumerate(target_layers, 1):
            print(f"  {i:2d}. {l}")
    print(f"Output: {workspace}")

    # 创建数据加载器（全程复用）
    print("\n创建数据加载器...")
    # 需要 opt 来创建 DataManager，临时加载一下
    tmp_mgr = ModelManager(config)
    tmp_mgr.load_model(device='cpu')
    opt = tmp_mgr.opt
    del tmp_mgr.model
    del tmp_mgr

    target_data_mgr = DataManager(config, opt)
    target_data_mgr.setup_dataloaders(train=True, val=True, subset='target')

    source_data_mgr = DataManager(config, opt)
    source_data_mgr.setup_dataloaders(train=False, val=True, subset='source')

    target_train_loader = target_data_mgr.train_loader
    target_val_loader = target_data_mgr.val_loader
    source_val_loader = source_data_mgr.val_loader

    print(f"  Target train: {len(target_train_loader.dataset)} 样本")
    print(f"  Target val: {len(target_val_loader.dataset)} 样本")
    print(f"  Source val: {len(source_val_loader.dataset)} 样本")

    # ========== Phase 1: Baseline Attack（带缓存）==========
    baseline_hash = compute_baseline_hash(config, attack_epochs, args.num_render)
    cache_dir = os.path.join(BASELINE_CACHE_DIR, baseline_hash)
    phase1_dir = os.path.join(workspace, 'phase1_baseline_attack')

    baseline_history, cache_hit = load_baseline_cache(cache_dir)
    if cache_hit:
        # 复制渲染图片到当前 workspace
        copy_cached_renders(cache_dir, phase1_dir)
    else:
        baseline_history = run_attack_phase(
            config, "Phase 1: Baseline Attack",
            target_train_loader, target_val_loader, source_val_loader,
            save_dir=phase1_dir,
            attack_epochs=attack_epochs,
            num_render=args.num_render,
            eval_every_steps=args.eval_every_steps,
        )
        # 保存到缓存
        save_baseline_cache(cache_dir, baseline_history)
        copy_cached_renders(phase1_dir, cache_dir)

    # ========== Phase 2: Defense Training ==========
    defense_history = run_defense_phase(
        config,
        save_dir=os.path.join(workspace, 'phase2_defense'),
        defense_epochs=defense_epochs,
        target_layers=target_layers,
    )

    # ========== Phase 3: Post-Defense Attack ==========
    postdef_history = run_attack_phase(
        config, "Phase 3: Post-Defense Attack",
        target_train_loader, target_val_loader, source_val_loader,
        save_dir=os.path.join(workspace, 'phase3_postdefense_attack'),
        attack_epochs=attack_epochs,
        num_render=args.num_render,
        eval_every_steps=args.eval_every_steps,
        model_resume_override=f"tag:{tag}",
    )

    # ========== Phase 4: 绘图 + 保存结果 ==========
    plot_results(
        baseline_history, postdef_history, defense_history,
        save_path=os.path.join(workspace, 'pipeline_result.png'),
    )

    # 保存指标 JSON
    all_metrics = {
        'config': {
            'tag': tag,
            'attack_epochs': attack_epochs,
            'defense_epochs': defense_epochs,
            'trap_losses': [k for k, v in config['defense']['trap_losses'].items()
                            if v.get('static')],
            'trap_combo': config['defense'].get('trap_combo'),
            'num_target_layers': config['defense'].get('num_target_layers'),
            'target_layers': target_layers,
        },
        'baseline_attack': baseline_history,
        'defense_training': defense_history,
        'postdefense_attack': postdef_history,
    }
    metrics_path = os.path.join(workspace, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # 汇总
    b_final = baseline_history[-1] if baseline_history else {}
    p_final = postdef_history[-1] if postdef_history else {}
    print(f"\n{'='*80}")
    print("Pipeline 结果汇总")
    print(f"{'='*80}")
    print(f"{'指标':<20} {'Baseline':>10} {'Post-Defense':>12} {'Delta':>10}")
    print("-" * 55)
    for key, label in [('psnr', 'Target PSNR'), ('masked_psnr', 'Target MaskedPSNR'),
                        ('loss_lpips', 'Target LPIPS'),
                        ('source_psnr', 'Source PSNR'), ('source_masked_psnr', 'Source MaskedPSNR'),
                        ('source_lpips', 'Source LPIPS')]:
        bv = b_final.get(key, 0)
        pv = p_final.get(key, 0)
        print(f"{label:<20} {bv:>10.4f} {pv:>12.4f} {pv - bv:>+10.4f}")
    print("-" * 55)
    print(f"\n对比图: {os.path.join(workspace, 'pipeline_result.png')}")
    print(f"指标文件: {metrics_path}")
    print(f"工作目录: {workspace}")


if __name__ == '__main__':
    main()
