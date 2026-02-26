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
import torch
from datetime import datetime

from project_core import ConfigManager
from models import ModelManager
from data import DataManager
from training import load_or_train_defense, run_attack
from evaluation import Evaluator
from tools import (
    BASELINE_CACHE_DIR, compute_baseline_hash,
    load_baseline_cache, save_baseline_cache, copy_cached_renders,
    plot_pipeline_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline: Attack → Defense → Attack')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--categories', type=str, default=None,
                        help='Target类别，逗号分隔（如 knife,broccoli）覆盖config')
    parser.add_argument('--defense_method', type=str, default=None,
                        help='防御方法：geotrap / naive_unlearning / none')
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
    parser.add_argument('--semantic_deflection', action='store_true',
                        help='启用语义偏转攻击模式')
    parser.add_argument('--supervision_categories', type=str, default=None,
                        help='监督类别（语义偏转模式），逗号分隔（如 durian）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率（覆盖 config）')
    parser.add_argument('--optimizer', type=str, default=None,
                        help='优化器类型：adamw / sgd（覆盖 config）')
    parser.add_argument('--lora_r', type=int, default=None,
                        help='LoRA rank（覆盖 config）')
    parser.add_argument('--lora_alpha', type=int, default=None,
                        help='LoRA alpha（覆盖 config）')
    parser.add_argument('--training_mode', type=str, default=None,
                        help='训练模式：full / lora（覆盖 config）')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='跳过 Baseline Attack（Phase 1），直接从 Defense Training 开始')
    # 互锁机制参数
    parser.add_argument('--robustness', type=str, default=None,
                        help='参数加噪鲁棒性开关：true / false（覆盖 config）')
    return parser.parse_args()



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
    if args.categories is not None:
        categories = [c.strip() for c in args.categories.split(',')]
        config['data']['target']['categories'] = categories
        config['attack']['malicious_content']['malicious_categories'] = categories
        print(f"[Pipeline] Categories 覆盖: {categories}")

    # 语义偏转模式
    supervision_categories = None
    if args.semantic_deflection or args.supervision_categories:
        if not args.supervision_categories:
            raise ValueError("--semantic_deflection 需要指定 --supervision_categories")
        supervision_categories = [c.strip() for c in args.supervision_categories.split(',')]
        config['attack']['semantic_deflection'] = {
            'enabled': True,
            'input_categories': config['data']['target']['categories'],
            'supervision_categories': supervision_categories,
        }
        print(f"[Pipeline] 语义偏转模式: 输入={config['data']['target']['categories']}, "
              f"监督={supervision_categories}")

    if args.defense_method is not None:
        config['defense']['method'] = args.defense_method
        print(f"[Pipeline] Defense method 覆盖: {args.defense_method}")

    if args.attack_epochs is not None:
        config['training']['attack_epochs'] = args.attack_epochs
    if args.defense_epochs is not None:
        config['training']['defense_epochs'] = args.defense_epochs

    # 攻击训练参数覆盖（只影响攻击，不影响防御）
    attack_lr_override = args.lr
    attack_optimizer_override = args.optimizer
    if attack_lr_override is not None:
        print(f"[Pipeline] 攻击 Learning rate 覆盖: {attack_lr_override}")
    if attack_optimizer_override is not None:
        print(f"[Pipeline] 攻击 Optimizer 覆盖: {attack_optimizer_override}")

    # 训练模式和 LoRA 参数（影响攻击和防御）
    if args.training_mode is not None:
        config['training']['mode'] = args.training_mode
        print(f"[Pipeline] Training mode 覆盖: {args.training_mode}")
    if args.lora_r is not None:
        config['lora']['r'] = args.lora_r
        print(f"[Pipeline] LoRA r 覆盖: {args.lora_r}")
    if args.lora_alpha is not None:
        config['lora']['alpha'] = args.lora_alpha
        print(f"[Pipeline] LoRA alpha 覆盖: {args.lora_alpha}")

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

    # 互锁机制覆盖
    if args.robustness is not None:
        val = args.robustness.lower() == 'true'
        config['defense']['robustness']['enabled'] = val
        print(f"[Pipeline] 参数加噪鲁棒性覆盖: {val}")

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

    # 输入数据加载器（target）
    target_data_mgr = DataManager(config, opt)
    target_data_mgr.setup_dataloaders(train=True, val=False, subset='target')
    target_train_loader = target_data_mgr.train_loader

    # 监督数据加载器（语义偏转模式）
    supervision_train_loader = None
    if supervision_categories:
        from data.dataset import SemanticDeflectionDataset

        # 语义偏转模式：攻击输入固定为 coconut
        attack_input_config = copy.deepcopy(config)
        attack_input_config['data']['target']['categories'] = ['coconut']
        attack_input_data_mgr = DataManager(attack_input_config, opt)
        attack_input_data_mgr.setup_dataloaders(train=True, val=False, subset='target')

        supervision_config = copy.deepcopy(config)
        supervision_config['data']['target']['categories'] = supervision_categories
        supervision_data_mgr = DataManager(supervision_config, opt)
        supervision_data_mgr.setup_dataloaders(train=True, val=False, subset='target')

        # 创建配对数据集：input=coconut（固定）, supervision=durian，按 index 对齐
        paired_dataset = SemanticDeflectionDataset(
            attack_input_data_mgr.train_loader.dataset,  # 固定为 coconut
            supervision_data_mgr.train_loader.dataset,
        )
        # 用配对数据集替换 target_train_loader（单个 loader，shuffle 时一起 shuffle）
        from torch.utils.data import DataLoader
        deflection_train_loader = DataLoader(
            paired_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
        )
        # supervision_train_loader 用于评估时的 GT（不 shuffle，保持原始顺序）
        supervision_train_loader = DataLoader(
            supervision_data_mgr.train_loader.dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
        )
        # target_eval_loader 用于评估 vs input（coconut），不 shuffle
        target_eval_loader = DataLoader(
            attack_input_data_mgr.train_loader.dataset,  # 使用固定的 coconut
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
        )
        print(f"  Deflection train (paired): {len(paired_dataset)} 样本")
        print(f"  Supervision eval: {len(supervision_train_loader.dataset)} 样本")

    # Source 数据加载器
    source_data_mgr = DataManager(config, opt)
    source_data_mgr.setup_dataloaders(train=False, val=True, subset='source')
    source_val_loader = source_data_mgr.val_loader

    print(f"  Target train: {len(target_train_loader.dataset)} 样本")
    print(f"  Source val: {len(source_val_loader.dataset)} 样本")

    # 创建攻击专用 config（应用 lr 和 optimizer 覆盖），Phase 1 和 Phase 3 共用
    attack_config = copy.deepcopy(config)
    if attack_lr_override is not None:
        attack_config['training']['lr'] = attack_lr_override
    if attack_optimizer_override is not None:
        attack_config['training']['optimizer'] = attack_optimizer_override

    # ========== Phase 1: Baseline Attack（带缓存）==========
    if args.skip_baseline:
        print(f"\n{'='*80}")
        print("  跳过 Phase 1: Baseline Attack")
        print(f"{'='*80}")
        baseline_history, baseline_source, baseline_target = None, None, None
        baseline_gaussians = None
    else:
        baseline_hash = compute_baseline_hash(attack_config, attack_epochs, args.num_render, supervision_categories)
        cache_dir = os.path.join(BASELINE_CACHE_DIR, baseline_hash)
        phase1_dir = os.path.join(workspace, 'phase1_baseline_attack')
        gaussians_cache_path = os.path.join(cache_dir, 'baseline_gaussians.pt')

        baseline_history, baseline_source, baseline_target, cache_hit = load_baseline_cache(cache_dir)
        if cache_hit:
            # 缓存命中：baseline_source 和 baseline_target 都从缓存加载
            # 如果旧缓存没有 baseline_source，重新评估
            if baseline_source is None:
                temp_mgr = ModelManager(config)
                temp_mgr.setup(device='cuda')
                temp_evaluator = Evaluator(temp_mgr.model, device='cuda')
                baseline_source = temp_evaluator.evaluate_on_loader(source_val_loader)
                del temp_evaluator, temp_mgr
                torch.cuda.empty_cache()
            copy_cached_renders(cache_dir, phase1_dir)
            # 加载缓存的 Gaussian
            if os.path.exists(gaussians_cache_path):
                baseline_gaussians = torch.load(gaussians_cache_path, map_location='cpu', weights_only=True)
                print(f"[Cache] baseline Gaussians 已加载: {len(baseline_gaussians)} 个样本")
            else:
                baseline_gaussians = None
                print("[Cache] 旧缓存无 baseline Gaussians，Phase 3 将不计算距离")
        else:
            baseline_history, baseline_source, baseline_target, baseline_gaussians = run_attack(
                attack_config, deflection_train_loader if supervision_categories else target_train_loader,
                source_val_loader,
                supervision_loader=supervision_train_loader,
                target_eval_loader=target_eval_loader if supervision_categories else None,
                save_dir=phase1_dir,
                attack_epochs=attack_epochs,
                num_render=args.num_render,
                eval_every_steps=args.eval_every_steps,
                phase_name="Phase 1: Baseline Attack",
                return_gaussians=True,
            )
            save_baseline_cache(cache_dir, baseline_history, baseline_source, baseline_target)
            copy_cached_renders(phase1_dir, cache_dir)
            # 缓存 Gaussian
            if baseline_gaussians:
                torch.save(baseline_gaussians, gaussians_cache_path)
                print(f"[Cache] baseline Gaussians 已缓存: {len(baseline_gaussians)} 个样本")

    # ========== Phase 2: Defense Training ==========
    defense_tag, defense_history = load_or_train_defense(
        config, device='cuda',
        save_dir=os.path.join(workspace, 'phase2_defense'),
    )

    # ========== Phase 3: Post-Defense Attack ==========
    if defense_tag is not None:
        postdef_history, postdef_source, postdef_target = run_attack(
            attack_config, deflection_train_loader if supervision_categories else target_train_loader,
            source_val_loader,
            supervision_loader=supervision_train_loader,
            target_eval_loader=target_eval_loader if supervision_categories else None,
            save_dir=os.path.join(workspace, 'phase3_postdefense_attack'),
            attack_epochs=attack_epochs,
            num_render=args.num_render,
            eval_every_steps=args.eval_every_steps,
            model_resume_override=f"tag:{defense_tag}",
            phase_name="Phase 3: Post-Defense Attack",
            ref_gaussians=baseline_gaussians if not args.skip_baseline else None,
        )
    else:
        # defense.method=none，跳过 Phase 3
        print("\n[Pipeline] defense.method=none，跳过 Post-Defense Attack")
        postdef_history, postdef_source, postdef_target = None, None, None

    # ========== Phase 4: 绘图 + 保存结果 ==========
    plot_pipeline_results(
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
            'semantic_deflection': supervision_categories is not None,
            'supervision_categories': supervision_categories,
        },
        'baseline_attack': baseline_history,
        'baseline_source': baseline_source,
        'baseline_target': baseline_target,
        'defense_training': defense_history,
        'postdefense_attack': postdef_history,
        'postdefense_source': postdef_source,
        'postdefense_target': postdef_target,
    }
    metrics_path = os.path.join(workspace, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # 汇总：4 个核心指标
    print(f"\n{'='*80}")
    print("Pipeline 结果汇总")
    print(f"{'='*80}")

    if supervision_categories:
        # 语义偏转模式：分别显示 vs input 和 vs supervision
        print(f"{'指标':<25} {'Baseline':>10} {'Post-Defense':>12} {'Delta':>10}")
        print("-" * 60)

        bt = baseline_target or {}
        pt = postdef_target or {}

        # vs Input 指标
        bt_input = bt.get('input', bt)  # 兼容旧格式
        pt_input = pt.get('input', pt)
        print("\n[vs Input Category]")
        for key, label in [('lpips', 'Input LPIPS↑'),
                          ('psnr', 'Input PSNR↓')]:
            bv = bt_input.get(key, 0)
            pv = pt_input.get(key, 0)
            print(f"{label:<25} {bv:>10.4f} {pv:>12.4f} {pv - bv:>+10.4f}")

        # vs Supervision 指标
        bt_sup = bt.get('supervision', {})
        pt_sup = pt.get('supervision', {})
        print("\n[vs Supervision Category]")
        for key, label in [('lpips', 'Supervision LPIPS↓'),
                          ('psnr', 'Supervision PSNR↑')]:
            bv = bt_sup.get(key, 0)
            pv = pt_sup.get(key, 0)
            print(f"{label:<25} {bv:>10.4f} {pv:>12.4f} {pv - bv:>+10.4f}")
    else:
        # 标准模式：原有显示方式
        print(f"{'指标':<20} {'Baseline':>10} {'Post-Defense':>12} {'Delta':>10}")
        print("-" * 55)

        # Target 指标（攻击后）：LPIPS↑ = 防御有效，PSNR↓ = 防御有效
        bt = baseline_target or {}
        pt = postdef_target or {}
        for key, label in [('lpips', 'Target LPIPS↑'),
                          ('psnr', 'Target PSNR↓')]:
            bv = bt.get(key, 0)
            pv = pt.get(key, 0)
            print(f"{label:<20} {bv:>10.4f} {pv:>12.4f} {pv - bv:>+10.4f}")

    # Source 指标（防御后攻击前）：PSNR↑ = 保持能力，LPIPS↓ = 保持能力
    print("\n[Source Capability]")
    bs = baseline_source or {}
    ps = postdef_source or {}
    for key, label in [('psnr', 'Source PSNR↑'),
                        ('lpips', 'Source LPIPS↓')]:
        bv = bs.get(key, 0)
        pv = ps.get(key, 0)
        col_width = 25 if supervision_categories else 20
        print(f"{label:<{col_width}} {bv:>10.4f} {pv:>12.4f} {pv - bv:>+10.4f}")

    separator_len = 60 if supervision_categories else 55
    print("-" * separator_len)

    # Gaussian 诊断汇总
    for phase_label, tgt in [("Baseline", baseline_target), ("Post-Defense", postdef_target)]:
        if tgt and 'gaussian_diag' in tgt:
            gd = tgt['gaussian_diag']
            print(f"\n[Gaussian 诊断 - {phase_label}]")
            print(f"  diagnosis: {gd.get('diagnosis', 'N/A')}")
            print(f"  opacity_mean={gd['opacity_mean']:.4f}, "
                  f"pos_spread={gd['pos_spread']:.4f}, "
                  f"scale_mean={gd['scale_mean']:.6f}")
            print(f"  render_white_ratio={gd['render_white_ratio']:.4f}, "
                  f"rgb_white_ratio={gd['rgb_white_ratio']:.4f}")
            print(f"  trap: position={gd['trap_position']:.4f}, "
                  f"scale={gd['trap_scale']:.4f}, "
                  f"opacity={gd['trap_opacity']:.4f}, "
                  f"rotation={gd['trap_rotation']:.4f}")
            if 'gaussian_dist_to_baseline' in gd:
                print(f"  gaussian_dist_to_baseline={gd['gaussian_dist_to_baseline']:.6f}")

    print(f"\n对比图: {os.path.join(workspace, 'pipeline_result.png')}")
    print(f"指标文件: {metrics_path}")
    print(f"工作目录: {workspace}")


if __name__ == '__main__':
    main()
