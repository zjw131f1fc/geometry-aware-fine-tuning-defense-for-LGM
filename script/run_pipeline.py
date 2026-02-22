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

    if args.defense_method is not None:
        config['defense']['method'] = args.defense_method
        print(f"[Pipeline] Defense method 覆盖: {args.defense_method}")

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
    target_data_mgr.setup_dataloaders(train=True, val=False, subset='target')

    source_data_mgr = DataManager(config, opt)
    source_data_mgr.setup_dataloaders(train=False, val=True, subset='source')

    target_train_loader = target_data_mgr.train_loader
    source_val_loader = source_data_mgr.val_loader

    print(f"  Target train: {len(target_train_loader.dataset)} 样本")
    print(f"  Source val: {len(source_val_loader.dataset)} 样本")

    # ========== Phase 1: Baseline Attack（带缓存）==========
    baseline_hash = compute_baseline_hash(config, attack_epochs, args.num_render)
    cache_dir = os.path.join(BASELINE_CACHE_DIR, baseline_hash)
    phase1_dir = os.path.join(workspace, 'phase1_baseline_attack')

    baseline_history, cache_hit = load_baseline_cache(cache_dir)
    baseline_source = None
    baseline_target = None
    if cache_hit:
        # 复用缓存，但需要重新评估初始 source（缓存中没有）
        temp_mgr = ModelManager(config)
        temp_mgr.setup(device='cuda')
        temp_evaluator = Evaluator(temp_mgr.model, device='cuda')
        baseline_source = temp_evaluator.evaluate_on_loader(source_val_loader)
        del temp_evaluator, temp_mgr
        torch.cuda.empty_cache()
        copy_cached_renders(cache_dir, phase1_dir)
    else:
        baseline_history, baseline_source, baseline_target = run_attack(
            config, target_train_loader, source_val_loader,
            save_dir=phase1_dir,
            attack_epochs=attack_epochs,
            num_render=args.num_render,
            eval_every_steps=args.eval_every_steps,
            phase_name="Phase 1: Baseline Attack",
        )
        save_baseline_cache(cache_dir, baseline_history)
        copy_cached_renders(phase1_dir, cache_dir)

    # ========== Phase 2: Defense Training ==========
    defense_tag, defense_history = load_or_train_defense(
        config, device='cuda',
        save_dir=os.path.join(workspace, 'phase2_defense'),
    )

    # ========== Phase 3: Post-Defense Attack ==========
    if defense_tag is not None:
        postdef_history, postdef_source, postdef_target = run_attack(
            config, target_train_loader, source_val_loader,
            save_dir=os.path.join(workspace, 'phase3_postdefense_attack'),
            attack_epochs=attack_epochs,
            num_render=args.num_render,
            eval_every_steps=args.eval_every_steps,
            model_resume_override=f"tag:{defense_tag}",
            phase_name="Phase 3: Post-Defense Attack",
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
    bs = baseline_source or {}
    ps = postdef_source or {}
    for key, label in [('psnr', 'Source PSNR↑'),
                        ('lpips', 'Source LPIPS↓')]:
        bv = bs.get(key, 0)
        pv = ps.get(key, 0)
        print(f"{label:<20} {bv:>10.4f} {pv:>12.4f} {pv - bv:>+10.4f}")

    print("-" * 55)
    print(f"\n对比图: {os.path.join(workspace, 'pipeline_result.png')}")
    print(f"指标文件: {metrics_path}")
    print(f"工作目录: {workspace}")


if __name__ == '__main__':
    main()
