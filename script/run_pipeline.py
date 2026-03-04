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
# Make common caches writable / stable (important for headless + multiprocessing)
if not os.environ.get('OMP_NUM_THREADS', '').isdigit():
    os.environ['OMP_NUM_THREADS'] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

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
    parser.add_argument('--attack_steps', type=int, default=None,
                        help='攻击训练 step 数（优先于 attack_epochs）')
    parser.add_argument('--defense_epochs', type=int, default=None,
                        help='防御训练 epoch 数')
    parser.add_argument('--defense_steps', type=int, default=None,
                        help='防御训练 step 数（优先于 defense_epochs）')
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
    parser.add_argument('--attack_target_dataset', type=str, default=None,
                        help="攻击阶段 target 数据集：omni / gso / objaverse（覆盖 config.data.target.dataset）")
    parser.add_argument('--defense_target_dataset', type=str, default=None,
                        help="防御阶段 target 数据集：omni / gso / objaverse（覆盖 config.defense.target.dataset）")
    parser.add_argument('--defense_batch_size', type=int, default=None,
                        help="防御训练 batch size（覆盖 config.defense.batch_size；不影响攻击训练 batch size）")
    parser.add_argument('--attack_grad_accumulation_steps', type=int, default=None,
                        help="攻击梯度累计步数（覆盖 config.training.gradient_accumulation_steps）")
    parser.add_argument('--defense_grad_accumulation_steps', type=int, default=None,
                        help="防御梯度累计步数（覆盖 config.defense.gradient_accumulation_steps；不影响攻击）")
    parser.add_argument('--defense_cache_mode', type=str, default='registry',
                        help="防御模型缓存策略：registry(读写) / readonly(只读不写) / none(不读不写，最省磁盘)")
    # 互锁机制参数
    parser.add_argument('--robustness', type=str, default=None,
                        help='参数加噪鲁棒性开关：true / false（覆盖 config）')
    parser.add_argument('--noise_scale', type=float, default=None,
                        help='参数加噪 σ（覆盖 config.defense.robustness.noise_scale）')
    # 梯度手术参数
    parser.add_argument('--grad_surgery_enabled', type=str, default=None,
                        help='梯度手术开关：true / false（覆盖 config.defense.grad_surgery.enabled）')
    parser.add_argument('--grad_surgery_mode', type=str, default=None,
                        help='梯度手术模式：pcgrad / projection（覆盖 config.defense.grad_surgery.mode）')
    # 梯度裁剪参数
    parser.add_argument('--grad_clip_mode', type=str, default=None,
                        help='梯度裁剪模式：norm / energy / both（覆盖 config.defense.grad_clip.mode）')
    parser.add_argument('--grad_energy_mult', type=float, default=None,
                        help='能量梯度裁剪倍数（覆盖 config.defense.grad_clip.energy_mult）')
    # 陷阱聚合参数
    parser.add_argument('--trap_aggregation_method', type=str, default=None,
                        help='陷阱聚合方法：sum / mean / bottleneck_logsumexp（覆盖 config.defense.trap_aggregation.method）')
    parser.add_argument('--trap_bottleneck_tau', type=float, default=None,
                        help='Bottleneck聚合温度参数（覆盖 config.defense.trap_aggregation.tau）')
    # 反捷径机制参数
    parser.add_argument('--freeze_head', type=str, default=None,
                        help='冻结头部卷积层：true / false（覆盖 config.defense.antishortcut.freeze_head）')
    parser.add_argument('--freeze_lora_targets', type=str, default=None,
                        help='冻结 LoRA 目标层（qkv/proj）：true / false（覆盖 config.defense.antishortcut.freeze_lora_targets）')
    return parser.parse_args()



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from tools import set_seed  # 延迟加载

    args = parse_args()
    device = 'cuda'

    def _dump_json(path: str, obj) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

    def _print_kv(key: str, value) -> None:
        print(f"  {key}: {value}")

    def _last_attack_eval_entry(step_history):
        if not step_history:
            return None
        for entry in reversed(step_history):
            if not isinstance(entry, dict):
                continue
            if 'masked_psnr' in entry and 'masked_lpips' in entry and 'step' in entry:
                return entry
        return None

    def _steps_to_reach_attack_quality(step_history, target_masked_psnr, target_masked_lpips,
                                       eps_psnr=0.0, eps_lpips=0.0):
        """
        用攻击阶段的 step-eval 指标（masked_psnr / masked_lpips）计算：
        达到某个“攻击质量阈值”（PSNR >= 阈值 且 LPIPS <= 阈值）所需的最早 step。

        注意：PSNR 越大越好，LPIPS 越小越好。
        """
        if not step_history:
            return None
        for entry in step_history:
            if not isinstance(entry, dict):
                continue
            psnr = entry.get('masked_psnr')
            lpips = entry.get('masked_lpips')
            step = entry.get('step')
            if psnr is None or lpips is None or step is None:
                continue
            if (psnr >= (target_masked_psnr - eps_psnr)) and (lpips <= (target_masked_lpips + eps_lpips)):
                try:
                    return int(step)
                except Exception:
                    return step
        return None

    # 加载配置
    config = ConfigManager(args.config).config
    set_seed(config['misc']['seed'])

    # Defense cache mode（需要在打印 Summary 前就解析出来）
    defense_cache_mode = (args.defense_cache_mode or "registry").lower()
    if defense_cache_mode not in ("registry", "readonly", "none"):
        raise ValueError(f"--defense_cache_mode 不支持: {args.defense_cache_mode}")

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

    # attack/defense steps/epochs 覆盖：保持“二选一”语义，避免 config key 存在但为 null 导致 None 分支
    if args.attack_steps is not None:
        config['training']['attack_steps'] = args.attack_steps
        config['training']['attack_epochs'] = None
    if args.attack_epochs is not None:
        config['training']['attack_epochs'] = args.attack_epochs
        config['training']['attack_steps'] = None
    if args.defense_steps is not None:
        config['training']['defense_steps'] = args.defense_steps
        config['training']['defense_epochs'] = None
    if args.defense_epochs is not None:
        config['training']['defense_epochs'] = args.defense_epochs
        config['training']['defense_steps'] = None

    # 跨数据集泛化：允许 Attack/Defense 使用不同 target dataset
    if args.attack_target_dataset is not None:
        config['data']['target']['dataset'] = args.attack_target_dataset
        print(f"[Pipeline] Attack target dataset 覆盖: {args.attack_target_dataset}")
    if args.defense_target_dataset is not None:
        config.setdefault('defense', {}).setdefault('target', {})
        config['defense']['target']['dataset'] = args.defense_target_dataset
        print(f"[Pipeline] Defense target dataset 覆盖: {args.defense_target_dataset}")

    if args.defense_batch_size is not None:
        if args.defense_batch_size <= 0:
            raise ValueError("--defense_batch_size 必须为正整数")
        config.setdefault('defense', {})
        config['defense']['batch_size'] = args.defense_batch_size
        print(f"[Pipeline] Defense batch_size 覆盖: {args.defense_batch_size}")

    if args.attack_grad_accumulation_steps is not None:
        if args.attack_grad_accumulation_steps <= 0:
            raise ValueError("--attack_grad_accumulation_steps 必须为正整数")
        config.setdefault('training', {})
        config['training']['gradient_accumulation_steps'] = args.attack_grad_accumulation_steps
        print(f"[Pipeline] Attack gradient_accumulation_steps 覆盖: {args.attack_grad_accumulation_steps}")

    if args.defense_grad_accumulation_steps is not None:
        if args.defense_grad_accumulation_steps <= 0:
            raise ValueError("--defense_grad_accumulation_steps 必须为正整数")
        config.setdefault('defense', {})
        config['defense']['gradient_accumulation_steps'] = args.defense_grad_accumulation_steps
        print(f"[Pipeline] Defense gradient_accumulation_steps 覆盖: {args.defense_grad_accumulation_steps}")

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
        for loss_name in ('position', 'scale', 'opacity', 'rotation', 'color'):
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
    if args.noise_scale is not None:
        config.setdefault('defense', {}).setdefault('robustness', {})
        config['defense']['robustness']['noise_scale'] = float(args.noise_scale)
        print(f"[Pipeline] noise_scale 覆盖: {args.noise_scale}")

    # 梯度手术覆盖
    if args.grad_surgery_enabled is not None:
        val = args.grad_surgery_enabled.lower() == 'true'
        config.setdefault('defense', {}).setdefault('grad_surgery', {})
        config['defense']['grad_surgery']['enabled'] = val
        print(f"[Pipeline] grad_surgery.enabled 覆盖: {val}")
    if args.grad_surgery_mode is not None:
        config.setdefault('defense', {}).setdefault('grad_surgery', {})
        config['defense']['grad_surgery']['mode'] = args.grad_surgery_mode
        print(f"[Pipeline] grad_surgery.mode 覆盖: {args.grad_surgery_mode}")

    # 梯度裁剪覆盖
    if args.grad_clip_mode is not None:
        config.setdefault('defense', {}).setdefault('grad_clip', {})
        config['defense']['grad_clip']['mode'] = args.grad_clip_mode
        print(f"[Pipeline] grad_clip.mode 覆盖: {args.grad_clip_mode}")
    if args.grad_energy_mult is not None:
        config.setdefault('defense', {}).setdefault('grad_clip', {})
        config['defense']['grad_clip']['energy_mult'] = float(args.grad_energy_mult)
        print(f"[Pipeline] grad_clip.energy_mult 覆盖: {args.grad_energy_mult}")

    # 陷阱聚合覆盖
    if args.trap_aggregation_method is not None:
        config.setdefault('defense', {}).setdefault('trap_aggregation', {})
        config['defense']['trap_aggregation']['method'] = args.trap_aggregation_method
        print(f"[Pipeline] trap_aggregation.method 覆盖: {args.trap_aggregation_method}")
    if args.trap_bottleneck_tau is not None:
        config.setdefault('defense', {}).setdefault('trap_aggregation', {})
        config['defense']['trap_aggregation']['tau'] = float(args.trap_bottleneck_tau)
        print(f"[Pipeline] trap_aggregation.tau 覆盖: {args.trap_bottleneck_tau}")

    # 反捷径机制覆盖
    if args.freeze_head is not None:
        val = args.freeze_head.lower() == 'true'
        config.setdefault('defense', {}).setdefault('antishortcut', {})
        config['defense']['antishortcut']['freeze_head'] = val
        print(f"[Pipeline] antishortcut.freeze_head 覆盖: {val}")

    if args.freeze_lora_targets is not None:
        val = args.freeze_lora_targets.lower() == 'true'
        config.setdefault('defense', {}).setdefault('antishortcut', {})
        config['defense']['antishortcut']['freeze_lora_targets'] = val
        print(f"[Pipeline] antishortcut.freeze_lora_targets 覆盖: {val}")

    # CLI 覆盖了 trap_combo/num_target_layers 时，重新解析 target_layers
    combo = config['defense'].get('trap_combo')
    num_layers = config['defense'].get('num_target_layers')
    if combo and num_layers:
        from project_core.config import resolve_target_layers
        config['defense']['target_layers'] = resolve_target_layers(combo, num_layers)
        print(f"[Pipeline] target_layers 已解析: {combo} top-{num_layers}")

    attack_steps = config['training'].get('attack_steps')
    attack_epochs = config['training'].get('attack_epochs')
    if attack_steps is None and attack_epochs is None:
        attack_epochs = 5

    defense_steps = config['training'].get('defense_steps')
    defense_epochs = config['training'].get('defense_epochs')
    if defense_steps is None and defense_epochs is None:
        defense_epochs = 25

    # 解析 target_layers（ConfigManager 已自动解析，这里取最终值）
    target_layers = config.get('defense', {}).get('target_layers')

    # 工作目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    workspace = args.output_dir or os.path.join(
        config['misc']['workspace'], f"pipeline_{tag}_{timestamp}")
    os.makedirs(workspace, exist_ok=True)

    # 保存可复现实验的配置快照（比单纯打印更可靠）
    _dump_json(os.path.join(workspace, "pipeline_args.json"), vars(args))
    _dump_json(os.path.join(workspace, "pipeline_config.json"), config)
    _dump_json(os.path.join(workspace, "pipeline_effective.json"), {
        "pipeline_tag": tag,
        "config_path": args.config,
        "attack_steps": attack_steps,
        "attack_epochs": attack_epochs,
        "defense_steps": defense_steps,
        "defense_epochs": defense_epochs,
        "defense_cache_mode": defense_cache_mode,
        "semantic_deflection": bool(supervision_categories),
        "supervision_categories": supervision_categories,
        "num_render": args.num_render,
        "eval_every_steps": args.eval_every_steps,
        "skip_baseline": bool(args.skip_baseline),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR"),
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
        },
    })

    print("=" * 80)
    print("Pipeline: Baseline Attack → Defense → Post-Defense Attack")
    print("=" * 80)
    print(f"Pipeline tag: {tag}")
    print(f"Config: {args.config}")
    if attack_steps is not None:
        print(f"Attack steps: {attack_steps}")
    else:
        print(f"Attack epochs: {attack_epochs}")
    if defense_steps is not None:
        print(f"Defense steps: {defense_steps}")
    else:
        print(f"Defense epochs: {defense_epochs}")
    print(f"Defense cache mode: {defense_cache_mode}")
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
    print(f"Repro dumps: {os.path.join(workspace, 'pipeline_config.json')} "
          f"(+ {os.path.join(workspace, 'pipeline_args.json')}, {os.path.join(workspace, 'pipeline_effective.json')})")

    # 打印关键配置，方便事后追溯
    print("\n--- Config Summary ---")
    print("  [env]")
    _print_kv("gpu", args.gpu)
    _print_kv("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
    _print_kv("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS"))
    _print_kv("MPLCONFIGDIR", os.environ.get("MPLCONFIGDIR"))
    _print_kv("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"))

    print("  [model]")
    _print_kv("model.size", config.get('model', {}).get('size'))
    _print_kv("model.resume", config.get('model', {}).get('resume'))
    _print_kv("model.device", config.get('model', {}).get('device'))
    _print_kv("lora.r", config.get('lora', {}).get('r'))
    _print_kv("lora.alpha", config.get('lora', {}).get('alpha'))
    _print_kv("lora.target_modules", config.get('lora', {}).get('target_modules'))

    print("  [data]")
    _print_kv("data.root", config.get('data', {}).get('root'))
    _print_kv("target.dataset", config.get('data', {}).get('target', {}).get('dataset'))
    _print_kv("target.categories", config.get('data', {}).get('target', {}).get('categories'))
    _print_kv("source.dataset", config.get('data', {}).get('source', {}).get('dataset'))
    _print_kv("source.categories", config.get('data', {}).get('source', {}).get('categories'))
    _print_kv("source_ratio", config.get('data', {}).get('source_ratio'))
    _print_kv("attack_samples_per_category", config.get('data', {}).get('attack_samples_per_category'))
    _print_kv("data.num_workers", config.get('data', {}).get('num_workers'))
    obj_split = config.get('data', {}).get('object_split', {})
    if isinstance(obj_split, dict) and obj_split:
        obj_split_counts = {
            k: (len(v) if isinstance(v, list) else None)
            for k, v in obj_split.items()
        }
        _print_kv("data.object_split.counts", obj_split_counts)

    print("  [attack/train]")
    _print_kv("training.mode", config.get('training', {}).get('mode'))
    _print_kv("training.optimizer", config.get('training', {}).get('optimizer'))
    _print_kv("training.lr", config.get('training', {}).get('lr'))
    _print_kv("training.batch_size", config.get('training', {}).get('batch_size'))
    _print_kv("training.gradient_accumulation_steps", config.get('training', {}).get('gradient_accumulation_steps'))
    _print_kv("attack.scenario", config.get('attack', {}).get('scenario'))
    _print_kv("attack.mode", config.get('attack', {}).get('mode'))
    _print_kv("attack.malicious_categories", config.get('attack', {}).get('malicious_content', {}).get('malicious_categories'))
    _print_kv("attack.lr_override", attack_lr_override)
    _print_kv("attack.optimizer_override", attack_optimizer_override)
    _print_kv("attack.eval_every_steps", args.eval_every_steps)
    _print_kv("attack.num_render", args.num_render)
    _print_kv("attack.skip_baseline", args.skip_baseline)
    _print_kv("attack.semantic_deflection", bool(supervision_categories))
    if supervision_categories:
        _print_kv("attack.supervision_categories", supervision_categories)

    print("  [defense]")
    _print_kv("defense.method", config.get('defense', {}).get('method'))
    _print_kv("defense.cache_mode", defense_cache_mode)
    defense_target_cfg = config.get('defense', {}).get('target', {})
    _print_kv("defense.target.dataset", defense_target_cfg.get('dataset') if isinstance(defense_target_cfg, dict) else None)
    _print_kv("defense.target.categories", defense_target_cfg.get('categories') if isinstance(defense_target_cfg, dict) else None)
    _print_kv("defense.target.samples_per_object", defense_target_cfg.get('samples_per_object') if isinstance(defense_target_cfg, dict) else None)
    _print_kv("lambda_trap", config.get('defense', {}).get('lambda_trap'))
    _print_kv("lambda_distill", config.get('defense', {}).get('lambda_distill'))
    _print_kv("distill_loss_order", config.get('defense', {}).get('distill_loss_order'))
    _print_kv("defense.batch_size", config.get('defense', {}).get('batch_size'))
    _print_kv("defense.gradient_accumulation_steps", config.get('defense', {}).get('gradient_accumulation_steps'))
    robust_cfg = config.get('defense', {}).get('robustness', {})
    _print_kv("robustness.enabled", robust_cfg.get('enabled'))
    _print_kv("robustness.noise_scale", robust_cfg.get('noise_scale'))
    conflict_cfg = config.get('defense', {}).get('gradient_conflict', {})
    _print_kv("gradient_conflict.enabled", conflict_cfg.get('enabled'))
    _print_kv("gradient_conflict.weight", conflict_cfg.get('weight'))
    _print_kv("gradient_conflict.every_k_steps", conflict_cfg.get('every_k_steps'))
    _print_kv("trap_losses.static", [k for k, v in config.get('defense', {}).get('trap_losses', {}).items() if v.get('static')])
    _print_kv("trap_combo", config.get('defense', {}).get('trap_combo'))
    _print_kv("num_target_layers", config.get('defense', {}).get('num_target_layers'))
    defense_eval_cfg = config.get('defense', {}).get('eval', {})
    if isinstance(defense_eval_cfg, dict) and defense_eval_cfg:
        _print_kv("defense.eval", defense_eval_cfg)

    try:
        attack_bs = int(config['training'].get('batch_size', 1))
        attack_ga = int(config['training'].get('gradient_accumulation_steps', 1))
        defense_bs_cfg = config.get('defense', {}).get('batch_size')
        defense_bs = attack_bs if defense_bs_cfg is None else int(defense_bs_cfg)
        # DefenseTrainer inherits training.gradient_accumulation_steps when defense.* is unset.
        defense_ga = int(config.get('defense', {}).get(
            'gradient_accumulation_steps',
            config.get('training', {}).get('gradient_accumulation_steps', 1),
        ))
        print(f"  attack.effective_batch_size: {attack_bs}×{attack_ga}={attack_bs * attack_ga}")
        print(f"  defense.effective_batch_size: {defense_bs}×{defense_ga}={defense_bs * defense_ga}")
    except Exception:
        pass
    print("--- End Config ---")

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
    _dump_json(os.path.join(workspace, "attack_config.json"), attack_config)

    # ========== Phase 1: Baseline Attack（带缓存）==========
    if args.skip_baseline:
        print(f"\n{'='*80}")
        print("  跳过 Phase 1: Baseline Attack")
        print(f"{'='*80}")
        baseline_history, baseline_source, baseline_target = None, None, None
        baseline_gaussians = None
    else:
        baseline_hash = compute_baseline_hash(
            attack_config,
            attack_epochs,
            args.num_render,
            supervision_categories,
            attack_steps=attack_steps,
            eval_every_steps=args.eval_every_steps,
        )
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
                attack_steps=attack_steps,
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

    # ========== Phase 1.9: 清理 Phase 1 残留的显存 ==========
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ========== Phase 2: Defense Training ==========
    need_defense_state = (defense_cache_mode != "registry")
    if need_defense_state:
        defense_tag, defense_history, defense_state_dict = load_or_train_defense(
            config, device='cuda',
            save_dir=os.path.join(workspace, 'phase2_defense'),
            cache_mode=defense_cache_mode,
            return_state_dict=True,
        )
    else:
        defense_tag, defense_history = load_or_train_defense(
            config, device='cuda',
            save_dir=os.path.join(workspace, 'phase2_defense'),
            cache_mode=defense_cache_mode,
            return_state_dict=False,
        )
        defense_state_dict = None

    # ========== Phase 2.5: Defense transfer diagnostics (pre-attack) ==========
    transfer_diag_cfg = (config.get('defense', {}).get('transfer_diag', {}) or {})
    transfer_diag_enabled = transfer_diag_cfg.get('enabled', None)
    if transfer_diag_enabled is None:
        # Default: enable only when attack.debug is enabled (analysis mode).
        transfer_diag_enabled = bool(config.get('attack', {}).get('debug', {}).get('enabled', False))
    transfer_diag_samples = int(transfer_diag_cfg.get('num_samples', 8) or 8)

    defense_transfer_diag = None
    if defense_tag is not None and transfer_diag_enabled:
        print(f"\n{'='*80}")
        print("  Phase 2.5: Defense transfer diagnostics (pre-attack)")
        print(f"  num_samples={transfer_diag_samples}")
        print(f"{'='*80}")

        diag_cfg = copy.deepcopy(config)
        if defense_state_dict is None:
            diag_cfg.setdefault('model', {})
            diag_cfg['model']['resume'] = f"tag:{defense_tag}"
        diag_mgr = ModelManager(diag_cfg)
        # If we have an in-memory defense state_dict, load it into a *vanilla* model first.
        # This avoids PEFT (LoRA) key mismatches (qkv/proj -> base_layer/lora_A/lora_B).
        if defense_state_dict is not None:
            diag_mgr.setup(apply_lora=False, device='cuda')
        else:
            diag_mgr.setup(device='cuda')
        diag_model = diag_mgr.model

        if defense_state_dict is not None:
            raw = diag_model
            while hasattr(raw, 'module'):
                raw = raw.module

            override_keys = list(getattr(defense_state_dict, "keys", lambda: [])())
            has_peft_prefix = any(str(k).startswith("base_model.") for k in override_keys)

            load_target = raw
            if (not has_peft_prefix) and hasattr(raw, "base_model"):
                base = getattr(raw, "base_model", None)
                if hasattr(base, "model") and hasattr(base.model, "load_state_dict"):
                    load_target = base.model
                else:
                    gbm = getattr(raw, "get_base_model", None)
                    if callable(gbm):
                        try:
                            load_target = gbm()
                        except Exception:
                            load_target = raw

            missing, unexpected = load_target.load_state_dict(defense_state_dict, strict=False)
            if missing or unexpected:
                print(f"  [TransferDiag] state_dict non-strict (missing={len(missing)}, unexpected={len(unexpected)})")

        diag_eval = Evaluator(diag_model, device='cuda')

        # Build a defense_target loader (OmniObject3D subset) for diagnosis.
        defense_target_mgr = DataManager(config, opt)
        defense_target_mgr.setup_dataloaders(train=True, val=False, subset='defense_target')
        defense_target_loader = defense_target_mgr.train_loader

        try:
            diag_on_defense_target = diag_eval.diagnose_gaussians(
                defense_target_loader, num_samples=transfer_diag_samples)
            diag_on_attack_target = diag_eval.diagnose_gaussians(
                target_train_loader, num_samples=transfer_diag_samples)
            defense_transfer_diag = {
                'on_defense_target': diag_on_defense_target,
                'on_attack_target': diag_on_attack_target,
            }
            print("  [TransferDiag] done.")
        except Exception as e:
            print(f"  [TransferDiag] skipped due to error: {type(e).__name__}: {e}")

        del diag_eval, diag_model, diag_mgr, defense_target_mgr
        torch.cuda.empty_cache()

    # ========== Phase 2.9: 显式清理 defense 阶段残留的显存 ==========
    # 在 Phase 3 开始前，确保所有 defense 相关的对象都被释放
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ========== Phase 3: Post-Defense Attack ==========
    if defense_tag is not None:
        postdef_history, postdef_source, postdef_target = run_attack(
            attack_config, deflection_train_loader if supervision_categories else target_train_loader,
            source_val_loader,
            supervision_loader=supervision_train_loader,
            target_eval_loader=target_eval_loader if supervision_categories else None,
            save_dir=os.path.join(workspace, 'phase3_postdefense_attack'),
            attack_epochs=attack_epochs,
            attack_steps=attack_steps,
            num_render=args.num_render,
            eval_every_steps=args.eval_every_steps,
            model_resume_override=f"tag:{defense_tag}" if defense_state_dict is None else None,
            phase_name="Phase 3: Post-Defense Attack",
            ref_gaussians=baseline_gaussians if not args.skip_baseline else None,
            model_state_dict_override=defense_state_dict,
        )
    else:
        # defense.method=none，跳过 Phase 3
        print("\n[Pipeline] defense.method=none，跳过 Post-Defense Attack")
        postdef_history, postdef_source, postdef_target = None, None, None

    # ========== Phase X: 攻击阶段步数分析（baseline 达标步数） ==========
    baseline_end_entry = _last_attack_eval_entry(baseline_history)
    if baseline_end_entry is not None:
        baseline_effect_psnr = baseline_end_entry.get('masked_psnr')
        baseline_effect_lpips = baseline_end_entry.get('masked_lpips')
        baseline_effect_step = baseline_end_entry.get('step')
    else:
        baseline_effect_psnr, baseline_effect_lpips, baseline_effect_step = None, None, None

    baseline_steps_to_reach_baseline = None
    postdef_steps_to_reach_baseline = None
    if baseline_effect_psnr is not None and baseline_effect_lpips is not None:
        baseline_steps_to_reach_baseline = _steps_to_reach_attack_quality(
            baseline_history, baseline_effect_psnr, baseline_effect_lpips
        )
        postdef_steps_to_reach_baseline = _steps_to_reach_attack_quality(
            postdef_history, baseline_effect_psnr, baseline_effect_lpips
        )

    print(f"\n{'='*80}")
    print("Attack 阶段达标步数（达到 Baseline Attack 最终质量阈值）")
    print(f"{'='*80}")
    print(f"eval_every_steps: {args.eval_every_steps}")
    if baseline_end_entry is None:
        print("Baseline Attack: 无 step 历史（可能 skip_baseline 或旧缓存）")
    else:
        if baseline_effect_psnr is None or baseline_effect_lpips is None:
            print(f"Baseline Attack: step 历史缺少 masked_psnr/masked_lpips（step={baseline_effect_step}）")
        else:
            print(f"Baseline Attack 最终阈值（来自最后一次 step-eval）: "
                  f"step={baseline_effect_step}, masked_psnr={baseline_effect_psnr:.2f}, "
                  f"masked_lpips={baseline_effect_lpips:.4f}")
            print(f"Baseline Attack 达到自身最终效果的最早 step: {baseline_steps_to_reach_baseline}")
        if postdef_history is None:
            print("Post-Defense Attack: 无（defense.method=none）")
        else:
            print(f"Post-Defense Attack 达到 Baseline Attack 最终效果的最早 step: {postdef_steps_to_reach_baseline}")

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
            'attack_steps': attack_steps,
            'defense_epochs': defense_epochs,
            'defense_steps': defense_steps,
            'eval_every_steps': args.eval_every_steps,
            'defense_cache_mode': defense_cache_mode,
            'trap_losses': [
                k for k, v in config['defense']['trap_losses'].items()
                if v.get('static')
            ],
            'trap_combo': config['defense'].get('trap_combo'),
            'num_target_layers': config['defense'].get('num_target_layers'),
            'target_layers': target_layers,
            'semantic_deflection': supervision_categories is not None,
            'supervision_categories': supervision_categories,
        },
        'defense_transfer_diag': defense_transfer_diag,
        'analysis': {
            'baseline_attack_effect_at_end': (
                {
                    'step': baseline_effect_step,
                    'masked_psnr': baseline_effect_psnr,
                    'masked_lpips': baseline_effect_lpips,
                } if baseline_end_entry is not None else None
            ),
            'baseline_attack_steps_to_effect': baseline_steps_to_reach_baseline,
            'postdefense_attack_steps_to_baseline_effect': postdef_steps_to_reach_baseline,
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
        for key, label in [('psnr', 'Input PSNR↑'),
                          ('lpips', 'Input LPIPS↓')]:
            bv = bt_input.get(key, 0)
            pv = pt_input.get(key, 0)
            print(f"{label:<25} {bv:>10.4f} {pv:>12.4f} {pv - bv:>+10.4f}")

        # vs Supervision 指标
        bt_sup = bt.get('supervision', {})
        pt_sup = pt.get('supervision', {})
        print("\n[vs Supervision Category]")
        for key, label in [('psnr', 'Supervision PSNR↑'),
                          ('lpips', 'Supervision LPIPS↓')]:
            bv = bt_sup.get(key, 0)
            pv = pt_sup.get(key, 0)
            print(f"{label:<25} {bv:>10.4f} {pv:>12.4f} {pv - bv:>+10.4f}")
    else:
        # 标准模式：原有显示方式
        print(f"{'指标':<20} {'Baseline':>10} {'Post-Defense':>12} {'Delta':>10}")
        print("-" * 55)

        # Target 指标（攻击后，质量口径）：PSNR↑ 越好，LPIPS↓ 越好
        bt = baseline_target or {}
        pt = postdef_target or {}
        for key, label in [('psnr', 'Target PSNR↑'),
                          ('lpips', 'Target LPIPS↓')]:
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

    # Attack 训练过程中（step-eval）最后一次的 masked PSNR/LPIPS，方便快速查看
    baseline_last = _last_attack_eval_entry(baseline_history)
    postdef_last = _last_attack_eval_entry(postdef_history)
    print("\n[Attack Step-Eval (masked)]")
    if baseline_last is not None:
        print(f"Baseline Attack: step={baseline_last.get('step')}, "
              f"masked_psnr={baseline_last.get('masked_psnr', 0):.2f}, "
              f"masked_lpips={baseline_last.get('masked_lpips', 0):.4f}")
    else:
        print("Baseline Attack: (无)")
    if postdef_last is not None:
        print(f"Post-Defense Attack: step={postdef_last.get('step')}, "
              f"masked_psnr={postdef_last.get('masked_psnr', 0):.2f}, "
              f"masked_lpips={postdef_last.get('masked_lpips', 0):.4f}")
    else:
        print("Post-Defense Attack: (无)")

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
