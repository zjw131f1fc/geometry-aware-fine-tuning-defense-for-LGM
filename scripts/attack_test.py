"""
攻击测试主脚本
"""

import sys
sys.path.append('/mnt/huangjiaxin/3d-defense/LGM')
sys.path.append('/mnt/huangjiaxin/3d-defense')

import os
import yaml
import argparse

# 设置HuggingFace镜像（用于加载预训练模型）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 解析命令行参数（在导入torch之前）
def parse_args_early():
    """早期解析参数以设置GPU"""
    parser = argparse.ArgumentParser(description='LGM LoRA攻击测试')
    parser.add_argument('--config', type=str,
                       default='configs/attack_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--gpu', type=str, default=None,
                       help='指定GPU ID，如 "0" 或 "0,1,2,3"（优先级高于配置文件）')
    args, _ = parser.parse_known_args()
    return args

# 早期解析参数
early_args = parse_args_early()

# 设置GPU（优先级：命令行 > 环境变量 > 配置文件）
if early_args.gpu is not None:
    # 命令行指定的GPU（最高优先级）
    os.environ["CUDA_VISIBLE_DEVICES"] = early_args.gpu
    print(f"[INFO] 使用命令行指定的GPU: {early_args.gpu}")
elif "CUDA_VISIBLE_DEVICES" not in os.environ:
    # 如果环境变量没有设置，尝试从配置文件读取
    try:
        with open(early_args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config['model'].get('gpu_ids') is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config['model']['gpu_ids'])
            print(f"[INFO] 使用配置文件中的GPU: {config['model']['gpu_ids']}")
    except:
        pass
else:
    print(f"[INFO] 使用环境变量中的GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

import torch
import numpy as np
from datetime import datetime
from pathlib import Path

from core.options import Options
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
    parser = argparse.ArgumentParser(description='LGM LoRA攻击测试')
    parser.add_argument('--config', type=str,
                       default='configs/attack_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--gpu', type=str, default=None,
                       help='指定GPU ID，如 "0" 或 "0,1,2,3"（优先级高于配置文件）')
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    args = parse_args()

    # 加载配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    print(f"[INFO] 加载配置文件: {config_path}")
    config = load_config(config_path)

    # 设置随机种子
    set_seed(config['misc']['seed'])

    # 创建工作目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    attack_scenario = config['attack']['scenario']
    workspace = os.path.join(config['misc']['workspace'], f"{attack_scenario}_{timestamp}")
    os.makedirs(workspace, exist_ok=True)
    print(f"[INFO] 工作目录: {workspace}")

    # 创建LGM配置
    model_size = config['model']['size']
    from core.options import config_defaults

    # 使用预定义配置
    if model_size not in config_defaults:
        raise ValueError(f"不支持的模型大小: {model_size}，可选: {list(config_defaults.keys())}")

    opt = config_defaults[model_size]
    print(f"[INFO] 使用 {model_size} 配置:")
    print(f"  - input_size: {opt.input_size}")
    print(f"  - splat_size: {opt.splat_size}")
    print(f"  - output_size: {opt.output_size}")
    print(f"  - up_channels: {opt.up_channels}")

    print("=" * 80)
    print("步骤1: 加载LGM模型")
    print("=" * 80)

    # 加载模型（使用 FP32 以获得更好的稳定性）
    model = load_lgm_model(
        opt=opt,
        resume_path=config['model']['resume'],
        device=config['model']['device'],
        dtype=torch.float32,  # 使用 FP32 训练，更稳定
    )

    # 根据训练模式决定是否应用LoRA
    training_mode = config['training']['mode']
    print(f"\n[INFO] 训练模式: {training_mode}")

    if training_mode == 'lora':
        print("[INFO] 应用 LoRA 微调")
        model = apply_lora(
            model=model,
            target_modules=config['lora']['target_modules'],
            r=config['lora']['r'],
            lora_alpha=config['lora']['alpha'],
            lora_dropout=config['lora']['dropout'],
        )
    elif training_mode == 'full':
        print("[INFO] 使用全量微调")
        print("[INFO] 所有参数将被训练")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] 总参数量: {total_params:,}")
        print(f"[INFO] 可训练参数量: {trainable_params:,}")
        print(f"[INFO] 可训练参数占比: {trainable_params / total_params * 100:.2f}%")
    else:
        raise ValueError(f"不支持的训练模式: {training_mode}，可选: lora, full")

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
        num_input_views=4,  # 输入视图数
        num_supervision_views=config['data'].get('num_supervision_views', 4),  # 监督视图数
        input_size=opt.input_size,
        fovy=opt.fovy,
        view_selector=config['data']['view_selector'],
        angle_offset=config['data']['angle_offset'],
        samples_per_object=config['data'].get('samples_per_object', 1),  # 多视角采样
        max_samples_per_category=config['data'].get('max_samples_per_category'),  # 每个类别最大样本数
    )

    print(f"[INFO] 数据加载器创建完成，共 {len(dataloader)} 个批次")
    print(f"[INFO] 视角选择策略: {config['data']['view_selector']}")
    print(f"[INFO] 角度偏移: {config['data']['angle_offset']}°")

    print("\n" + "=" * 80)
    print("步骤3: 创建攻击场景")
    print("=" * 80)

    # 创建攻击场景
    if attack_scenario == 'category_bias':
        scenario_config = config['attack']['category_bias']
        scenario = create_attack_scenario(
            'category_bias',
            category_pairs=scenario_config['category_pairs'],
        )
    else:  # malicious_content
        scenario_config = config['attack']['malicious_content']
        scenario = create_attack_scenario(
            'malicious_content',
            malicious_categories=scenario_config['malicious_categories'],
        )

    print(f"[INFO] 攻击场景: {scenario.name}")
    print(f"[INFO] 描述: {scenario.description}")

    print("\n" + "=" * 80)
    if training_mode == 'lora':
        print("步骤4: LoRA微调（攻击）")
    else:
        print("步骤4: 全量微调（攻击）")
    print("=" * 80)

    # 创建微调器
    finetuner = AutoFineTuner(
        model=model,
        device=config['model']['device'],
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        gradient_clip=config['training']['gradient_clip'],
        mixed_precision=False,  # 模型已经是FP16，不需要混合精度
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
    )

    print(f"[INFO] 梯度累计步数: {config['training'].get('gradient_accumulation_steps', 1)}")
    print(f"[INFO] 有效batch size: {config['training']['batch_size'] * config['training'].get('gradient_accumulation_steps', 1)}")

    # 训练
    num_epochs = config['training']['num_epochs']
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        avg_loss = finetuner.train_epoch(dataloader, epoch)

        print(f"[INFO] Epoch {epoch} 平均损失:")
        for k, v in avg_loss.items():
            print(f"  {k}: {v:.4f}")

        # 保存检查点
        if epoch % 5 == 0 or epoch == num_epochs:
            checkpoint_path = os.path.join(workspace, f"checkpoint_epoch{epoch}.pt")
            finetuner.save_checkpoint(checkpoint_path, epoch)

    print("\n" + "=" * 80)
    print("步骤5: 评估攻击效果")
    print("=" * 80)

    # 创建评估器
    evaluator = Evaluator(model=model, device=config['model']['device'])

    # 生成测试样本并评估攻击效果
    print("[INFO] 生成测试样本...")
    test_batch = next(iter(dataloader))
    test_images = test_batch['input_images'].to(config['model']['device'])

    # 生成Gaussian
    gaussians = evaluator.generate_gaussians(test_images)

    # 保存PLY文件
    num_samples = min(3, test_images.shape[0])
    for i in range(num_samples):
        ply_path = os.path.join(workspace, f"sample_{i}.ply")
        evaluator.save_ply(gaussians[i:i+1], ply_path)

    # 渲染360度视频
    for i in range(num_samples):
        video_path = os.path.join(workspace, f"sample_{i}_360.mp4")
        evaluator.render_360_video(gaussians[i:i+1], video_path)

    print("\n" + "=" * 80)
    print("攻击测试完成！")
    print("=" * 80)
    print(f"结果保存在: {workspace}")


if __name__ == '__main__':
    main()