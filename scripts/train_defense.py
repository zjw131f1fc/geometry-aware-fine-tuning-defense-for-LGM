#!/usr/bin/env python3
"""
GeoTrap 防御训练脚本

使用 DefenseTrainer 进行防御训练的示例脚本
"""

import argparse
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_core import ConfigManager
from training import DefenseTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='GeoTrap 防御训练')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--gpu', type=int, default=0,
                        help='使用的 GPU 编号')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='训练轮数（覆盖配置文件）')
    parser.add_argument('--target_layers', type=str, default=None,
                        help='要微调的敏感层（逗号分隔，如 "conv.weight,unet.conv_in.weight"）')
    parser.add_argument('--output_dir', type=str, default='output/defense_training',
                        help='输出目录')
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # 加载配置
    print("=" * 80)
    print("加载配置...")
    print("=" * 80)
    config_mgr = ConfigManager(args.config)
    config = config_mgr.config

    # 覆盖配置
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs

    # 解析敏感层
    target_layers = None
    if args.target_layers:
        target_layers = [layer.strip() for layer in args.target_layers.split(',')]
    elif 'target_layers' in config.get('defense', {}):
        target_layers = config['defense']['target_layers']

    print(f"\n配置信息:")
    print(f"  - 训练轮数: {config['training']['num_epochs']}")
    print(f"  - 学习率: {config['training']['lr']}")
    print(f"  - Batch Size: {config['training']['batch_size']}")
    print(f"  - 敏感层: {target_layers if target_layers else '全部层'}")

    # 创建防御训练器
    trainer = DefenseTrainer(config)

    # 设置训练器
    trainer.setup(device='cuda', target_layers=target_layers)

    # 开始训练
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_dir=args.output_dir,
        validate_every=1
    )

    print(f"\n训练完成！模型保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
