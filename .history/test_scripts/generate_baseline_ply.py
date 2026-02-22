"""
生成基线 PLY 文件（训练前）
"""

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from project_core import ConfigManager
from models import ModelManager
from data import DataManager
from evaluation import Evaluator


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--gpu', type=str, default='7')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='output/baseline')
    args = parser.parse_args()

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda'

    print(f"\n生成基线 PLY 文件...")
    print(f"输出目录: {args.output_dir}\n")

    # 加载配置
    config_mgr = ConfigManager(args.config)
    config = config_mgr.config

    # 加载模型（原始预训练模型）
    print("[1/3] 加载预训练模型...")
    model_mgr = ModelManager(config)
    model_mgr.load_model(device=device, dtype=torch.float32)
    model = model_mgr.model

    # 加载数据
    print("[2/3] 加载数据...")
    data_mgr = DataManager(config, model_mgr.opt)
    data_mgr.setup_dataloaders(train=True, val=False)
    data_loader = data_mgr.train_loader

    # 生成 PLY 文件
    print(f"[3/3] 生成 {args.num_samples} 个 PLY 文件...")
    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = Evaluator(model, device)
    sample_count = 0

    for batch in data_loader:
        if sample_count >= args.num_samples:
            break

        input_images = batch['input_images'].to(device)
        gaussians = evaluator.generate_gaussians(input_images)

        B = gaussians.shape[0]
        for b in range(B):
            if sample_count >= args.num_samples:
                break

            ply_path = os.path.join(args.output_dir, f'sample_{sample_count}_before.ply')
            evaluator.save_ply(gaussians[b:b+1], ply_path)
            sample_count += 1

    print(f"\n完成！生成了 {sample_count} 个 PLY 文件。\n")


if __name__ == '__main__':
    main()
