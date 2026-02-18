#!/usr/bin/env python3
"""
测试 ObjaverseRenderedDataset 加载器
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import ObjaverseRenderedDataset
from torch.utils.data import DataLoader


def test_objaverse_dataset():
    """测试 Objaverse 数据集加载"""

    print("=" * 80)
    print("测试 ObjaverseRenderedDataset")
    print("=" * 80)

    # 创建数据集
    dataset = ObjaverseRenderedDataset(
        data_root='datas/objaverse_rendered',
        num_input_views=4,
        num_supervision_views=4,
        input_size=256,
        view_selector='orthogonal',
        max_samples=10,  # 只测试10个样本
        samples_per_object=1,
    )

    print(f"\n数据集大小: {len(dataset)}")

    # 测试加载第一个样本
    print("\n测试加载第一个样本...")
    sample = dataset[0]

    print(f"\n样本信息:")
    print(f"  UUID: {sample['uuid']}")
    print(f"  Input images shape: {sample['input_images'].shape}")
    print(f"  Supervision images shape: {sample['supervision_images'].shape}")
    print(f"  Supervision masks shape: {sample['supervision_masks'].shape}")
    print(f"  Input transforms shape: {sample['input_transforms'].shape}")
    print(f"  Supervision transforms shape: {sample['supervision_transforms'].shape}")

    # 测试 DataLoader
    print("\n测试 DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataloader))
    print(f"\nBatch 信息:")
    print(f"  Input images shape: {batch['input_images'].shape}")
    print(f"  Supervision images shape: {batch['supervision_images'].shape}")
    print(f"  Supervision masks shape: {batch['supervision_masks'].shape}")
    print(f"  UUIDs: {batch['uuid']}")

    print("\n" + "=" * 80)
    print("测试通过！")
    print("=" * 80)


if __name__ == '__main__':
    test_objaverse_dataset()
