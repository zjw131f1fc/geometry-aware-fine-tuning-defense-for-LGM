"""
Debug script: 测试所有翻转组合，找出让所有相机都朝向原点的最佳方案
"""
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ObjaverseRenderedDataset


def test_flip_combination(cam_poses_original, flip_name, flip_axes):
    """
    测试一种翻转组合

    Args:
        cam_poses_original: 原始相机姿态 [N, 4, 4]
        flip_name: 翻转名称（如 "Y+Z"）
        flip_axes: 要翻转的轴列表（如 [1, 2] 表示翻转 Y 和 Z）

    Returns:
        dict: 包含测试结果的字典
    """
    cam_poses = cam_poses_original.clone()

    # 应用翻转
    for axis in flip_axes:
        cam_poses[:, :3, axis] *= -1

    # 旋转对齐：使 camera 0 的旋转矩阵变为单位矩阵
    R_align = torch.inverse(cam_poses[0, :3, :3])
    for i in range(cam_poses.shape[0]):
        cam_poses[i, :3, 3] = R_align @ cam_poses[i, :3, 3]
        cam_poses[i, :3, :3] = R_align @ cam_poses[i, :3, :3]

    # 检查所有相机
    results = {
        'flip_name': flip_name,
        'cameras': [],
        'looking_away_count': 0,
        'avg_dot_product': 0.0,
    }

    origin = torch.tensor([0.0, 0.0, 0.0])
    dot_products = []

    for i in range(cam_poses.shape[0]):
        pos = cam_poses[i, :3, 3]
        R = cam_poses[i, :3, :3]

        # OpenGL: 相机看向 -Z 轴
        forward = -R[:, 2]  # -Z 列

        # 从相机到原点的方向
        to_origin = origin - pos
        to_origin_norm = to_origin / torch.norm(to_origin)

        # 点积：>0 表示朝向原点，<0 表示背对原点
        dot = torch.dot(forward, to_origin_norm).item()
        dot_products.append(dot)

        looking_away = dot < 0
        if looking_away:
            results['looking_away_count'] += 1

        results['cameras'].append({
            'id': i,
            'pos': pos.numpy(),
            'forward': forward.numpy(),
            'dot_product': dot,
            'looking_away': looking_away,
        })

    results['avg_dot_product'] = np.mean(dot_products)
    results['min_dot_product'] = np.min(dot_products)

    return results


def main():
    print("=" * 80)
    print("测试所有翻转组合")
    print("=" * 80)

    # 创建数据集（只加载一个样本）
    dataset = ObjaverseRenderedDataset(
        data_root='./datas/objaverse_rendered',
        input_size=256,
        fovy=49.1,
    )

    if len(dataset) == 0:
        print("错误：数据集为空")
        return

    # 加载第一个样本
    print(f"\n加载样本 0...")
    sample = dataset[0]

    # 重新加载相机姿态（不经过归一化）
    # 我们需要手动复制数据集的加载逻辑，但跳过归一化步骤
    import json
    from pathlib import Path

    data_dir = Path(dataset.data_root)
    obj_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    obj_dir = obj_dirs[0]

    # 加载 transforms.json（在 render 子目录下）
    transforms_file = obj_dir / 'render' / 'transforms.json'
    with open(transforms_file, 'r') as f:
        transforms_data = json.load(f)

    frames = transforms_data['frames']
    cam_poses = []

    for frame in frames:
        c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)

        # 除以 scale
        scale = frame.get('scale', 1.0)
        c2w[:3, :] /= scale

        # Blender+OpenCV → OpenGL 坐标转换
        c2w[1] *= -1
        c2w[[1, 2]] = c2w[[2, 1]]
        c2w[:3, 1:3] *= -1

        cam_poses.append(c2w)

    cam_poses = torch.stack(cam_poses, dim=0)

    # SVD 正交化
    for i in range(cam_poses.shape[0]):
        R = cam_poses[i, :3, :3]
        U, _, Vt = torch.linalg.svd(R)
        cam_poses[i, :3, :3] = U @ Vt

    # 单独缩放每个相机到半径 1.5
    target_radius = 1.5
    for i in range(cam_poses.shape[0]):
        dist = torch.norm(cam_poses[i, :3, 3])
        if dist > 1e-6:
            cam_poses[i, :3, 3] *= (target_radius / dist)

    print(f"加载了 {cam_poses.shape[0]} 个相机")
    print(f"所有相机已缩放到半径 {target_radius}")

    # 测试所有有效的翻转组合（det=+1）
    flip_combinations = [
        ("无翻转", []),
        ("X+Y", [0, 1]),
        ("X+Z", [0, 2]),
        ("Y+Z", [1, 2]),
    ]

    all_results = []

    for flip_name, flip_axes in flip_combinations:
        print(f"\n{'=' * 80}")
        print(f"测试翻转组合: {flip_name}")
        print(f"{'=' * 80}")

        results = test_flip_combination(cam_poses, flip_name, flip_axes)
        all_results.append(results)

        print(f"\n统计:")
        print(f"  背对原点的相机数: {results['looking_away_count']} / {len(results['cameras'])}")
        print(f"  平均点积: {results['avg_dot_product']:.4f}")
        print(f"  最小点积: {results['min_dot_product']:.4f}")

        print(f"\n相机详情:")
        for cam in results['cameras']:
            status = "❌ 背对" if cam['looking_away'] else "✓ 朝向"
            print(f"  Camera {cam['id']}: {status}, dot={cam['dot_product']:+.4f}, "
                  f"pos=[{cam['pos'][0]:.3f},{cam['pos'][1]:.3f},{cam['pos'][2]:.3f}], "
                  f"forward=[{cam['forward'][0]:.3f},{cam['forward'][1]:.3f},{cam['forward'][2]:.3f}]")

    # 总结
    print(f"\n{'=' * 80}")
    print("总结")
    print(f"{'=' * 80}")

    # 按背对相机数排序
    all_results.sort(key=lambda x: (x['looking_away_count'], -x['avg_dot_product']))

    print(f"\n排名（按背对相机数，然后按平均点积）:")
    for i, results in enumerate(all_results, 1):
        print(f"{i}. {results['flip_name']:8s}: "
              f"背对={results['looking_away_count']}, "
              f"平均点积={results['avg_dot_product']:+.4f}, "
              f"最小点积={results['min_dot_product']:+.4f}")

    best = all_results[0]
    print(f"\n推荐方案: {best['flip_name']}")
    if best['looking_away_count'] == 0:
        print("  ✓ 所有相机都朝向原点！")
    else:
        print(f"  ⚠ 仍有 {best['looking_away_count']} 个相机背对原点")


if __name__ == '__main__':
    main()
