#!/usr/bin/env python3
"""
将OmniObject3D的blender_renders数据转换为LGM格式

输入格式:
  {object_id}/
    ├── 000.png - 023.png  (1024x1024 RGBA)
    └── transforms.json

输出格式:
  {object_id}/
    ├── rgb/
    │   ├── 000.png - 023.png  (512x512 RGBA)
    └── pose/
        ├── 000.txt - 023.txt  (4x4矩阵，空格分隔)
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

def convert_object(source_dir, target_dir, resize=512):
    """转换单个对象的数据"""

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # 创建目标目录
    rgb_dir = target_path / "rgb"
    pose_dir = target_path / "pose"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    # 读取transforms.json
    transforms_file = source_path / "transforms.json"
    with open(transforms_file, 'r') as f:
        transforms = json.load(f)

    print(f"转换 {source_path.name}: {len(transforms['frames'])} 个视图")

    # 处理每个视图
    for frame in transforms['frames']:
        file_path = frame['file_path']
        view_id = Path(file_path).stem  # 获取文件名（不含扩展名）

        # 1. 处理图像
        src_img_path = source_path / file_path
        dst_img_path = rgb_dir / f"{view_id}.png"

        if src_img_path.exists():
            # 读取并resize图像
            img = Image.open(src_img_path)
            if img.size != (resize, resize):
                img = img.resize((resize, resize), Image.LANCZOS)
            img.save(dst_img_path)
        else:
            print(f"  警告: 图像不存在 {src_img_path}")
            continue

        # 2. 处理位姿
        transform_matrix = np.array(frame['transform_matrix'])

        # 调整相机距离: 从1.2缩放到1.5
        # LGM使用cam_radius=1.5，而OmniObject3D使用1.2
        scale_factor = 1.5 / 1.2
        transform_matrix[:3, 3] *= scale_factor

        # 保存为文本文件
        pose_file = pose_dir / f"{view_id}.txt"
        with open(pose_file, 'w') as f:
            # 将4x4矩阵展平为16个数字，空格分隔
            f.write(' '.join(map(str, transform_matrix.flatten())))

    print(f"  完成: {len(transforms['frames'])} 个视图")
    return len(transforms['frames'])


def batch_convert(source_root, target_root, resize=512, max_objects=None):
    """批量转换多个对象"""

    source_root = Path(source_root)
    target_root = Path(target_root)

    # 查找所有对象目录
    object_dirs = sorted([d for d in source_root.iterdir() if d.is_dir()])

    if max_objects:
        object_dirs = object_dirs[:max_objects]

    print(f"找到 {len(object_dirs)} 个对象")
    print(f"目标目录: {target_root}")
    print(f"图像resize: {resize}x{resize}")
    print()

    total_views = 0
    for i, obj_dir in enumerate(object_dirs, 1):
        print(f"[{i}/{len(object_dirs)}] ", end='')
        target_dir = target_root / obj_dir.name
        try:
            views = convert_object(obj_dir, target_dir, resize)
            total_views += views
        except Exception as e:
            print(f"  错误: {e}")

    print(f"\n总计: {len(object_dirs)} 个对象, {total_views} 个视图")

    # 生成object_list.txt
    list_file = target_root / "object_list.txt"
    with open(list_file, 'w') as f:
        for obj_dir in object_dirs:
            f.write(f"{obj_dir.name}\n")
    print(f"已生成: {list_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='转换OmniObject3D数据为LGM格式')
    parser.add_argument('source', help='源目录 (解压后的数据)')
    parser.add_argument('target', help='目标目录')
    parser.add_argument('--resize', type=int, default=512, help='图像resize大小 (默认512)')
    parser.add_argument('--max', type=int, help='最多转换多少个对象 (用于测试)')

    args = parser.parse_args()

    batch_convert(args.source, args.target, args.resize, args.max)
