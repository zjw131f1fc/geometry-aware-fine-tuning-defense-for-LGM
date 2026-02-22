#!/usr/bin/env python3
"""分析渲染图像的像素值"""

import numpy as np
from PIL import Image
import sys

if len(sys.argv) < 2:
    print("用法: python analyze_image.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]
img = Image.open(img_path)
img_array = np.array(img)

print(f"图像形状: {img_array.shape}")
print(f"图像dtype: {img_array.dtype}")
print(f"图像范围: [{img_array.min()}, {img_array.max()}]")

# 分析上半部分（GT）和下半部分（Pred）
H, W = img_array.shape[:2]
half_H = H // 2

gt_part = img_array[:half_H, :, :]
pred_part = img_array[half_H:, :, :]

print(f"\n=== GT部分（上半部分）===")
print(f"形状: {gt_part.shape}")
print(f"均值: {gt_part.mean(axis=(0,1))}")
print(f"标准差: {gt_part.std(axis=(0,1))}")
print(f"最小值: {gt_part.min(axis=(0,1))}")
print(f"最大值: {gt_part.max(axis=(0,1))}")

print(f"\n=== Pred部分（下半部分）===")
print(f"形状: {pred_part.shape}")
print(f"均值: {pred_part.mean(axis=(0,1))}")
print(f"标准差: {pred_part.std(axis=(0,1))}")
print(f"最小值: {pred_part.min(axis=(0,1))}")
print(f"最大值: {pred_part.max(axis=(0,1))}")

# 统计Pred部分的像素值分布
print(f"\n=== Pred部分像素值分布 ===")
pred_flat = pred_part.reshape(-1, 3)
unique_vals, counts = np.unique(pred_flat[:, 0], return_counts=True)
print(f"R通道唯一值数量: {len(unique_vals)}")
print(f"最常见的5个R值:")
top5_idx = np.argsort(counts)[-5:][::-1]
for idx in top5_idx:
    val = unique_vals[idx]
    cnt = counts[idx]
    pct = cnt / len(pred_flat) * 100
    print(f"  {val}: {cnt} 像素 ({pct:.2f}%)")

# 检查是否有非白色像素（RGB < 250）
non_white_mask = (pred_part < 250).any(axis=2)
non_white_count = non_white_mask.sum()
total_pixels = pred_part.shape[0] * pred_part.shape[1]
print(f"\n非白色像素（RGB<250）: {non_white_count} / {total_pixels} ({non_white_count/total_pixels*100:.2f}%)")

# 如果有非白色像素，显示它们的位置和值
if non_white_count > 0:
    print(f"\n前10个非白色像素的位置和RGB值:")
    non_white_coords = np.argwhere(non_white_mask)[:10]
    for coord in non_white_coords:
        y, x = coord
        rgb = pred_part[y, x]
        print(f"  位置({y}, {x}): RGB={rgb}")
