#!/usr/bin/env python3
"""
找出GSO数据集中显著离群的物体

用法:
    python tools/find_gso_outliers.py --categories plant,shoe,bowl,dish,box
    python tools/find_gso_outliers.py --categories plant --threshold 0.6
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# HuggingFace 镜像（国内网络）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

GSO_DATA_ROOT = "datas/GSO/render_same_pose_all_50v_800_norm3.73"


def get_gso_objects_for_category(category: str, data_root: str = GSO_DATA_ROOT) -> list:
    """获取GSO某个类别下所有物体目录（sorted）"""
    render_dir = Path(data_root)
    if not render_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {render_dir}")

    objects = []
    for d in sorted(render_dir.iterdir()):
        if not d.is_dir() or d.name.startswith('_'):
            continue
        name = d.name
        # GSO使用 first_underscore 解析类别
        cat = name.split('_', 1)[0]
        if cat == category:
            objects.append(d)
    return objects


def load_representative_images(objects: list, view_indices: list = None) -> dict:
    """加载每个物体的代表视图图像"""
    if view_indices is None:
        view_indices = [0]

    result = {}
    for obj_dir in objects:
        images_dir = obj_dir / "render" / "images"
        if not images_dir.exists():
            continue

        imgs = []
        for vi in view_indices:
            img_path = images_dir / f"r_{vi}.png"
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            imgs.append(img)

        if imgs:
            result[obj_dir.name] = imgs

    return result


def load_clip_model(device: str = "cuda"):
    """加载 CLIP 模型"""
    from transformers import CLIPProcessor, CLIPModel

    print("加载 CLIP 模型...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device).eval()
    return model, processor


def compute_clip_embeddings(object_images: dict, clip_model, clip_processor, device: str = "cuda") -> tuple:
    """用 CLIP 提取每个物体的 embedding

    Returns:
        names: 物体名称列表
        embeddings: [N, D] numpy array
    """
    names = []
    embeddings = []

    print(f"提取 {len(object_images)} 个物体的 embedding...")
    with torch.no_grad():
        for obj_name, imgs in object_images.items():
            view_embeds = []
            for img in imgs:
                inputs = clip_processor(images=img, return_tensors="pt").to(device)
                embed = clip_model.get_image_features(**inputs)
                if not isinstance(embed, torch.Tensor):
                    embed = embed.pooler_output
                embed = embed / embed.norm(dim=-1, keepdim=True)
                view_embeds.append(embed.cpu())

            avg_embed = torch.stack(view_embeds).mean(dim=0)
            avg_embed = avg_embed / avg_embed.norm(dim=-1, keepdim=True)

            names.append(obj_name)
            embeddings.append(avg_embed.squeeze(0).numpy())

    embeddings = np.stack(embeddings)
    return names, embeddings


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """计算余弦相似度矩阵"""
    return embeddings @ embeddings.T


def find_outliers(sim_matrix: np.ndarray, names: list, threshold: float = 0.6) -> list:
    """找出离群物体

    策略：计算每个物体与其他所有物体的平均相似度，低于阈值的视为离群

    Args:
        sim_matrix: [N, N] 相似度矩阵
        names: 物体名称列表
        threshold: 平均相似度阈值，低于此值视为离群

    Returns:
        list of dict: [{"name": str, "avg_sim": float, "min_sim": float, "index": int}, ...]
    """
    n = len(names)
    outliers = []

    for i in range(n):
        # 计算与其他物体的相似度（排除自己）
        other_sims = [sim_matrix[i, j] for j in range(n) if j != i]
        avg_sim = np.mean(other_sims) if other_sims else 1.0
        min_sim = np.min(other_sims) if other_sims else 1.0
        max_sim = np.max(other_sims) if other_sims else 1.0

        if avg_sim < threshold:
            outliers.append({
                "name": names[i],
                "index": i,
                "avg_sim": float(avg_sim),
                "min_sim": float(min_sim),
                "max_sim": float(max_sim),
            })

    # 按平均相似度升序排序（最离群的在前）
    outliers.sort(key=lambda x: x["avg_sim"])
    return outliers


def analyze_category_outliers(category, clip_model, clip_processor, view_indices, device,
                               data_root, output_dir, threshold):
    """分析单个类别的离群物体"""
    objects = get_gso_objects_for_category(category, data_root)
    print(f"\n{'='*60}")
    print(f"类别 '{category}': {len(objects)} 个物体")
    print(f"{'='*60}")

    if len(objects) == 0:
        print(f"  未找到物体，跳过")
        return None

    cat_dir = os.path.join(output_dir, category)
    os.makedirs(cat_dir, exist_ok=True)

    object_images = load_representative_images(objects, view_indices)
    print(f"  成功加载 {len(object_images)} 个物体的图像")

    names, embeddings = compute_clip_embeddings(object_images, clip_model, clip_processor, device)
    sim_matrix = compute_similarity_matrix(embeddings)

    # 计算整体统计
    upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    overall_stats = {
        "mean": float(np.mean(upper_tri)),
        "std": float(np.std(upper_tri)),
        "min": float(np.min(upper_tri)),
        "max": float(np.max(upper_tri)),
    }

    print(f"\n  整体相似度统计:")
    print(f"    平均: {overall_stats['mean']:.4f} ± {overall_stats['std']:.4f}")
    print(f"    范围: [{overall_stats['min']:.4f}, {overall_stats['max']:.4f}]")

    # 找出离群物体
    outliers = find_outliers(sim_matrix, names, threshold)

    print(f"\n  离群物体 (平均相似度 < {threshold}): {len(outliers)} 个")
    if outliers:
        print(f"\n  {'索引':<6} {'平均相似度':<12} {'最小相似度':<12} {'最大相似度':<12} {'物体名称'}")
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*40}")
        for obj in outliers:
            print(f"  {obj['index']:<6} {obj['avg_sim']:<12.4f} {obj['min_sim']:<12.4f} "
                  f"{obj['max_sim']:<12.4f} {obj['name']}")

    # 保存结果
    result = {
        "category": category,
        "num_objects": len(names),
        "threshold": threshold,
        "overall_similarity": overall_stats,
        "outliers": outliers,
        "all_objects": [
            {
                "index": i,
                "name": names[i],
                "avg_sim": float(np.mean([sim_matrix[i, j] for j in range(len(names)) if j != i])),
            }
            for i in range(len(names))
        ]
    }

    json_path = os.path.join(cat_dir, "outliers.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {json_path}")

    return result, sim_matrix, names, outliers


def main():
    parser = argparse.ArgumentParser(description="找出GSO数据集中的离群物体")
    parser.add_argument("--categories", type=str, required=True,
                        help="类别名称，逗号分隔，如 'plant,shoe,bowl'")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="平均相似度阈值，低于此值视为离群 (default: 0.6)")
    parser.add_argument("--views", type=str, default="0,6,12",
                        help="使用的视图索引，逗号分隔 (default: 0,6,12)")
    parser.add_argument("--output_dir", type=str, default="output/gso_outliers",
                        help="输出根目录")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_root", type=str, default=GSO_DATA_ROOT)
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(',')]
    view_indices = [int(v) for v in args.views.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 CLIP 模型（只加载一次）
    clip_model, clip_processor = load_clip_model(args.device)

    # 逐类别分析
    all_results = {}
    for category in categories:
        result = analyze_category_outliers(
            category, clip_model, clip_processor, view_indices,
            args.device, args.data_root, args.output_dir, args.threshold
        )
        if result:
            all_results[category] = result[0]

    # 汇总报告
    print(f"\n\n{'#'*70}")
    print("# GSO 离群物体汇总")
    print(f"{'#'*70}")

    total_objects = sum(r["num_objects"] for r in all_results.values())
    total_outliers = sum(len(r["outliers"]) for r in all_results.values())

    print(f"\n总物体数: {total_objects}")
    print(f"总离群数: {total_outliers} ({total_outliers/total_objects*100:.1f}%)")
    print(f"\n各类别离群统计:")
    for cat, result in all_results.items():
        n_obj = result["num_objects"]
        n_out = len(result["outliers"])
        print(f"  {cat:10s}: {n_out:2d}/{n_obj:2d} ({n_out/n_obj*100:5.1f}%)  "
              f"平均相似度={result['overall_similarity']['mean']:.4f}")

    # 保存汇总
    summary_path = os.path.join(args.output_dir, "summary.json")
    summary = {
        "threshold": args.threshold,
        "total_objects": total_objects,
        "total_outliers": total_outliers,
        "categories": all_results,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()

