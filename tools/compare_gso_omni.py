#!/usr/bin/env python3
"""
比较GSO和OmniObject3D物体的相似度，找出GSO中与Omni显著不同的物体

用法:
    python tools/compare_gso_omni.py --categories plant,shoe,bowl
    python tools/compare_gso_omni.py --categories plant --threshold 0.7
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
OMNI_DATA_ROOT = "datas/omniobject3d___OmniObject3D-New/raw/blender_renders"


def get_objects_for_category(category: str, data_root: str, parse_mode: str) -> list:
    """获取某个类别下所有物体目录（sorted）

    Args:
        category: 类别名称
        data_root: 数据根目录
        parse_mode: 'first_underscore' (GSO) 或 'last_underscore' (Omni)
    """
    render_dir = Path(data_root)
    if not render_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {render_dir}")

    objects = []
    for d in sorted(render_dir.iterdir()):
        if not d.is_dir() or d.name.startswith('_'):
            continue
        name = d.name

        if parse_mode == 'first_underscore':
            cat = name.split('_', 1)[0]
        else:  # last_underscore
            cat = name.rsplit('_', 1)[0]

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
    """用 CLIP 提取每个物体的 embedding"""
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


def compute_cross_similarity(gso_embeddings: np.ndarray, omni_embeddings: np.ndarray) -> np.ndarray:
    """计算GSO和Omni之间的交叉相似度矩阵

    Returns:
        [N_gso, N_omni] 相似度矩阵
    """
    return gso_embeddings @ omni_embeddings.T


def find_dissimilar_gso_objects(cross_sim_matrix: np.ndarray, gso_names: list,
                                 omni_names: list, threshold: float = 0.7) -> list:
    """找出GSO中与Omni显著不同的物体

    策略：计算每个GSO物体与所有Omni物体的最大相似度，低于阈值的视为不同

    Args:
        cross_sim_matrix: [N_gso, N_omni] 交叉相似度矩阵
        gso_names: GSO物体名称列表
        omni_names: Omni物体名称列表
        threshold: 最大相似度阈值，低于此值视为显著不同

    Returns:
        list of dict: [{"name": str, "max_sim": float, "avg_sim": float, ...}, ...]
    """
    dissimilar = []

    for i, gso_name in enumerate(gso_names):
        # 与所有Omni物体的相似度
        sims_to_omni = cross_sim_matrix[i, :]
        max_sim = float(np.max(sims_to_omni))
        avg_sim = float(np.mean(sims_to_omni))

        # 找出最相似的Omni物体
        most_similar_idx = int(np.argmax(sims_to_omni))
        most_similar_name = omni_names[most_similar_idx]

        if max_sim < threshold:
            dissimilar.append({
                "name": gso_name,
                "index": i,
                "max_sim_to_omni": max_sim,
                "avg_sim_to_omni": avg_sim,
                "most_similar_omni": most_similar_name,
                "most_similar_sim": max_sim,
            })

    # 按最大相似度升序排序（最不相似的在前）
    dissimilar.sort(key=lambda x: x["max_sim_to_omni"])
    return dissimilar


def analyze_category_cross_similarity(category, clip_model, clip_processor, view_indices,
                                       device, gso_root, omni_root, output_dir, threshold):
    """分析单个类别的跨数据集相似度"""
    print(f"\n{'='*60}")
    print(f"类别 '{category}'")
    print(f"{'='*60}")

    # 加载GSO物体
    gso_objects = get_objects_for_category(category, gso_root, 'first_underscore')
    print(f"  GSO: {len(gso_objects)} 个物体")

    # 加载Omni物体
    omni_objects = get_objects_for_category(category, omni_root, 'last_underscore')
    print(f"  Omni: {len(omni_objects)} 个物体")

    if len(gso_objects) == 0 or len(omni_objects) == 0:
        print(f"  数据不足，跳过")
        return None

    cat_dir = os.path.join(output_dir, category)
    os.makedirs(cat_dir, exist_ok=True)

    # 加载图像
    gso_images = load_representative_images(gso_objects, view_indices)
    omni_images = load_representative_images(omni_objects, view_indices)
    print(f"  成功加载 GSO={len(gso_images)}, Omni={len(omni_images)} 个物体的图像")

    # 提取embeddings
    gso_names, gso_embeddings = compute_clip_embeddings(gso_images, clip_model, clip_processor, device)
    omni_names, omni_embeddings = compute_clip_embeddings(omni_images, clip_model, clip_processor, device)

    # 计算交叉相似度
    cross_sim = compute_cross_similarity(gso_embeddings, omni_embeddings)

    # 统计
    print(f"\n  交叉相似度统计 (GSO vs Omni):")
    print(f"    平均: {np.mean(cross_sim):.4f} ± {np.std(cross_sim):.4f}")
    print(f"    范围: [{np.min(cross_sim):.4f}, {np.max(cross_sim):.4f}]")

    # 找出不相似的GSO物体
    dissimilar = find_dissimilar_gso_objects(cross_sim, gso_names, omni_names, threshold)

    print(f"\n  GSO中与Omni显著不同的物体 (最大相似度 < {threshold}): {len(dissimilar)} 个")
    if dissimilar:
        print(f"\n  {'索引':<6} {'最大相似度':<12} {'平均相似度':<12} {'最相似Omni物体':<40} {'GSO物体名称'}")
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*40} {'-'*50}")
        for obj in dissimilar:
            omni_short = obj['most_similar_omni'].rsplit('_', 1)[-1][:38]
            print(f"  {obj['index']:<6} {obj['max_sim_to_omni']:<12.4f} {obj['avg_sim_to_omni']:<12.4f} "
                  f"{omni_short:<40} {obj['name']}")

    # 保存结果
    result = {
        "category": category,
        "num_gso_objects": len(gso_names),
        "num_omni_objects": len(omni_names),
        "threshold": threshold,
        "cross_similarity_stats": {
            "mean": float(np.mean(cross_sim)),
            "std": float(np.std(cross_sim)),
            "min": float(np.min(cross_sim)),
            "max": float(np.max(cross_sim)),
        },
        "dissimilar_gso_objects": dissimilar,
        "all_gso_objects": [
            {
                "index": i,
                "name": gso_names[i],
                "max_sim_to_omni": float(np.max(cross_sim[i, :])),
                "avg_sim_to_omni": float(np.mean(cross_sim[i, :])),
            }
            for i in range(len(gso_names))
        ]
    }

    json_path = os.path.join(cat_dir, "cross_similarity.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {json_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="比较GSO和OmniObject3D物体的相似度")
    parser.add_argument("--categories", type=str, required=True,
                        help="类别名称，逗号分隔，如 'plant,shoe,bowl'")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="最大相似度阈值，低于此值视为显著不同 (default: 0.7)")
    parser.add_argument("--views", type=str, default="0,6,12",
                        help="使用的视图索引，逗号分隔 (default: 0,6,12)")
    parser.add_argument("--output_dir", type=str, default="output/gso_omni_comparison",
                        help="输出根目录")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gso_root", type=str, default=GSO_DATA_ROOT)
    parser.add_argument("--omni_root", type=str, default=OMNI_DATA_ROOT)
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(',')]
    view_indices = [int(v) for v in args.views.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 CLIP 模型（只加载一次）
    clip_model, clip_processor = load_clip_model(args.device)

    # 逐类别分析
    all_results = {}
    for category in categories:
        result = analyze_category_cross_similarity(
            category, clip_model, clip_processor, view_indices,
            args.device, args.gso_root, args.omni_root, args.output_dir, args.threshold
        )
        if result:
            all_results[category] = result

    # 汇总报告
    print(f"\n\n{'#'*70}")
    print("# GSO vs Omni 跨数据集相似度汇总")
    print(f"{'#'*70}")

    total_gso = sum(r["num_gso_objects"] for r in all_results.values())
    total_dissimilar = sum(len(r["dissimilar_gso_objects"]) for r in all_results.values())

    print(f"\n总GSO物体数: {total_gso}")
    print(f"显著不同物体数: {total_dissimilar} ({total_dissimilar/total_gso*100:.1f}%)")
    print(f"\n各类别统计:")
    for cat, result in all_results.items():
        n_gso = result["num_gso_objects"]
        n_omni = result["num_omni_objects"]
        n_dissim = len(result["dissimilar_gso_objects"])
        stats = result["cross_similarity_stats"]
        print(f"  {cat:10s}: GSO={n_gso:2d}, Omni={n_omni:2d}, 不同={n_dissim:2d} ({n_dissim/n_gso*100:5.1f}%)  "
              f"交叉相似度={stats['mean']:.4f}")

    # 保存汇总
    summary_path = os.path.join(args.output_dir, "summary.json")
    summary = {
        "threshold": args.threshold,
        "total_gso_objects": total_gso,
        "total_dissimilar": total_dissimilar,
        "categories": all_results,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
