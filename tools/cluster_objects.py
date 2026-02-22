"""
类别内物体相似度计算与聚类工具

用法:
    # 单类别
    python tools/cluster_objects.py --categories knife
    # 多类别批量分析
    python tools/cluster_objects.py --categories knife,broccoli,conch,garlic,durian
    # 指定阈值
    python tools/cluster_objects.py --categories knife --threshold 0.85
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

DATA_ROOT = "datas/omniobject3d___OmniObject3D-New/raw/blender_renders"


def get_objects_for_category(category: str, data_root: str = DATA_ROOT) -> list:
    """获取某个类别下所有物体目录（sorted，与 dataset.py 一致）"""
    render_dir = Path(data_root)
    if not render_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {render_dir}")

    objects = []
    for d in sorted(render_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        cat = name.rsplit('_', 1)[0]
        if cat == category:
            objects.append(d)
    return objects


def list_available_categories(data_root: str = DATA_ROOT) -> list:
    """列出所有可用类别"""
    render_dir = Path(data_root)
    cats = set()
    for d in render_dir.iterdir():
        if d.is_dir() and '_' in d.name:
            cats.add(d.name.rsplit('_', 1)[0])
    return sorted(cats)


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
    """加载 CLIP 模型（只加载一次，多类别共享）"""
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


def cluster_objects(sim_matrix: np.ndarray, names: list, num_clusters: int = None, threshold: float = None) -> list:
    """层次聚类

    Returns:
        clusters: list of lists，每个子列表是一个簇的物体名称
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.maximum(dist_matrix, 0)

    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='average')

    if num_clusters is not None:
        labels = fcluster(Z, t=num_clusters, criterion='maxclust')
    elif threshold is not None:
        labels = fcluster(Z, t=1.0 - threshold, criterion='distance')
    else:
        dists = Z[:, 2]
        gaps = np.diff(dists)
        if len(gaps) > 0:
            cut_idx = np.argmax(gaps) + 1
            cut_dist = dists[cut_idx]
            labels = fcluster(Z, t=cut_dist, criterion='distance')
        else:
            labels = np.ones(len(names), dtype=int)

    clusters = {}
    for name, label in zip(names, labels):
        clusters.setdefault(int(label), []).append(name)

    return [members for _, members in sorted(clusters.items())]


def save_similarity_heatmap(sim_matrix, names, save_path, clusters=None):
    """保存相似度热力图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if clusters:
        ordered_names = []
        for c in clusters:
            ordered_names.extend(c)
        order = [names.index(n) for n in ordered_names]
        sim_matrix = sim_matrix[np.ix_(order, order)]
        names = ordered_names

    n = len(names)
    fig_size = max(8, n * 0.3)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=0.5, vmax=1.0)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    short_names = [f"{i}:{n.rsplit('_', 1)[-1]}" for i, n in enumerate(names)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=90, fontsize=max(4, 8 - n // 10))
    ax.set_yticklabels(short_names, fontsize=max(4, 8 - n // 10))

    if clusters:
        pos = 0
        for c in clusters[:-1]:
            pos += len(c)
            ax.axhline(y=pos - 0.5, color='black', linewidth=2)
            ax.axvline(x=pos - 0.5, color='black', linewidth=2)

    ax.set_title(f'Object Similarity ({n} objects)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"热力图已保存: {save_path}")


def save_cluster_grid(object_images, clusters, save_path, sorted_names=None, max_per_cluster=8):
    """保存聚类结果的缩略图网格"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    num_clusters = len(clusters)
    max_cols = min(max(len(c) for c in clusters), max_per_cluster)

    fig, axes = plt.subplots(num_clusters, max_cols, figsize=(max_cols * 2, num_clusters * 2.2), squeeze=False)

    for i, cluster in enumerate(clusters):
        for j in range(max_cols):
            ax = axes[i][j]
            ax.axis('off')
            if j < len(cluster):
                name = cluster[j]
                if name in object_images:
                    ax.imshow(object_images[name][0])
                    sorted_idx = sorted_names.index(name) if sorted_names else 0
                    short = name.rsplit('_', 1)[-1]
                    ax.set_title(f"{sorted_idx}:{short}", fontsize=8)
        axes[i][0].set_ylabel(f'Cluster {i+1}\n({len(cluster)})', fontsize=10, rotation=0, labelpad=50)

    plt.suptitle(f'{num_clusters} Clusters', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"聚类网格已保存: {save_path}")


def analyze_category(category, clip_model, clip_processor, view_indices, device, data_root, output_dir,
                     num_clusters=None, threshold=None):
    """分析单个类别，返回聚类结果"""
    objects = get_objects_for_category(category, data_root)
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

    clusters = cluster_objects(sim_matrix, names, num_clusters=num_clusters, threshold=threshold)

    name_to_sorted_idx = {name: idx for idx, name in enumerate(names)}

    # 打印结果
    print(f"\n  聚类结果: {len(clusters)} 个簇")
    for i, cluster in enumerate(clusters):
        avg_sim = np.mean([
            sim_matrix[names.index(a), names.index(b)]
            for a in cluster for b in cluster if a != b
        ]) if len(cluster) > 1 else 1.0
        indices = sorted([name_to_sorted_idx[n] for n in cluster])
        print(f"\n  Cluster {i+1} ({len(cluster)} 个, 簇内相似度={avg_sim:.3f}):")
        print(f"    排序索引: {indices}")
        for name in cluster:
            print(f"    [{name_to_sorted_idx[name]:3d}] {name}")

    # 保存可视化
    save_similarity_heatmap(sim_matrix, names, os.path.join(cat_dir, "similarity_heatmap.png"), clusters=clusters)
    save_cluster_grid(object_images, clusters, os.path.join(cat_dir, "cluster_grid.png"), sorted_names=names)

    # 构建结果
    result = {
        "category": category,
        "num_objects": len(names),
        "num_clusters": len(clusters),
        "sorted_index_mapping": {name: idx for idx, name in enumerate(names)},
        "clusters": {},
    }
    for i, members in enumerate(clusters):
        indices = sorted([name_to_sorted_idx[n] for n in members])
        avg_sim = np.mean([
            sim_matrix[names.index(a), names.index(b)]
            for a in members for b in members if a != b
        ]) if len(members) > 1 else 1.0
        result["clusters"][f"cluster_{i+1}"] = {
            "objects": members,
            "sorted_indices": indices,
            "avg_similarity": round(float(avg_sim), 4),
            "size": len(members),
        }

    result["similarity_stats"] = {
        "mean": round(float(np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])), 4),
        "min": round(float(np.min(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])), 4),
        "max": round(float(np.max(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])), 4),
    }

    json_path = os.path.join(cat_dir, "clusters.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {cat_dir}/")

    return result, sim_matrix, names, clusters


def recommend_defense_objects(sim_matrix: np.ndarray, names: list, clusters: list, count: int) -> dict:
    """从类别中推荐 defense 用的物体

    策略：覆盖所有子类型。先从每个簇选最中心的物体（保证每个子类型至少1个代表），
    剩余名额按簇大小比例分配，从每个簇中继续选最中心的。

    Args:
        sim_matrix: [N, N] 相似度矩阵
        names: 物体名称列表
        clusters: 聚类结果，list of lists（每个子列表是一个簇的物体名称）
        count: 要选多少个

    Returns:
        {"indices": [...], "objects": [...], "cluster_coverage": str}
    """
    n = len(names)
    count = min(count, n)
    name_to_idx = {name: i for i, name in enumerate(names)}

    # 对每个簇，按簇内中心度排序
    cluster_ranked = []
    for cluster_names in clusters:
        idxs = [name_to_idx[nm] for nm in cluster_names]
        # 簇内中心度：与簇内其他物体的平均相似度
        ranked = []
        for idx in idxs:
            if len(idxs) > 1:
                sims = [sim_matrix[idx, other] for other in idxs if other != idx]
                ranked.append((idx, np.mean(sims)))
            else:
                ranked.append((idx, 1.0))
        ranked.sort(key=lambda x: x[1], reverse=True)
        cluster_ranked.append(ranked)

    # 第一轮：每个簇选 1 个最中心的（保证覆盖）
    selected = []
    cluster_used = [0] * len(cluster_ranked)
    for ci, ranked in enumerate(cluster_ranked):
        if len(selected) < count and ranked:
            selected.append(ranked[0][0])
            cluster_used[ci] = 1

    # 第二轮：剩余名额按簇大小比例分配
    remaining_count = count - len(selected)
    if remaining_count > 0:
        cluster_sizes = [len(cr) for cr in cluster_ranked]
        total_remaining = sum(max(0, s - u) for s, u in zip(cluster_sizes, cluster_used))

        if total_remaining > 0:
            # 按簇剩余大小比例分配
            allocations = []
            for ci, (cr, used) in enumerate(zip(cluster_ranked, cluster_used)):
                avail = len(cr) - used
                share = avail / total_remaining * remaining_count
                allocations.append((ci, share, avail))

            # 先给每个簇分配 floor(share)，剩余按小数部分排序分配
            alloc_int = [(ci, int(share), avail) for ci, share, avail in allocations]
            leftover = remaining_count - sum(a for _, a, _ in alloc_int)
            # 按小数部分降序
            frac_order = sorted(allocations, key=lambda x: x[1] - int(x[1]), reverse=True)
            extra = {ci: 0 for ci, _, _ in allocations}
            for ci, share, avail in frac_order:
                if leftover <= 0:
                    break
                can_add = min(1, avail - int(share))
                if can_add > 0:
                    extra[ci] = 1
                    leftover -= 1

            for ci, base, avail in alloc_int:
                to_pick = min(base + extra[ci], avail)
                start = cluster_used[ci]
                for j in range(start, start + to_pick):
                    if j < len(cluster_ranked[ci]):
                        idx = cluster_ranked[ci][j][0]
                        if idx not in selected:
                            selected.append(idx)

    # 如果还没选够（舍入误差），从所有簇中继续选
    all_remaining = []
    for ci, ranked in enumerate(cluster_ranked):
        for idx, score in ranked:
            if idx not in selected:
                all_remaining.append((idx, score))
    all_remaining.sort(key=lambda x: x[1], reverse=True)
    for idx, _ in all_remaining:
        if len(selected) >= count:
            break
        selected.append(idx)

    selected.sort()
    defense_names = [names[i] for i in selected]

    # 统计覆盖情况
    coverage_parts = []
    for ci, cluster_names in enumerate(clusters):
        cluster_set = set(name_to_idx[nm] for nm in cluster_names)
        picked = [i for i in selected if i in cluster_set]
        coverage_parts.append(f"簇{ci+1}:{len(picked)}/{len(cluster_names)}")

    # 计算质量指标
    n = len(names)
    selected_set = set(selected)
    unselected = [i for i in range(n) if i not in selected_set]

    # coverage: 每个未选中物体到最近 defense 物体的相似度
    if unselected and selected:
        nearest_sims = []
        for u in unselected:
            max_sim = max(sim_matrix[u, s] for s in selected)
            nearest_sims.append(max_sim)
        avg_coverage = float(np.mean(nearest_sims))
        min_coverage = float(np.min(nearest_sims))
    else:
        avg_coverage = 1.0
        min_coverage = 1.0

    # diversity: defense 集内部平均两两距离
    if len(selected) > 1:
        pairwise_sims = [sim_matrix[a, b] for a in selected for b in selected if a != b]
        diversity = 1.0 - float(np.mean(pairwise_sims))
    else:
        diversity = 0.0

    return {
        "indices": selected,
        "objects": defense_names,
        "cluster_coverage": ", ".join(coverage_parts),
        "avg_coverage": round(avg_coverage, 4),
        "min_coverage": round(min_coverage, 4),
        "diversity": round(diversity, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="类别内物体相似度聚类")
    parser.add_argument("--categories", type=str, required=True,
                        help="类别名称，逗号分隔，如 'knife,broccoli,conch'")
    parser.add_argument("--defense_counts", type=str, default="2,3,5,8",
                        help="要对比的 defense 物体数量，逗号分隔，如 '2,3,5,8'")
    parser.add_argument("--num_clusters", type=int, default=None,
                        help="指定聚类数（与 --threshold 二选一）")
    parser.add_argument("--threshold", type=float, default=None,
                        help="相似度阈值（0-1），低于此值分到不同簇")
    parser.add_argument("--views", type=str, default="0",
                        help="使用的视图索引，逗号分隔，如 '0,6,12,18'")
    parser.add_argument("--output_dir", type=str, default="output/cluster",
                        help="输出根目录")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(',')]
    defense_counts = [int(c) for c in args.defense_counts.split(',')]
    view_indices = [int(v) for v in args.views.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 CLIP 模型（只加载一次）
    clip_model, clip_processor = load_clip_model(args.device)

    # 逐类别分析
    category_data = {}  # {cat: (result, sim_matrix, names, clusters)}
    for category in categories:
        result, sim_matrix, names, clusters = analyze_category(
            category, clip_model, clip_processor, view_indices,
            args.device, args.data_root, args.output_dir,
            num_clusters=args.num_clusters, threshold=args.threshold,
        )
        if result:
            category_data[category] = (result, sim_matrix, names, clusters)

    # 对每个 defense_count 生成推荐
    print(f"\n\n{'#'*70}")
    print("# Defense 推荐对比（不同 defense_count）")
    print(f"{'#'*70}")

    all_count_results = {}
    for dc in defense_counts:
        recs = {}
        for cat, (result, sim_matrix, names, clusters) in category_data.items():
            recs[cat] = recommend_defense_objects(sim_matrix, names, clusters, dc)
        all_count_results[dc] = recs

        print(f"\n--- defense_count = {dc} ---")
        print("  object_split:")
        for cat, rec in recs.items():
            print(f"    {cat}: {rec['indices']}  # 覆盖=[{rec['cluster_coverage']}]")
        print(f"  attack_samples_per_category: 15")
        # 指标汇总
        cats = list(recs.keys())
        avg_cov = np.mean([recs[c]["avg_coverage"] for c in cats])
        min_cov = min(recs[c]["min_coverage"] for c in cats)
        avg_div = np.mean([recs[c]["diversity"] for c in cats])
        print(f"  [指标] 平均覆盖={avg_cov:.4f}  最差覆盖={min_cov:.4f}  多样性={avg_div:.4f}")

    # 指标对比表
    print(f"\n\n{'='*70}")
    print("指标对比表（覆盖度越高越好，最差覆盖越高越安全，多样性越高越好）")
    print(f"{'='*70}")
    # 表头
    header = f"{'count':>5} | {'平均覆盖':>8} | {'最差覆盖':>8} | {'多样性':>8}"
    for cat in category_data:
        header += f" | {cat[:8]:>8}"
    print(header)
    print("-" * len(header))
    for dc in defense_counts:
        recs = all_count_results[dc]
        cats = list(recs.keys())
        avg_cov = np.mean([recs[c]["avg_coverage"] for c in cats])
        min_cov = min(recs[c]["min_coverage"] for c in cats)
        avg_div = np.mean([recs[c]["diversity"] for c in cats])
        row = f"{dc:>5} | {avg_cov:>8.4f} | {min_cov:>8.4f} | {avg_div:>8.4f}"
        for cat in category_data:
            row += f" | {recs[cat]['min_coverage']:>8.4f}"
        print(row)

    # 保存汇总 JSON
    summary_path = os.path.join(args.output_dir, "summary.json")
    summary = {}
    for dc, recs in all_count_results.items():
        summary[f"defense_count_{dc}"] = {
            "config_yaml_snippet": {
                "object_split": {cat: rec["indices"] for cat, rec in recs.items()},
                "attack_samples_per_category": 15,
            },
            "details": {
                cat: {
                    "defense_indices": rec["indices"],
                    "defense_objects": rec["objects"],
                    "cluster_coverage": rec["cluster_coverage"],
                    "avg_coverage": rec["avg_coverage"],
                    "min_coverage": rec["min_coverage"],
                    "diversity": rec["diversity"],
                }
                for cat, rec in recs.items()
            },
        }
    # 类别基本信息
    summary["category_info"] = {
        cat: {
            "num_objects": result["num_objects"],
            "num_clusters": result["num_clusters"],
            "similarity_stats": result["similarity_stats"],
        }
        for cat, (result, _, _, _) in category_data.items()
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()