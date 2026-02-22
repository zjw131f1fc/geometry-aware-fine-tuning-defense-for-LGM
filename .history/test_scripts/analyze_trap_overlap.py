"""
分析每种 trap loss 的 ratio>1 层，并找出两两组合的重叠层。

用法：
    python scripts/analyze_trap_overlap.py [--input path/to/sensitivity_results_trap.json]
"""

import json
import argparse
from itertools import combinations
from pathlib import Path


def build_combo_rankings(data):
    """
    为每种两两组合构建层排名（按几何平均 ratio 降序）。

    只保留两种 trap 都 ratio>1 的层。

    Returns:
        {
            "position+scale": [{"block": ..., "geo_mean": ..., "ratios": {...}}, ...],
            ...
        }
    """
    trap_names = list(data.keys())

    # 每种 trap 的 ratio>1 层
    trap_layer_ratios = {}
    for trap in trap_names:
        layers = {}
        for entry in data[trap]['differential']:
            if entry['ratio'] > 1.0:
                layers[entry['block']] = entry['ratio']
        trap_layer_ratios[trap] = layers

    rankings = {}
    for t1, t2 in combinations(trap_names, 2):
        key = f"{t1}+{t2}"
        overlap = set(trap_layer_ratios[t1].keys()) & set(trap_layer_ratios[t2].keys())
        items = []
        for block in overlap:
            r1 = trap_layer_ratios[t1][block]
            r2 = trap_layer_ratios[t2][block]
            geo_mean = (r1 * r2) ** 0.5
            items.append({
                "block": block,
                "geo_mean": round(geo_mean, 4),
                "ratios": {t1: round(r1, 4), t2: round(r2, 4)},
            })
        items.sort(key=lambda x: x["geo_mean"], reverse=True)
        rankings[key] = items

    return rankings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='output/layer_sensitivity_trap_all/sensitivity_results_trap.json')
    parser.add_argument('--min_ratio', type=float, default=1.0,
                        help='ratio 阈值（默认 1.0）')
    parser.add_argument('--export', type=str, default='configs/trap_combo_layers.json',
                        help='导出组合排名 JSON 路径')
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    trap_names = list(data.keys())
    print(f"Trap losses: {trap_names}")
    print(f"Ratio 阈值: > {args.min_ratio}")
    print()

    # 1) 每种 trap 的 ratio>threshold 层集合
    trap_layers = {}
    trap_layer_ratios = {}
    for trap in trap_names:
        layers = {}
        for entry in data[trap]['differential']:
            if entry['ratio'] > args.min_ratio:
                layers[entry['block']] = entry['ratio']
        trap_layers[trap] = set(layers.keys())
        trap_layer_ratios[trap] = layers

        print(f"{'='*70}")
        print(f"[{trap}] ratio > {args.min_ratio}: {len(layers)} 层")
        print(f"{'='*70}")
        sorted_layers = sorted(layers.items(), key=lambda x: x[1], reverse=True)
        for i, (block, ratio) in enumerate(sorted_layers, 1):
            safe = '✓ safe' if any(block.startswith(p) for p in
                                    ['unet.down_blocks.0.', 'unet.down_blocks.1.', 'unet.down_blocks.2.']) else ''
            print(f"  {i:3d}. {block:<50s}  ratio={ratio:.3f}  {safe}")
        print()

    # 2) 两两组合的重叠
    print(f"\n{'#'*70}")
    print(f"# 两两组合重叠分析")
    print(f"{'#'*70}\n")

    for t1, t2 in combinations(trap_names, 2):
        overlap = trap_layers[t1] & trap_layers[t2]
        print(f"{'='*70}")
        print(f"[{t1} ∩ {t2}] 重叠层: {len(overlap)} / "
              f"({len(trap_layers[t1])} ∩ {len(trap_layers[t2])})")
        print(f"{'='*70}")

        if not overlap:
            print("  (无重叠)\n")
            continue

        # 按几何平均 ratio 排序
        overlap_info = []
        for block in overlap:
            r1 = trap_layer_ratios[t1][block]
            r2 = trap_layer_ratios[t2][block]
            geo_mean = (r1 * r2) ** 0.5
            safe = any(block.startswith(p) for p in
                       ['unet.down_blocks.0.', 'unet.down_blocks.1.', 'unet.down_blocks.2.'])
            overlap_info.append((block, r1, r2, geo_mean, safe))

        overlap_info.sort(key=lambda x: x[3], reverse=True)

        print(f"  {'#':>3s}  {'Block':<50s}  {t1:>10s}  {t2:>10s}  {'GeoMean':>8s}  {'LoRA':>6s}")
        print(f"  {'---':>3s}  {'-'*50:<50s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*8:>8s}  {'-'*6:>6s}")
        for i, (block, r1, r2, gm, safe) in enumerate(overlap_info, 1):
            tag = '✓ safe' if safe else ''
            print(f"  {i:3d}  {block:<50s}  {r1:10.3f}  {r2:10.3f}  {gm:8.3f}  {tag:>6s}")
        print()

    # 3) 汇总：所有组合的重叠层数
    print(f"\n{'#'*70}")
    print(f"# 汇总")
    print(f"{'#'*70}\n")

    print(f"  {'组合':<30s}  {'重叠层数':>8s}  {'LoRA安全':>8s}")
    print(f"  {'-'*30:<30s}  {'-'*8:>8s}  {'-'*8:>8s}")
    for t1, t2 in combinations(trap_names, 2):
        overlap = trap_layers[t1] & trap_layers[t2]
        safe_count = sum(1 for b in overlap if any(b.startswith(p) for p in
                         ['unet.down_blocks.0.', 'unet.down_blocks.1.', 'unet.down_blocks.2.']))
        print(f"  {t1+' + '+t2:<30s}  {len(overlap):>8d}  {safe_count:>8d}")

    # 4) 全部 trap 的交集
    if len(trap_names) > 2:
        all_overlap = set.intersection(*trap_layers.values())
        print(f"\n  {'全部交集':<30s}  {len(all_overlap):>8d}")
        if all_overlap:
            print(f"\n  全部 {len(trap_names)} 种 trap 都 ratio>1 的层:")
            all_info = []
            for block in all_overlap:
                ratios = {t: trap_layer_ratios[t][block] for t in trap_names}
                geo_mean = 1.0
                for r in ratios.values():
                    geo_mean *= r
                geo_mean **= (1.0 / len(ratios))
                safe = any(block.startswith(p) for p in
                           ['unet.down_blocks.0.', 'unet.down_blocks.1.', 'unet.down_blocks.2.'])
                all_info.append((block, ratios, geo_mean, safe))
            all_info.sort(key=lambda x: x[2], reverse=True)

            header = f"  {'#':>3s}  {'Block':<45s}"
            for t in trap_names:
                header += f"  {t:>8s}"
            header += f"  {'GeoMean':>8s}  {'LoRA':>6s}"
            print(header)

            for i, (block, ratios, gm, safe) in enumerate(all_info, 1):
                line = f"  {i:3d}  {block:<45s}"
                for t in trap_names:
                    line += f"  {ratios[t]:8.3f}"
                line += f"  {gm:8.3f}  {'✓ safe' if safe else '':>6s}"
                print(line)

    # 5) 导出组合排名 JSON
    rankings = build_combo_rankings(data)
    export_path = Path(args.export)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, 'w') as f:
        json.dump(rankings, f, indent=2, ensure_ascii=False)
    print(f"\n[导出] 组合排名 → {export_path}")
    for key, items in rankings.items():
        print(f"  {key}: {len(items)} 层")


if __name__ == '__main__':
    main()
