#!/usr/bin/env python3
"""
导出 target 的 baseline / post-defense 攻击对比图。

工作流：
1. 读取 pipeline 导出的固定顺序素材：
   - <workspace>/phase1_baseline_attack/target_comparison_export
   - <workspace>/phase3_postdefense_attack/target_comparison_export
2. 生成逐样本编号预览图，方便人工挑选
3. 等待输入两个编号
4. 输出最终 2x8 排版图：
   - 第一排：baseline attack（前4张=物体A，后4张=物体B）
   - 第二排：post-defense attack（前4张=物体A，后4张=物体B）

注意：
- 该脚本依赖 run_pipeline.py 使用 --export_target_comparison_data 生成素材。
- 每个样本默认需要至少 4 张 rendered_view_*.png。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 target baseline / post-defense 攻击对比图")
    parser.add_argument(
        "workspace",
        type=str,
        help="pipeline 输出目录，如 output/experiments_output/xxx",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="baseline 素材目录，默认 <workspace>/phase1_baseline_attack/target_comparison_export",
    )
    parser.add_argument(
        "--postdef-dir",
        type=str,
        default=None,
        help="post-defense 素材目录，默认 <workspace>/phase3_postdefense_attack/target_comparison_export",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，默认 <workspace>/target_attack_comparison",
    )
    parser.add_argument(
        "--select",
        type=int,
        nargs=2,
        default=None,
        metavar=("ID_A", "ID_B"),
        help="直接指定两个编号，跳过交互输入",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=None,
        help="最多生成多少个预览图，默认全部",
    )
    parser.add_argument(
        "--cell-gap",
        type=int,
        default=18,
        help="最终大图中单元格间距像素",
    )
    parser.add_argument(
        "--group-gap",
        type=int,
        default=44,
        help="两个物体之间的额外间距像素",
    )
    parser.add_argument(
        "--row-gap",
        type=int,
        default=40,
        help="两排之间的间距像素",
    )
    parser.add_argument(
        "--header-height",
        type=int,
        default=64,
        help="最终大图顶部标题区域高度",
    )
    return parser.parse_args()


def load_manifest(export_dir: Path) -> Dict:
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    entries = manifest.get("entries") or []
    if not entries:
        raise RuntimeError(f"manifest 中没有 entries: {manifest_path}")
    return manifest


def sample_label(meta: Dict) -> str:
    if "category" in meta and "object" in meta:
        sample_idx = meta.get("sample_idx")
        if sample_idx is not None:
            return f"{meta['category']} / {meta['object']} / sample_idx={sample_idx}"
        return f"{meta['category']} / {meta['object']}"
    if "uuid" in meta:
        sample_idx = meta.get("sample_idx")
        if sample_idx is not None:
            return f"{meta['uuid']} / sample_idx={sample_idx}"
        return str(meta["uuid"])
    return f"global_index={meta.get('global_index', 'NA')}"


def build_matches(baseline_manifest: Dict, baseline_dir: Path, postdef_manifest: Dict, postdef_dir: Path) -> List[Dict]:
    post_by_key = {
        entry["sample_key"]: entry
        for entry in postdef_manifest.get("entries", [])
        if entry.get("sample_key")
    }

    matched: List[Dict] = []
    for baseline_entry in baseline_manifest.get("entries", []):
        sample_key = baseline_entry.get("sample_key")
        if not sample_key or sample_key not in post_by_key:
            continue
        post_entry = post_by_key[sample_key]
        matched.append({
            "sample_key": sample_key,
            "baseline": baseline_entry,
            "postdef": post_entry,
            "baseline_dir": baseline_dir,
            "postdef_dir": postdef_dir,
        })

    if not matched:
        raise RuntimeError("baseline 和 post-defense 素材没有可匹配的 sample_key")

    for idx, item in enumerate(matched, start=1):
        item["display_id"] = idx
    return matched


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def fit_same_width(img_a: Image.Image, img_b: Image.Image) -> Tuple[Image.Image, Image.Image]:
    target_width = max(img_a.width, img_b.width)
    out = []
    for img in (img_a, img_b):
        if img.width == target_width:
            out.append(img)
            continue
        new_height = max(1, round(img.height * target_width / img.width))
        out.append(img.resize((target_width, new_height), Image.BILINEAR))
    return out[0], out[1]


def draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    draw.text(xy, text, fill=(20, 20, 20), font=font)


def make_preview_image(item: Dict, font: ImageFont.ImageFont) -> Image.Image:
    baseline_strip = load_image(item["baseline_dir"] / item["baseline"]["rendered_strip_path"])
    postdef_strip = load_image(item["postdef_dir"] / item["postdef"]["rendered_strip_path"])
    baseline_strip, postdef_strip = fit_same_width(baseline_strip, postdef_strip)

    title = f"ID {item['display_id']:03d} | {sample_label(item['baseline'].get('metadata', {}))}"
    subtitle = "Top: Baseline Attack    Bottom: Post-Defense Attack"

    pad = 18
    title_h = 58
    canvas = Image.new(
        "RGB",
        (
            max(baseline_strip.width, postdef_strip.width) + pad * 2,
            title_h + baseline_strip.height + postdef_strip.height + pad * 3,
        ),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (pad, 10), title, font)
    draw_text(draw, (pad, 30), subtitle, font)

    y = title_h
    canvas.paste(baseline_strip, (pad, y))
    y += baseline_strip.height + pad
    canvas.paste(postdef_strip, (pad, y))
    return canvas


def write_index_files(output_dir: Path, matched: List[Dict]) -> Tuple[Path, Path]:
    txt_path = output_dir / "index.txt"
    csv_path = output_dir / "index.csv"

    with open(txt_path, "w", encoding="utf-8") as f:
        for item in matched:
            label = sample_label(item["baseline"].get("metadata", {}))
            f.write(f"{item['display_id']:03d}\t{item['sample_key']}\t{label}\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["display_id", "sample_key", "category", "object", "sample_idx", "uuid"])
        for item in matched:
            meta = item["baseline"].get("metadata", {})
            writer.writerow([
                item["display_id"],
                item["sample_key"],
                meta.get("category", ""),
                meta.get("object", ""),
                meta.get("sample_idx", ""),
                meta.get("uuid", ""),
            ])
    return txt_path, csv_path


def generate_previews(preview_dir: Path, matched: List[Dict], font: ImageFont.ImageFont, preview_limit: int | None) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    items = matched if preview_limit is None else matched[:preview_limit]
    for item in items:
        preview = make_preview_image(item, font)
        preview_path = preview_dir / f"{item['display_id']:03d}.png"
        preview.save(preview_path)


def resolve_selection(matched: List[Dict], select: List[int] | None) -> Tuple[Dict, Dict]:
    by_id = {item["display_id"]: item for item in matched}
    if select is None:
        raw = input("请输入两个编号（空格或逗号分隔，例如: 3 12）: ").strip()
        parts = [p for p in raw.replace(",", " ").split() if p]
        if len(parts) != 2:
            raise ValueError("需要输入两个编号")
        select = [int(parts[0]), int(parts[1])]

    if len(select) != 2:
        raise ValueError("必须提供两个编号")
    a_id, b_id = int(select[0]), int(select[1])
    if a_id not in by_id or b_id not in by_id:
        raise KeyError(f"编号不存在: {a_id}, {b_id}")
    return by_id[a_id], by_id[b_id]


def load_rendered_views(item: Dict, stage_key: str, expected: int = 4) -> List[Image.Image]:
    stage_entry = item[stage_key]
    root_dir = item["baseline_dir"] if stage_key == "baseline" else item["postdef_dir"]
    rel_paths = stage_entry.get("rendered_view_paths") or []
    if len(rel_paths) < expected:
        raise RuntimeError(
            f"{stage_key} 样本 {item['sample_key']} 只有 {len(rel_paths)} 张渲染图，少于需要的 {expected} 张"
        )
    return [load_image(root_dir / rel_paths[i]) for i in range(expected)]


def unify_cell_size(images: List[Image.Image]) -> Tuple[List[Image.Image], Tuple[int, int]]:
    target_w = max(img.width for img in images)
    target_h = max(img.height for img in images)
    out: List[Image.Image] = []
    for img in images:
        if img.size == (target_w, target_h):
            out.append(img)
            continue
        out.append(img.resize((target_w, target_h), Image.BILINEAR))
    return out, (target_w, target_h)


def build_final_canvas(item_a: Dict, item_b: Dict, font: ImageFont.ImageFont, *, cell_gap: int, group_gap: int, row_gap: int, header_height: int) -> Image.Image:
    baseline_images = load_rendered_views(item_a, "baseline") + load_rendered_views(item_b, "baseline")
    postdef_images = load_rendered_views(item_a, "postdef") + load_rendered_views(item_b, "postdef")
    all_images, cell_size = unify_cell_size(baseline_images + postdef_images)
    baseline_images = all_images[:8]
    postdef_images = all_images[8:]
    cell_w, cell_h = cell_size

    side_pad = 40
    right_pad = side_pad
    left_pad = side_pad
    top_pad = 24
    bottom_pad = 24
    inner_group_w = 4 * cell_w + 3 * cell_gap
    total_w = left_pad + inner_group_w * 2 + group_gap + right_pad
    total_h = top_pad + cell_h * 2 + row_gap + bottom_pad

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))

    def paste_row(images: List[Image.Image], y: int) -> None:
        x = left_pad
        for idx, img in enumerate(images):
            if idx == 4:
                x += group_gap
            canvas.paste(img, (x, y))
            x += cell_w + cell_gap

    paste_row(baseline_images, top_pad)
    paste_row(postdef_images, top_pad + cell_h + row_gap)

    draw = ImageDraw.Draw(canvas)
    sep_x = left_pad + inner_group_w + group_gap // 2
    draw.line([(sep_x, top_pad), (sep_x, total_h - bottom_pad)], fill=(180, 180, 180), width=2)
    return canvas


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"workspace 不存在: {workspace}")

    baseline_dir = Path(args.baseline_dir).expanduser().resolve() if args.baseline_dir else (workspace / "phase1_baseline_attack" / "target_comparison_export")
    postdef_dir = Path(args.postdef_dir).expanduser().resolve() if args.postdef_dir else (workspace / "phase3_postdefense_attack" / "target_comparison_export")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (workspace / "target_attack_comparison")
    preview_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_manifest = load_manifest(baseline_dir)
    postdef_manifest = load_manifest(postdef_dir)
    matched = build_matches(baseline_manifest, baseline_dir, postdef_manifest, postdef_dir)

    font = ImageFont.load_default()
    generate_previews(preview_dir, matched, font, args.preview_limit)
    index_txt, index_csv = write_index_files(output_dir, matched)

    print("=" * 72)
    print("Target attack comparison 预览已生成")
    print("=" * 72)
    print(f"workspace: {workspace}")
    print(f"baseline export: {baseline_dir}")
    print(f"post-defense export: {postdef_dir}")
    print(f"matched samples: {len(matched)}")
    print(f"preview dir: {preview_dir}")
    print(f"index txt: {index_txt}")
    print(f"index csv: {index_csv}")

    item_a, item_b = resolve_selection(matched, args.select)
    final_canvas = build_final_canvas(
        item_a,
        item_b,
        font,
        cell_gap=args.cell_gap,
        group_gap=args.group_gap,
        row_gap=args.row_gap,
        header_height=args.header_height,
    )

    out_stem = f"comparison_{item_a['display_id']:03d}_{item_b['display_id']:03d}"
    png_path = output_dir / f"{out_stem}.png"
    pdf_path = output_dir / f"{out_stem}.pdf"
    json_path = output_dir / f"{out_stem}.json"

    final_canvas.save(png_path)
    final_canvas.save(pdf_path, "PDF", resolution=300.0)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "selected_ids": [item_a["display_id"], item_b["display_id"]],
            "selected_sample_keys": [item_a["sample_key"], item_b["sample_key"]],
            "baseline_dir": str(baseline_dir),
            "postdef_dir": str(postdef_dir),
            "png_path": str(png_path),
            "pdf_path": str(pdf_path),
        }, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("最终对比图已导出")
    print("=" * 72)
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"Meta: {json_path}")


if __name__ == "__main__":
    main()
