#!/usr/bin/env python3
"""
导出 target 的 baseline / post-defense 对比图。

默认读取：
  <workspace>/phase1_baseline_attack/target_renders
  <workspace>/phase3_postdefense_attack/target_renders

并按同名 sample 文件配对，输出到：
  <workspace>/target_before_after
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

from PIL import Image


ALLOWED_SUFFIXES = (".png", ".jpg", ".jpeg")
PREFERRED_PATTERNS = (
    "target_*_paired.png",
    "target_*_rendered.png",
    "target_*.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导出 target baseline / post-defense before-after 对比图"
    )
    parser.add_argument(
        "workspace",
        type=str,
        help="pipeline 输出目录，如 output/experiments_output/single_xxx",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录，默认 <workspace>/target_before_after",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="horizontal",
        choices=("horizontal", "vertical"),
        help="拼接布局：horizontal 或 vertical",
    )
    parser.add_argument(
        "--spacer",
        type=int,
        default=24,
        help="两张图之间的留白像素数",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多导出多少张配对图，默认导出全部",
    )
    parser.add_argument(
        "--prefer",
        type=str,
        default="paired",
        choices=("paired", "rendered", "auto"),
        help="优先使用 paired 图、rendered 图，或自动回退",
    )
    return parser.parse_args()


def resolve_render_dir(workspace: Path, phase_dirname: str) -> Path:
    render_dir = workspace / phase_dirname / "target_renders"
    if not render_dir.is_dir():
        raise FileNotFoundError(f"目录不存在: {render_dir}")
    return render_dir


def _iter_patterns(prefer: str) -> Iterable[str]:
    if prefer == "paired":
        yield "target_*_paired.png"
        yield "target_*_rendered.png"
        yield "target_*.png"
        return
    if prefer == "rendered":
        yield "target_*_rendered.png"
        yield "target_*_paired.png"
        yield "target_*.png"
        return
    yield from PREFERRED_PATTERNS


def build_sample_map(render_dir: Path, prefer: str) -> Dict[str, Path]:
    sample_map: Dict[str, Path] = {}

    for pattern in _iter_patterns(prefer):
        for path in sorted(render_dir.glob(pattern)):
            if not path.is_file():
                continue
            if path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            sample_key = path.stem.replace("_paired", "").replace("_rendered", "")
            sample_map.setdefault(sample_key, path)

    return sample_map


def resize_to_match(img_a: Image.Image, img_b: Image.Image, layout: str) -> Tuple[Image.Image, Image.Image]:
    if layout == "horizontal":
        if img_a.height == img_b.height:
            return img_a, img_b
        target_h = max(img_a.height, img_b.height)
        if img_a.height != target_h:
            new_w = round(img_a.width * target_h / img_a.height)
            img_a = img_a.resize((new_w, target_h), Image.BILINEAR)
        if img_b.height != target_h:
            new_w = round(img_b.width * target_h / img_b.height)
            img_b = img_b.resize((new_w, target_h), Image.BILINEAR)
        return img_a, img_b

    if img_a.width == img_b.width:
        return img_a, img_b
    target_w = max(img_a.width, img_b.width)
    if img_a.width != target_w:
        new_h = round(img_a.height * target_w / img_a.width)
        img_a = img_a.resize((target_w, new_h), Image.BILINEAR)
    if img_b.width != target_w:
        new_h = round(img_b.height * target_w / img_b.width)
        img_b = img_b.resize((target_w, new_h), Image.BILINEAR)
    return img_a, img_b


def make_canvas(before: Image.Image, after: Image.Image, layout: str, spacer: int) -> Image.Image:
    before, after = resize_to_match(before, after, layout)
    spacer = max(int(spacer), 0)

    if layout == "horizontal":
        canvas = Image.new(
            "RGB",
            (before.width + spacer + after.width, max(before.height, after.height)),
            color=(255, 255, 255),
        )
        canvas.paste(before, (0, 0))
        canvas.paste(after, (before.width + spacer, 0))
        return canvas

    canvas = Image.new(
        "RGB",
        (max(before.width, after.width), before.height + spacer + after.height),
        color=(255, 255, 255),
    )
    canvas.paste(before, (0, 0))
    canvas.paste(after, (0, before.height + spacer))
    return canvas


def main() -> None:
    args = parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"workspace 不存在: {workspace}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (workspace / "target_before_after")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = resolve_render_dir(workspace, "phase1_baseline_attack")
    postdef_dir = resolve_render_dir(workspace, "phase3_postdefense_attack")

    baseline_map = build_sample_map(baseline_dir, prefer=args.prefer)
    postdef_map = build_sample_map(postdef_dir, prefer=args.prefer)
    common_keys = sorted(set(baseline_map) & set(postdef_map))

    if not common_keys:
        raise RuntimeError(
            "没有找到可配对的 target render 文件。"
            f"\nbaseline_dir={baseline_dir}"
            f"\npostdef_dir={postdef_dir}"
        )

    if args.limit is not None and args.limit > 0:
        common_keys = common_keys[: args.limit]

    exported = 0
    for sample_key in common_keys:
        before_path = baseline_map[sample_key]
        after_path = postdef_map[sample_key]

        with Image.open(before_path) as before_img, Image.open(after_path) as after_img:
            before_rgb = before_img.convert("RGB")
            after_rgb = after_img.convert("RGB")
            canvas = make_canvas(before_rgb, after_rgb, layout=args.layout, spacer=args.spacer)

        out_name = f"{sample_key}_baseline_vs_postdef.png"
        out_path = output_dir / out_name
        canvas.save(out_path)
        exported += 1
        print(f"[Export] {sample_key} -> {out_path}")

    baseline_only = sorted(set(baseline_map) - set(postdef_map))
    postdef_only = sorted(set(postdef_map) - set(baseline_map))

    print("\n" + "=" * 60)
    print("导出完成")
    print("=" * 60)
    print(f"workspace: {workspace}")
    print(f"baseline target_renders: {baseline_dir}")
    print(f"postdef target_renders: {postdef_dir}")
    print(f"output: {output_dir}")
    print(f"matched: {len(common_keys)}")
    print(f"exported: {exported}")
    if baseline_only:
        print(f"baseline-only samples ({len(baseline_only)}): {baseline_only[:8]}")
    if postdef_only:
        print(f"postdef-only samples ({len(postdef_only)}): {postdef_only[:8]}")


if __name__ == "__main__":
    main()
