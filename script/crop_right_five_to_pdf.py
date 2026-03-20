#!/usr/bin/env python3
"""
裁剪拼图中最右侧 5 个视角块，并分别导出为 PDF。

典型输入是两行多列、白底的对比拼图。脚本会：
1. 识别非白色前景区域
2. 按 x 方向自动切成多个列块
3. 取最右侧 5 个列块
4. 计算包含这 5 个列块的最小裁剪框
5. 为每张输入图各输出一个 PDF

用法:
  python script/crop_right_five_to_pdf.py img1.png img2.png

  python script/crop_right_five_to_pdf.py \
      img1.png img2.png \
      --output_dir cropped_pdfs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


Box = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="裁剪拼图最右侧 5 个视角块并导出为 PDF"
    )
    parser.add_argument(
        "images",
        nargs=2,
        help="输入的两张拼图路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录，默认与第一张图同目录",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=245,
        help="白底阈值，小于该值视为前景，默认 245",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=16,
        help="裁剪框四周补边像素，默认 16",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=40,
        help="两个列块之间小于该间距时自动合并，默认 40",
    )
    parser.add_argument(
        "--min-segment-width",
        type=int,
        default=20,
        help="忽略过窄的列块噪声，默认 20",
    )
    parser.add_argument(
        "--num-panels",
        type=int,
        default=5,
        help="取最右边多少个列块，默认 5",
    )
    parser.add_argument(
        "--expand-left",
        type=int,
        default=0,
        help="导出前在左侧额外补白边像素，默认 0",
    )
    parser.add_argument(
        "--expand-right",
        type=int,
        default=0,
        help="导出前在右侧额外补白边像素，默认 0",
    )
    parser.add_argument(
        "--expand-top",
        type=int,
        default=0,
        help="导出前在上侧额外补白边像素，默认 0",
    )
    parser.add_argument(
        "--expand-bottom",
        type=int,
        default=0,
        help="导出前在下侧额外补白边像素，默认 0",
    )
    parser.add_argument(
        "--balance-horizontal",
        action="store_true",
        help="基于最终前景框自动把左右白边补成一致",
    )
    parser.add_argument(
        "--balance-vertical",
        action="store_true",
        help="基于最终前景框自动把上下白边补成一致",
    )
    return parser.parse_args()


def iter_runs(mask_1d: np.ndarray) -> Iterable[Tuple[int, int]]:
    start = None
    for idx, value in enumerate(mask_1d.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            yield start, idx
            start = None
    if start is not None:
        yield start, len(mask_1d)


def merge_runs(runs: Sequence[Tuple[int, int]], min_gap: int) -> List[Tuple[int, int]]:
    if not runs:
        return []

    merged: List[Tuple[int, int]] = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < min_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def find_rightmost_crop_box(
    image: Image.Image,
    white_threshold: int,
    padding: int,
    min_gap: int,
    min_segment_width: int,
    num_panels: int,
) -> Box:
    rgb = np.asarray(image.convert("RGB"))
    foreground = np.any(rgb < white_threshold, axis=2)

    if not foreground.any():
        raise RuntimeError("图像中没有检测到前景，可能阈值过低或输入图为空白。")

    col_mask = foreground.any(axis=0)
    runs = [(start, end) for start, end in iter_runs(col_mask) if (end - start) >= min_segment_width]
    runs = merge_runs(runs, min_gap=min_gap)

    if len(runs) < num_panels:
        raise RuntimeError(
            f"只检测到 {len(runs)} 个列块，少于请求的 {num_panels} 个。"
        )

    selected = runs[-num_panels:]
    x1 = selected[0][0]
    x2 = selected[-1][1]

    local_foreground = foreground[:, x1:x2]
    row_mask = local_foreground.any(axis=1)
    row_runs = list(iter_runs(row_mask))
    if not row_runs:
        raise RuntimeError("已选中的右侧列块内没有检测到前景。")

    y1 = row_runs[0][0]
    y2 = row_runs[-1][1]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)
    return x1, y1, x2, y2


def default_output_dir(image_paths: Sequence[Path]) -> Path:
    return image_paths[0].resolve().parent


def output_pdf_path(image_path: Path, output_dir: Path) -> Path:
    stem = image_path.stem
    return output_dir / f"{stem}_right5.pdf"


def expand_with_white_border(
    image: Image.Image,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> Image.Image:
    left = max(int(left), 0)
    right = max(int(right), 0)
    top = max(int(top), 0)
    bottom = max(int(bottom), 0)

    if left == 0 and right == 0 and top == 0 and bottom == 0:
        return image

    canvas = Image.new(
        "RGB",
        (image.width + left + right, image.height + top + bottom),
        color=(255, 255, 255),
    )
    canvas.paste(image, (left, top))
    return canvas


def _foreground_bbox(image: Image.Image, white_threshold: int) -> Box | None:
    arr = np.asarray(image.convert("RGB"))
    fg = np.any(arr < white_threshold, axis=2)
    if not fg.any():
        return None
    rows = np.where(fg.any(axis=1))[0]
    cols = np.where(fg.any(axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def balance_margins(
    image: Image.Image,
    white_threshold: int,
    horizontal: bool,
    vertical: bool,
) -> Image.Image:
    bbox = _foreground_bbox(image, white_threshold=white_threshold)
    if bbox is None:
        return image

    x1, y1, x2, y2 = bbox
    left = x1
    right = image.width - x2
    top = y1
    bottom = image.height - y2

    add_left = add_right = add_top = add_bottom = 0
    if horizontal and left != right:
        target = max(left, right)
        add_left = target - left
        add_right = target - right
    if vertical and top != bottom:
        target = max(top, bottom)
        add_top = target - top
        add_bottom = target - bottom

    return expand_with_white_border(
        image,
        left=add_left,
        right=add_right,
        top=add_top,
        bottom=add_bottom,
    )


def main() -> None:
    args = parse_args()

    image_paths = [Path(p).expanduser().resolve() for p in args.images]
    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"输入图像不存在: {image_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(image_paths)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        with Image.open(image_path) as img:
            box = find_rightmost_crop_box(
                img,
                white_threshold=args.white_threshold,
                padding=args.padding,
                min_gap=args.min_gap,
                min_segment_width=args.min_segment_width,
                num_panels=args.num_panels,
            )
            cropped = img.crop(box).convert("RGB")
            cropped = expand_with_white_border(
                cropped,
                left=args.expand_left,
                right=args.expand_right,
                top=args.expand_top,
                bottom=args.expand_bottom,
            )
            cropped = balance_margins(
                cropped,
                white_threshold=args.white_threshold,
                horizontal=bool(args.balance_horizontal),
                vertical=bool(args.balance_vertical),
            )
            pdf_path = output_pdf_path(image_path, output_dir)
            cropped.save(pdf_path, "PDF", resolution=300.0)
            print(f"[OK] {image_path.name} -> {pdf_path}")
            print(f"     crop_box={box}")


if __name__ == "__main__":
    main()
