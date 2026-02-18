#!/usr/bin/env python
"""
Render Objaverse objects in OmniObject3D format.

Usage:
    python scripts/render_objaverse.py --num_objects 400
    python scripts/render_objaverse.py --num_objects 10 --test
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.batch_renderer import ObjaverseBatchRenderer


def main():
    parser = argparse.ArgumentParser(description="Render Objaverse objects in OmniObject3D format")

    # Paths
    parser.add_argument(
        "--glb_dir",
        type=str,
        default="datas/objaverse/glbs",
        help="Directory containing GLB files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datas/objaverse_rendered",
        help="Output directory for rendered images"
    )
    parser.add_argument(
        "--blender_path",
        type=str,
        default="/mnt/huangjiaxin/blender-3.2.2-linux-x64/blender",
        help="Path to Blender executable"
    )

    # Rendering parameters
    parser.add_argument(
        "--num_objects",
        type=int,
        default=400,
        help="Number of objects to render"
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=100,
        help="Number of views per object"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=800,
        help="Image resolution (width and height)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel rendering workers"
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Randomly sample objects instead of taking first N"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use EEVEE engine for faster rendering (3-5x faster, slightly lower quality)"
    )

    # Test mode
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: render 1 object with 10 views"
    )

    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        args.num_objects = 1
        args.num_views = 10
        args.output_dir = "datas/objaverse_rendered_test"
        print("\n🧪 TEST MODE: Rendering 1 object with 10 views\n")

    # Convert relative paths to absolute
    glb_dir = project_root / args.glb_dir
    output_dir = project_root / args.output_dir
    render_script = project_root / "tools" / "render_omni_format.py"

    # Create renderer
    renderer = ObjaverseBatchRenderer(
        blender_path=args.blender_path,
        render_script_path=str(render_script),
        glb_dir=str(glb_dir),
        output_dir=str(output_dir),
        num_views=args.num_views,
        resolution=args.resolution,
        num_workers=args.num_workers,
        fast_mode=args.fast,
    )

    # Render
    results = renderer.render_batch(
        num_objects=args.num_objects,
        random_sample=args.random_sample
    )

    # Exit with error if any failures
    if results["failed"] > 0 or results["error"] > 0 or results["timeout"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
