#!/usr/bin/env python3
"""
Check rendering quality and identify problematic renders.
"""

import argparse
from pathlib import Path
import json
from PIL import Image
import numpy as np


def check_image_quality(image_path):
    """Check if an image is empty or has issues."""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)

        # Check if image is completely transparent
        if img.mode == 'RGBA':
            alpha = img_array[:, :, 3]
            if np.all(alpha == 0):
                return {"status": "empty", "reason": "completely transparent"}

            # Check if very few pixels are visible
            visible_pixels = np.sum(alpha > 0)
            total_pixels = alpha.size
            visibility_ratio = visible_pixels / total_pixels

            if visibility_ratio < 0.01:
                return {"status": "mostly_empty", "reason": f"only {visibility_ratio*100:.2f}% visible"}

        # Check if image is all white/black
        if img.mode in ['RGB', 'RGBA']:
            rgb = img_array[:, :, :3]
            mean_color = np.mean(rgb, axis=(0, 1))
            std_color = np.std(rgb, axis=(0, 1))

            # All white
            if np.all(mean_color > 250) and np.all(std_color < 5):
                return {"status": "all_white", "reason": "image is all white"}

            # All black
            if np.all(mean_color < 5) and np.all(std_color < 5):
                return {"status": "all_black", "reason": "image is all black"}

        return {"status": "ok", "visibility": visibility_ratio if img.mode == 'RGBA' else 1.0}

    except Exception as e:
        return {"status": "error", "reason": str(e)}


def check_object_renders(object_dir):
    """Check all renders for a single object."""
    render_dir = object_dir / "render"
    images_dir = render_dir / "images"

    if not images_dir.exists():
        return {"status": "no_images", "object": object_dir.name}

    # Check debug info
    debug_file = render_dir / "debug_info.txt"
    debug_info = {}
    if debug_file.exists():
        with open(debug_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    debug_info[key.strip()] = value.strip()

    # Check all images
    image_files = sorted(images_dir.glob("r_*.png"))
    issues = []

    for img_path in image_files:
        quality = check_image_quality(img_path)
        if quality["status"] != "ok":
            issues.append({
                "image": img_path.name,
                "status": quality["status"],
                "reason": quality.get("reason", "unknown")
            })

    return {
        "status": "checked",
        "object": object_dir.name,
        "total_images": len(image_files),
        "issues": issues,
        "debug_info": debug_info
    }


def main():
    parser = argparse.ArgumentParser(description="Check rendering quality")
    parser.add_argument("--render_dir", type=str, required=True,
                        help="Directory containing rendered objects")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for detailed report")

    args = parser.parse_args()

    render_dir = Path(args.render_dir)
    if not render_dir.exists():
        print(f"Error: Directory not found: {render_dir}")
        return

    # Find all object directories
    object_dirs = [d for d in render_dir.iterdir() if d.is_dir()]
    print(f"Found {len(object_dirs)} objects to check\n")

    # Check each object
    results = []
    objects_with_issues = []

    for obj_dir in object_dirs:
        result = check_object_renders(obj_dir)
        results.append(result)

        if result.get("issues"):
            objects_with_issues.append(result)
            print(f"⚠ {result['object']}: {len(result['issues'])}/{result['total_images']} images have issues")
            for issue in result['issues'][:3]:  # Show first 3 issues
                print(f"    {issue['image']}: {issue['reason']}")
            if len(result['issues']) > 3:
                print(f"    ... and {len(result['issues']) - 3} more")
        else:
            print(f"✓ {result['object']}: all {result['total_images']} images OK")

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total objects: {len(results)}")
    print(f"Objects with issues: {len(objects_with_issues)}")
    print(f"{'='*60}\n")

    # Save detailed report
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write("Rendering Quality Report\n")
            f.write("="*60 + "\n\n")

            for result in objects_with_issues:
                f.write(f"\nObject: {result['object']}\n")
                f.write(f"Total images: {result['total_images']}\n")
                f.write(f"Issues: {len(result['issues'])}\n")

                if result.get('debug_info'):
                    f.write("\nDebug Info:\n")
                    for key, value in result['debug_info'].items():
                        f.write(f"  {key}: {value}\n")

                f.write("\nProblematic images:\n")
                for issue in result['issues']:
                    f.write(f"  {issue['image']}: {issue['reason']}\n")

                f.write("\n" + "-"*60 + "\n")

        print(f"Detailed report saved to: {output_path}")


if __name__ == "__main__":
    main()
