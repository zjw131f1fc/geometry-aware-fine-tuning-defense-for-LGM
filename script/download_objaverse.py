#!/usr/bin/env python3
"""
Download Objaverse dataset to local directory.

Usage:
    python scripts/download_objaverse.py --num_objects 100 --output_dir ./datas/objaverse
    python scripts/download_objaverse.py --category chair --max_per_category 50
    python scripts/download_objaverse.py --sample_random 1000
"""

import os
# Configure HF mirror before importing objaverse
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import objaverse
import shutil
from pathlib import Path
import random
from tqdm import tqdm


def download_objaverse(
    output_dir: str = "./datas/objaverse",
    num_objects: int = None,
    category: str = None,
    max_per_category: int = None,
    sample_random: int = None,
    download_processes: int = 8,
    license_filter: str = None,
):
    """Download Objaverse objects to specified directory."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_path.absolute()}")
    print(f"Download processes: {download_processes}")

    # Load all UIDs
    print("\nLoading object UIDs...")
    all_uids = objaverse.load_uids()
    print(f"Total objects available: {len(all_uids)}")

    # Select UIDs based on criteria
    selected_uids = []

    if category or max_per_category:
        print("\nLoading LVIS annotations...")
        lvis_annotations = objaverse.load_lvis_annotations()

        if category:
            if category in lvis_annotations:
                category_uids = lvis_annotations[category]
                if max_per_category:
                    category_uids = category_uids[:max_per_category]
                selected_uids = category_uids
                print(f"Selected {len(selected_uids)} objects from category '{category}'")
            else:
                print(f"Category '{category}' not found!")
                print(f"Available categories: {list(lvis_annotations.keys())[:20]}...")
                return
        else:
            # Sample from all categories
            for cat, uids in lvis_annotations.items():
                selected_uids.extend(uids[:max_per_category])
            print(f"Selected {len(selected_uids)} objects from all categories")

    elif sample_random:
        selected_uids = random.sample(all_uids, min(sample_random, len(all_uids)))
        print(f"Randomly sampled {len(selected_uids)} objects")

    elif num_objects:
        selected_uids = all_uids[:num_objects]
        print(f"Selected first {len(selected_uids)} objects")

    else:
        print("WARNING: No selection criteria specified. Downloading first 100 objects.")
        selected_uids = all_uids[:100]

    # Apply license filter if specified
    if license_filter:
        print(f"\nFiltering by license: {license_filter}")
        print("Loading annotations (this may take a few minutes)...")
        annotations = objaverse.load_annotations(selected_uids)
        selected_uids = [
            uid for uid in selected_uids
            if annotations.get(uid, {}).get('license') == license_filter
        ]
        print(f"After license filter: {len(selected_uids)} objects")

    if not selected_uids:
        print("No objects selected. Exiting.")
        return

    # Download objects
    print(f"\nDownloading {len(selected_uids)} objects...")
    print("Note: Objects will first download to ~/.objaverse/ cache")

    objects = objaverse.load_objects(
        uids=selected_uids,
        download_processes=download_processes
    )

    # Copy to output directory
    print(f"\nCopying objects to {output_path}...")
    glb_dir = output_path / "glbs"
    glb_dir.mkdir(exist_ok=True)

    for uid, source_path in tqdm(objects.items(), desc="Copying files"):
        dest_path = glb_dir / f"{uid}.glb"
        if not dest_path.exists():
            shutil.copy2(source_path, dest_path)

    # Save UID list
    uid_list_path = output_path / "downloaded_uids.txt"
    with open(uid_list_path, 'w') as f:
        for uid in selected_uids:
            f.write(f"{uid}\n")

    print(f"\n✓ Download complete!")
    print(f"  - Objects: {len(objects)}")
    print(f"  - Location: {glb_dir.absolute()}")
    print(f"  - UID list: {uid_list_path.absolute()}")

    # Print statistics
    total_size = sum(os.path.getsize(glb_dir / f"{uid}.glb") for uid in objects.keys())
    print(f"  - Total size: {total_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download Objaverse dataset")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datas/objaverse",
        help="Output directory for downloaded objects"
    )

    parser.add_argument(
        "--num_objects",
        type=int,
        help="Number of objects to download (from start of list)"
    )

    parser.add_argument(
        "--sample_random",
        type=int,
        help="Randomly sample N objects"
    )

    parser.add_argument(
        "--category",
        type=str,
        help="Download objects from specific LVIS category (e.g., 'chair', 'car')"
    )

    parser.add_argument(
        "--max_per_category",
        type=int,
        help="Max objects per category (use with --category or alone for all categories)"
    )

    parser.add_argument(
        "--license_filter",
        type=str,
        choices=['by', 'by-sa', 'by-nd', 'by-nc', 'by-nc-sa', 'by-nc-nd', 'cc0'],
        help="Filter by license type (e.g., 'by' for CC-BY)"
    )

    parser.add_argument(
        "--download_processes",
        type=int,
        default=8,
        help="Number of parallel download processes"
    )

    args = parser.parse_args()

    download_objaverse(
        output_dir=args.output_dir,
        num_objects=args.num_objects,
        category=args.category,
        max_per_category=args.max_per_category,
        sample_random=args.sample_random,
        download_processes=args.download_processes,
        license_filter=args.license_filter,
    )


if __name__ == "__main__":
    main()
