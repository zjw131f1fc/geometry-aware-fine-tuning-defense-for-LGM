#!/usr/bin/env python3
"""
Download Objaverse objects directly from HF mirror.

Usage:
    python scripts/download_objaverse_direct.py --num_objects 1000 --output_dir ./datas/objaverse
    python scripts/download_objaverse_direct.py --sample_random 13000 --max_size_gb 200
"""

import argparse
import json
import gzip
import urllib.request
import random
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_object_paths(cache_dir):
    """Load object paths from cache."""
    metadata_file = cache_dir / "object-paths.json.gz"

    if not metadata_file.exists():
        print("Downloading object paths...")
        url = "https://hf-mirror.com/datasets/allenai/objaverse/resolve/main/object-paths.json.gz"
        urllib.request.urlretrieve(url, metadata_file)

    with gzip.open(metadata_file, 'rb') as f:
        return json.load(f)


def download_object(uid, glb_path, output_dir, base_url):
    """Download a single object."""
    output_file = output_dir / f"{uid}.glb"

    if output_file.exists():
        return uid, output_file, os.path.getsize(output_file), True

    try:
        # Construct download URL
        url = f"{base_url}/{glb_path}"

        # Download
        urllib.request.urlretrieve(url, output_file)
        size = os.path.getsize(output_file)

        return uid, output_file, size, False
    except Exception as e:
        return uid, None, 0, False


def download_objaverse(
    output_dir="./datas/objaverse",
    num_objects=None,
    sample_random=None,
    max_size_gb=None,
    num_workers=8,
):
    """Download Objaverse objects from HF mirror."""

    output_path = Path(output_dir)
    glb_dir = output_path / "glbs"
    glb_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path.home() / ".objaverse" / "hf-objaverse-v1"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {glb_dir.absolute()}")
    print(f"Download workers: {num_workers}")

    # Load object paths
    print("\nLoading object paths...")
    object_paths = load_object_paths(cache_dir)
    all_uids = list(object_paths.keys())
    print(f"Total available objects: {len(all_uids):,}")

    # Select objects
    if sample_random:
        selected_uids = random.sample(all_uids, min(sample_random, len(all_uids)))
        print(f"Randomly sampled {len(selected_uids):,} objects")
    elif num_objects:
        selected_uids = all_uids[:num_objects]
        print(f"Selected first {len(selected_uids):,} objects")
    else:
        print("No selection criteria. Downloading first 100 objects.")
        selected_uids = all_uids[:100]

    # Estimate size
    avg_size_mb = 15.65  # From analysis
    estimated_size_gb = (len(selected_uids) * avg_size_mb) / 1024
    print(f"\nEstimated download size: {estimated_size_gb:.1f} GB")

    if max_size_gb and estimated_size_gb > max_size_gb:
        max_objects = int((max_size_gb * 1024) / avg_size_mb)
        print(f"WARNING: Estimated size exceeds limit of {max_size_gb} GB")
        print(f"Reducing to {max_objects:,} objects")
        selected_uids = selected_uids[:max_objects]

    # Base URL for downloads
    base_url = "https://hf-mirror.com/datasets/allenai/objaverse/resolve/main"

    # Download objects
    print(f"\nDownloading {len(selected_uids):,} objects...")
    print("This may take a while depending on your network speed.")

    downloaded = 0
    skipped = 0
    failed = 0
    total_size = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                download_object,
                uid,
                object_paths[uid],
                glb_dir,
                base_url
            ): uid
            for uid in selected_uids
        }

        with tqdm(total=len(selected_uids), desc="Downloading") as pbar:
            for future in as_completed(futures):
                uid, output_file, size, was_cached = future.result()

                if output_file:
                    if was_cached:
                        skipped += 1
                    else:
                        downloaded += 1
                    total_size += size
                else:
                    failed += 1

                pbar.update(1)
                pbar.set_postfix({
                    'downloaded': downloaded,
                    'cached': skipped,
                    'failed': failed,
                    'size_gb': f'{total_size / 1024 / 1024 / 1024:.1f}'
                })

    # Save UID list
    uid_list_path = output_path / "downloaded_uids.txt"
    with open(uid_list_path, 'w') as f:
        for uid in selected_uids:
            f.write(f"{uid}\n")

    print(f"\n{'='*60}")
    print(f"Download Complete!")
    print(f"{'='*60}")
    print(f"Downloaded: {downloaded:,} objects")
    print(f"Cached: {skipped:,} objects")
    print(f"Failed: {failed:,} objects")
    print(f"Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"Location: {glb_dir.absolute()}")
    print(f"UID list: {uid_list_path.absolute()}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Objaverse dataset from HF mirror"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datas/objaverse",
        help="Output directory"
    )

    parser.add_argument(
        "--num_objects",
        type=int,
        help="Number of objects (from start of list)"
    )

    parser.add_argument(
        "--sample_random",
        type=int,
        help="Randomly sample N objects"
    )

    parser.add_argument(
        "--max_size_gb",
        type=float,
        help="Maximum download size in GB (will limit number of objects)"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel download workers"
    )

    args = parser.parse_args()

    download_objaverse(
        output_dir=args.output_dir,
        num_objects=args.num_objects,
        sample_random=args.sample_random,
        max_size_gb=args.max_size_gb,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
