#!/usr/bin/env python3
"""
Download Objaverse metadata from HF mirror and check dataset size.
"""

import os
import json
import gzip
import urllib.request
from pathlib import Path

# Create cache directory
cache_dir = Path.home() / ".objaverse" / "hf-objaverse-v1"
cache_dir.mkdir(parents=True, exist_ok=True)

# Download object-paths.json.gz from mirror
metadata_file = cache_dir / "object-paths.json.gz"

if not metadata_file.exists():
    print("Downloading metadata from HF mirror...")
    mirror_url = "https://hf-mirror.com/datasets/allenai/objaverse/resolve/main/object-paths.json.gz"

    print(f"URL: {mirror_url}")
    print(f"Saving to: {metadata_file}")

    urllib.request.urlretrieve(mirror_url, metadata_file)
    print("✓ Download complete")
else:
    print(f"✓ Using cached metadata: {metadata_file}")

# Load and analyze
print("\nLoading metadata...")
with gzip.open(metadata_file, 'rb') as f:
    object_paths = json.load(f)

total_objects = len(object_paths)
print(f"✓ Total objects: {total_objects:,}")

# Download annotations to get size info
annotations_file = cache_dir / "annotations.json.gz"

if not annotations_file.exists():
    print("\nDownloading annotations from HF mirror...")
    annotations_url = "https://hf-mirror.com/datasets/allenai/objaverse/resolve/main/annotations.json.gz"

    print(f"URL: {annotations_url}")
    print(f"Saving to: {annotations_file}")
    print("(This may take 2-5 minutes, file size ~50MB)")

    urllib.request.urlretrieve(annotations_url, annotations_file)
    print("✓ Download complete")
else:
    print(f"\n✓ Using cached annotations: {annotations_file}")

# Analyze sizes
print("\nAnalyzing object sizes...")
with gzip.open(annotations_file, 'rb') as f:
    annotations = json.load(f)

sizes = []
for uid, ann in list(annotations.items())[:10000]:  # Sample first 10k
    if 'archives' in ann and 'glb' in ann['archives']:
        size = ann['archives']['glb'].get('size', 0)
        if size > 0:
            sizes.append(size)

if sizes:
    avg_size = sum(sizes) / len(sizes)
    min_size = min(sizes)
    max_size = max(sizes)

    print(f"\n{'='*60}")
    print(f"Size Statistics (from {len(sizes):,} objects)")
    print(f"{'='*60}")
    print(f"Average size: {avg_size / 1024 / 1024:.2f} MB")
    print(f"Min size: {min_size / 1024 / 1024:.2f} MB")
    print(f"Max size: {max_size / 1024 / 1024:.2f} MB")

    # Estimate total
    total_gb = (avg_size * total_objects) / 1024 / 1024 / 1024
    total_tb = total_gb / 1024

    print(f"\n{'='*60}")
    print(f"Estimated Total Dataset Size")
    print(f"{'='*60}")
    print(f"Total objects: {total_objects:,}")
    print(f"Estimated total: {total_gb:.1f} GB ({total_tb:.2f} TB)")

    # Download size estimates
    print(f"\n{'='*60}")
    print(f"Download Size Estimates")
    print(f"{'='*60}")
    for n in [100, 1000, 10000, 50000, 100000]:
        if n <= total_objects:
            est_size = (avg_size * n) / 1024 / 1024 / 1024
            print(f"{n:7,} objects: ~{est_size:6.1f} GB")

    print(f"\n{'='*60}")
    print(f"Available disk space: 3.3 TB")
    print(f"{'='*60}")
