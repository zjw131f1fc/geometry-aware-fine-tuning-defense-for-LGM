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

# Download sample metadata to get size info
# Metadata is split into multiple files (000-000.json.gz, 000-001.json.gz, etc.)
# We'll download the first few files as samples
print("\nDownloading sample metadata files from HF mirror...")

sample_files = ["000-000.json.gz", "000-001.json.gz", "000-002.json.gz"]
annotations = {}

for sample_file in sample_files:
    metadata_file = cache_dir / f"metadata_{sample_file}"

    if not metadata_file.exists():
        metadata_url = f"https://hf-mirror.com/datasets/allenai/objaverse/resolve/main/metadata/{sample_file}"
        print(f"Downloading {sample_file}...")
        urllib.request.urlretrieve(metadata_url, metadata_file)
        print(f"✓ Downloaded {sample_file}")
    else:
        print(f"✓ Using cached {sample_file}")

    # Load and merge
    with gzip.open(metadata_file, 'rb') as f:
        data = json.load(f)
        annotations.update(data)

print(f"✓ Loaded {len(annotations):,} sample annotations")

# Analyze sizes
print("\nAnalyzing object sizes...")

sizes = []
for uid, ann in annotations.items():
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
