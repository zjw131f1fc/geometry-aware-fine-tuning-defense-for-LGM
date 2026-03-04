#!/usr/bin/env python3
"""
显存占用诊断工具

用于分析 baseline attack 和 post defense attack 之间的显存差异
"""

import torch
import gc


def print_gpu_memory(tag=""):
    """打印当前 GPU 显存使用情况"""
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA 不可用")
        return

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
        print(f"[{tag}] GPU {i}:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Max Allocated: {max_allocated:.2f} GB")


def print_cuda_tensors(tag="", limit=20):
    """打印当前存在的 CUDA tensors"""
    import sys

    cuda_tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                cuda_tensors.append((type(obj).__name__, obj.size(), obj.element_size() * obj.nelement() / 1024**2))
        except Exception:
            pass

    if cuda_tensors:
        print(f"\n[{tag}] CUDA Tensors (前 {limit} 个):")
        cuda_tensors.sort(key=lambda x: x[2], reverse=True)
        for i, (name, size, mem_mb) in enumerate(cuda_tensors[:limit]):
            print(f"  {i+1}. {name} {size} - {mem_mb:.2f} MB")
        print(f"  总计: {len(cuda_tensors)} 个 CUDA tensors, {sum(t[2] for t in cuda_tensors):.2f} MB")
    else:
        print(f"[{tag}] 没有 CUDA tensors")


def aggressive_cleanup():
    """激进的显存清理"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
