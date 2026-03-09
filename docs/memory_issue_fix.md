# 显存占用问题诊断与修复

## 问题描述

用户报告 **post defense attack 比 baseline attack 显存占用更高**，怀疑 defense 阶段有模型没有释放。

## 根本原因分析

经过代码审查，发现以下潜在问题：

### 1. Defense Trainer 清理不彻底

在 `training/defense_trainer.py` 的 `load_or_train_defense` 函数中（第 2685 行），只是简单地执行了：

```python
del trainer
torch.cuda.empty_cache()
```

但是 **没有显式清理 trainer 内部的显存占用对象**：
- `trainer.optimizer`：优化器状态（Adam 的动量、方差等）占用大量显存
- `trainer.model_mgr.model`：模型本身
- `trainer.data_mgr`：数据管理器可能持有的缓存

虽然 Python 的垃圾回收机制理论上会清理这些对象，但在复杂的对象引用关系中，可能存在循环引用或其他问题导致显存没有被及时释放。

### 2. Phase 之间缺少显式清理

在 `script/run_pipeline.py` 中，Phase 1 → Phase 2 → Phase 3 之间没有显式的垃圾回收和显存清理步骤。虽然每个 phase 结束时会调用 `torch.cuda.empty_cache()`，但没有调用 `gc.collect()` 来触发 Python 的垃圾回收。

### 3. 可能的原因

**为什么 post defense attack 显存占用更高？**

1. **Defense state_dict 保存在内存中**：当 `cache_mode != "registry"` 时，defense 训练完成后会将 `state_dict` 保存在 CPU 内存中（第 2683 行），虽然这不占用 GPU 显存，但可能影响整体内存管理。

2. **优化器状态残留**：Defense training 的优化器状态可能没有被完全释放，导致 GPU 显存碎片化。

3. **CUDA 缓存碎片化**：多次训练后，CUDA 的内存分配器可能产生碎片，导致后续分配需要更多显存。

## 修复方案

### 修复 1：Defense Trainer 显式清理

在 `training/defense_trainer.py` 的 `load_or_train_defense` 函数中，添加显式清理逻辑：

```python
# 显式清理 trainer 内部的显存占用对象
if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
    del trainer.optimizer
if hasattr(trainer, 'model_mgr') and trainer.model_mgr is not None:
    if hasattr(trainer.model_mgr, 'model') and trainer.model_mgr.model is not None:
        del trainer.model_mgr.model
    del trainer.model_mgr
if hasattr(trainer, 'data_mgr') and trainer.data_mgr is not None:
    del trainer.data_mgr

del trainer
torch.cuda.empty_cache()
```

### 修复 2：Phase 之间添加显式清理

在 `script/run_pipeline.py` 中，在每个 Phase 结束后添加：

```python
import gc
gc.collect()
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.synchronize()
```

## 验证方法

### 方法 1：使用 nvidia-smi 监控

在运行 pipeline 时，使用 `watch -n 1 nvidia-smi` 实时监控显存占用：

```bash
watch -n 1 nvidia-smi
```

观察：
- Phase 1 (Baseline Attack) 结束后的显存占用
- Phase 2 (Defense Training) 结束后的显存占用
- Phase 3 (Post-Defense Attack) 开始时的显存占用

### 方法 2：使用诊断工具

在 pipeline 的关键位置添加显存诊断代码：

```python
from tools.debug_memory import print_gpu_memory, print_cuda_tensors, aggressive_cleanup

# Phase 1 结束后
print_gpu_memory("Phase 1 结束")
print_cuda_tensors("Phase 1 结束")

# Phase 2 结束后
print_gpu_memory("Phase 2 结束")
print_cuda_tensors("Phase 2 结束")

# Phase 3 开始前
aggressive_cleanup()
print_gpu_memory("Phase 3 开始前")
print_cuda_tensors("Phase 3 开始前")
```

### 方法 3：对比测试

运行两个实验：
1. 只运行 Phase 1 (Baseline Attack)，记录峰值显存
2. 运行完整 pipeline (Phase 1 → Phase 2 → Phase 3)，记录 Phase 3 的峰值显存

如果修复有效，Phase 3 的显存占用应该与 Phase 1 相近。

## 其他可能的优化

### 1. 使用 registry 模式

如果磁盘空间充足，建议使用 `cache_mode="registry"`，这样 defense 训练完成后会将模型保存到磁盘，Phase 3 从磁盘加载，避免在内存中保存 `state_dict`。

```bash
DEFENSE_CACHE_MODE=registry bash experiments/run_main_omni.sh
```

### 2. 减小 batch size

如果显存仍然不足，可以减小 defense 的 batch size：

```bash
DEFENSE_BATCH_SIZE=1 bash experiments/run_main_omni.sh
```

### 3. 使用梯度累积

增加梯度累积步数，减小实际 batch size：

```bash
DEFENSE_GRAD_ACCUM=4 bash experiments/run_main_omni.sh
```

## 总结

修复后，defense 阶段的所有显存占用对象都会被显式清理，Phase 3 开始时的显存占用应该与 Phase 1 相近。如果问题仍然存在，可能需要进一步检查：

1. 是否有全局变量持有模型引用
2. 是否有回调函数或钩子持有模型引用
3. 是否有日志或调试代码持有中间结果

可以使用 `tools/debug_memory.py` 中的工具来进一步诊断。
