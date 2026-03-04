# Skip Baseline 功能说明

## 功能描述

`run_ablation_mechanisms.sh` 现在支持跳过 baseline attack 阶段，直接从 defense training 开始。这在以下场景很有用：

1. **已有 baseline 缓存**：之前运行过相同配置的实验，baseline 结果已缓存
2. **只关注 defense 效果**：不需要重新运行 baseline attack
3. **节省时间**：跳过 baseline attack 可以显著减少实验时间

## 使用方法

### 方法 1：环境变量

```bash
SKIP_BASELINE=1 bash experiments/run_ablation_mechanisms.sh 0,1,2,3
```

### 方法 2：在脚本中设置

```bash
export SKIP_BASELINE=1
bash experiments/run_ablation_mechanisms.sh 0,1,2,3
```

### 方法 3：结合其他环境变量

```bash
SKIP_BASELINE=1 \
DEFENSE_CACHE_MODE=readonly \
EVAL_EVERY_STEPS=20 \
bash experiments/run_ablation_mechanisms.sh 0,1,2,3
```

## 环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SKIP_BASELINE` | `0` | 设置为 `1` 跳过 baseline attack |
| `DEFENSE_CACHE_MODE` | `none` | 防御缓存模式：`none`/`readonly`/`registry` |
| `EVAL_EVERY_STEPS` | `10` | 评估间隔步数 |
| `DEFENSE_BATCH_SIZE` | - | 防御训练 batch size（覆盖配置文件） |
| `DEFENSE_GRAD_ACCUM` | - | 防御梯度累积步数（覆盖配置文件） |

## 注意事项

### 1. Baseline 缓存

跳过 baseline 时，pipeline 会尝试从缓存加载 baseline 结果。缓存位置：

```
output/baseline_cache/<hash>/
```

如果缓存不存在，pipeline 会报错或产生不完整的结果。

### 2. 缓存 Hash

Baseline 缓存的 hash 基于以下配置计算：
- Attack 配置（lr, optimizer, batch_size, etc.）
- Target 数据配置（dataset, categories, samples）
- 模型配置（size, resume）

如果这些配置改变，缓存会失效，需要重新运行 baseline。

### 3. 实验完整性

跳过 baseline 后，实验结果中不会包含：
- Baseline attack 的训练历史
- Baseline 的 step-by-step 指标
- Baseline 的渲染图片

但会包含：
- Defense training 的完整历史
- Post-defense attack 的完整历史
- 最终的对比指标（如果 baseline 缓存存在）

## 示例

### 示例 1：快速测试 defense 配置

```bash
# 第一次运行：生成 baseline 缓存
bash experiments/run_ablation_mechanisms.sh 0

# 后续运行：跳过 baseline，只测试不同的 defense 配置
SKIP_BASELINE=1 bash experiments/run_ablation_mechanisms.sh 0
```

### 示例 2：多卡并行 + 跳过 baseline

```bash
SKIP_BASELINE=1 bash experiments/run_ablation_mechanisms.sh 0,1,2,3
```

### 示例 3：结合其他优化

```bash
# 跳过 baseline + 使用缓存的 defense 模型 + 减小显存占用
SKIP_BASELINE=1 \
DEFENSE_CACHE_MODE=readonly \
DEFENSE_BATCH_SIZE=2 \
DEFENSE_GRAD_ACCUM=4 \
bash experiments/run_ablation_mechanisms.sh 0,1
```

## 验证

运行时，如果 `SKIP_BASELINE=1` 生效，你会看到：

```
==========================================
防御机制消融实验
测试配置: configs/config.yaml
Output: output/experiments_output/ablation_mechanisms_20260304_150134
模式: 跳过 Baseline Attack
==========================================
```

在日志文件中，你会看到：

```
=== GPU 0: baseline_all ===
Params: --grad_surgery_enabled false ...
Skip baseline: true

...
attack.skip_baseline: True
...

================================================================================
  跳过 Phase 1: Baseline Attack
================================================================================
```

## 故障排查

### 问题 1：Baseline 缓存未找到

**错误信息**：
```
[Cache] baseline 缓存未找到: output/baseline_cache/<hash>
```

**解决方法**：
1. 先运行一次不跳过 baseline 的实验，生成缓存
2. 或者不使用 `SKIP_BASELINE=1`

### 问题 2：Baseline 仍然运行

**可能原因**：
1. 环境变量未正确设置
2. 使用了旧版本的脚本

**解决方法**：
```bash
# 确认环境变量
echo $SKIP_BASELINE  # 应该输出 1

# 重新运行
SKIP_BASELINE=1 bash experiments/run_ablation_mechanisms.sh 0
```

### 问题 3：结果不完整

**可能原因**：
Baseline 缓存存在但不完整

**解决方法**：
删除旧缓存，重新生成：
```bash
rm -rf output/baseline_cache/<hash>
bash experiments/run_ablation_mechanisms.sh 0
```

## 性能对比

| 模式 | 单个任务时间 | 6个任务总时间（单卡） | 6个任务总时间（4卡） |
|------|-------------|---------------------|---------------------|
| 完整运行 | ~30分钟 | ~3小时 | ~45分钟 |
| 跳过 baseline | ~20分钟 | ~2小时 | ~30分钟 |

**节省时间**：约 33%

## 相关文档

- [多卡并行说明](./README_multi_gpu.md)
- [显存优化指南](../docs/memory_issue_fix.md)
- [Pipeline 配置说明](../docs/pipeline_config.md)
