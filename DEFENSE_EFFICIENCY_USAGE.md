# 防御训练效率测量 - 使用说明

## 快速使用

### 1. 测量 GeoTrap 防御训练效率

```bash
MEASURE_EFFICIENCY=1 TAG=geotrap bash experiments/run_single.sh 0
```

### 2. 测量 Naive Unlearning 防御训练效率

```bash
MEASURE_EFFICIENCY=1 TAG=naive_unlearning bash experiments/run_single.sh 0
```

## 输出结果

每次运行后会在输出目录生成 `defense_efficiency.json`，包含：

```json
{
  "method_name": "Defense Training (geotrap)",
  "flops_per_step": null,
  "peak_memory_mb": 12288.0,
  "history": [
    {
      "step": 10,
      "epoch": 1,
      "elapsed_time": 25.3,
      "step_time": 2.53,
      "loss": 0.245,
      ...
    },
    ...
  ]
}
```

终端会显示：

```
================================================================================
防御训练效率报告
================================================================================

效率数据已保存: output/.../defense_efficiency.json

【防御训练统计】
方法: Defense Training (geotrap)
总步数: 1000
总时间: 0.85 小时
平均步时: 3.06 秒/步
峰值显存: 12288 MB

【效率数据】
| 指标         | 数值                    |
|--------------|-------------------------|
| 训练步数     | 1,000 steps      |
| 训练时间     | 0.85 hours       |
| 平均步时     | 3.060 s/step       |
| 峰值显存     | 12288 MB          |

提示: 运行不同防御方法后，可以对比各自的 defense_efficiency.json
================================================================================
```

## 对比两种方法

运行完两次实验后，手动对比两个 `defense_efficiency.json` 文件：

```bash
# GeoTrap 的结果
cat output/.../geotrap_xxx/defense_efficiency.json

# Naive Unlearning 的结果
cat output/.../naive_unlearning_xxx/defense_efficiency.json
```

## 论文表格

手动整理成表格：

```markdown
| Defense Method    | Steps | Time (h) | Time/Step (s) | Memory (GB) |
|-------------------|-------|----------|---------------|-------------|
| GeoTrap           | 1000  | 0.85     | 3.06          | 12.0        |
| Naive Unlearning  | 1000  | 1.20     | 4.32          | 15.5        |
| Speedup           | -     | 1.4x     | 1.4x          | 23% less    |
```

## 论文描述

> "Our GeoTrap defense method demonstrates superior training efficiency compared to naive unlearning. Under the same training budget (1000 steps), GeoTrap completes training in 0.85 hours, 1.4× faster than naive unlearning (1.20 hours), while consuming 23% less GPU memory (12GB vs 15.5GB)."

## 注意事项

1. **相同配置**: 确保两种方法使用相同的 steps/epochs 配置
2. **多次运行**: 建议运行3次取平均值
3. **清除缓存**: 如果使用了 defense cache，需要清除后重新训练才能记录数据

就这么简单！
