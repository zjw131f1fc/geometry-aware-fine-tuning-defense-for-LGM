# 训练效率追踪集成指南

## 概述

`EfficiencyTracker` 支持三种论文常用的训练效率对比模式：

1. **相同step数对比** - 展示per-step效率优势
2. **训练到收敛对比** - 展示收敛速度优势
3. **相同时间预算对比** - 展示实际应用效率

## 快速集成到 run_pipeline.py

### 1. 导入模块

在 `run_pipeline.py` 顶部添加：

```python
from tools.efficiency_tracker import EfficiencyTracker
```

### 2. 在训练前初始化tracker

在 `run_attack()` 调用前：

```python
# 创建tracker
baseline_tracker = EfficiencyTracker(
    method_name="Baseline",
    flops_per_step=120e9,  # 如果已知每step的FLOPs
)
ours_tracker = EfficiencyTracker(
    method_name="Ours (GeoTrap)",
    flops_per_step=65e9,
)

baseline_tracker.start()
```

### 3. 在训练循环中记录

修改 `finetuner.py` 的 `run_attack()` 函数，在每次 `step_history.append(metrics)` 后添加：

```python
# 在 line 1155 附近
step_history.append(metrics)

# 添加效率追踪
if hasattr(config, '_efficiency_tracker'):
    config._efficiency_tracker.record(
        step=global_step,
        epoch=epoch,
        step_time=avg_step_time,
        masked_psnr=metrics.get('masked_psnr'),
        masked_lpips=metrics.get('masked_lpips'),
        psnr=metrics.get('psnr'),
        lpips=metrics.get('lpips'),
        loss=metrics.get('loss'),
    )
```

### 4. 生成报告

在 `run_pipeline.py` 的 `main()` 函数末尾：

```python
# 生成效率对比报告
efficiency_report = ours_tracker.generate_report(baseline_tracker=baseline_tracker)

# 保存报告
import json
report_path = os.path.join(output_dir, 'efficiency_report.json')
with open(report_path, 'w') as f:
    json.dump(efficiency_report, f, indent=2)

# 打印关键指标
print("\n" + "="*60)
print("训练效率对比报告")
print("="*60)

# 最终性能
print(f"\n【最终性能】")
print(f"Baseline: PSNR={efficiency_report['baseline_final']['masked_psnr']:.2f}, "
      f"Time={efficiency_report['baseline_final']['time_hours']:.2f}h")
print(f"Ours:     PSNR={efficiency_report['final_performance']['masked_psnr']:.2f}, "
      f"Time={efficiency_report['final_performance']['time_hours']:.2f}h")

# 三种对比模式
if 'comparison_same_steps' in efficiency_report:
    comp = efficiency_report['comparison_same_steps']
    print(f"\n【模式1：相同step数 ({comp['target_step']} steps)】")
    print(f"PSNR提升: {comp['psnr_improvement']:+.2f}")
    print(f"时间加速: {comp['time_speedup']:.2f}x")

if 'comparison_convergence' in efficiency_report:
    comp = efficiency_report['comparison_convergence']
    print(f"\n【模式2：训练到收敛】")
    print(f"Steps减少: {comp['step_reduction']*100:.1f}%")
    print(f"时间加速: {comp['time_speedup']:.2f}x")

if 'comparison_same_time' in efficiency_report:
    comp = efficiency_report['comparison_same_time']
    print(f"\n【模式3：相同时间预算】")
    print(f"PSNR提升: {comp['psnr_improvement']:+.2f}")
    print(f"Steps增加: {comp['ours_steps'] - comp['baseline_steps']} steps")

print("="*60)
```

## 完整示例：修改后的 run_attack() 调用

```python
# Phase 1: Baseline Attack
baseline_tracker = EfficiencyTracker(method_name="Baseline Attack")
baseline_tracker.start()
config._efficiency_tracker = baseline_tracker  # 传递给训练函数

baseline_step_history, baseline_source_metrics, baseline_target_metrics = run_attack(
    config=config,
    target_train_loader=target_train_loader,
    source_val_loader=source_val_loader,
    attack_epochs=attack_epochs,
    attack_steps=attack_steps,
    eval_every_steps=args.eval_every_steps,
    # ... 其他参数
)

# Phase 3: Post-Defense Attack
ours_tracker = EfficiencyTracker(method_name="Post-Defense Attack")
ours_tracker.start()
config._efficiency_tracker = ours_tracker

postdef_step_history, postdef_source_metrics, postdef_target_metrics = run_attack(
    config=config,
    # ... 参数
)

# 生成报告
efficiency_report = ours_tracker.generate_report(baseline_tracker=baseline_tracker)
```

## 论文中如何使用

### 表格示例

```markdown
| Method   | PSNR↑ | LPIPS↓ | Time (hrs) | FLOPs/step | Total FLOPs | GPU Mem |
|----------|-------|--------|------------|------------|-------------|---------|
| Baseline | 28.5  | 0.15   | 4.2        | 120 G      | 3.6 T       | 18 GB   |
| Ours     | 29.8  | 0.12   | 2.1        | 65 G       | 1.4 T       | 12 GB   |
| Improve  | +1.3  | -0.03  | 2.0x       | 46% less   | 61% less    | 33% less|
```

### 文字描述

> Our method achieves 1.3 dB higher PSNR (29.8 vs 28.5) while reducing training time by 50% (2.1 hrs vs 4.2 hrs) and computational cost by 61% (1.4T vs 3.6T FLOPs). Under the same training budget (30K steps), our method outperforms the baseline by 1.3 dB PSNR. When trained to convergence, our method requires 40% fewer steps while achieving better final performance.

## 注意事项

1. **FLOPs计算**: 如果不知道确切的FLOPs，可以设为None，只报告时间和steps
2. **收敛判断**: `find_convergence_point()` 默认使用最终性能的95%作为阈值，可以根据需要调整
3. **显存测量**: 自动使用 `torch.cuda.max_memory_allocated()`，确保在训练前reset
4. **多次运行**: 建议运行3次取平均值，报告标准差
