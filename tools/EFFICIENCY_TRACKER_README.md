# 训练效率测量工具 - 使用总结

## 已创建的文件

1. **`tools/efficiency_tracker.py`** - 核心追踪器类
   - `EfficiencyTracker`: 主追踪器类
   - `EfficiencyMetrics`: 指标数据类
   - 支持三种对比模式的自动计算

2. **`tools/efficiency_integration_guide.md`** - 详细集成指南
   - 如何集成到现有代码
   - 论文中如何使用
   - 注意事项

3. **`tools/efficiency_integration_patch.py`** - 具体代码修改示例
   - 展示需要修改的具体位置
   - 包含完整的代码片段

4. **`tools/test_efficiency_tracker.py`** - 测试脚本
   - 验证功能正确性
   - 展示输出格式

## 三种测量模式说明

### 模式1：相同step数对比
**适用场景**: 你们的方法效果好且轻量

**测量内容**:
- 在相同训练步数下（如30K steps）
- 对比最终性能（PSNR, LPIPS）
- 对比训练时间

**论文描述**:
> "Under the same training budget (30K steps), our method achieves 1.3 dB higher PSNR (29.8 vs 28.5) while reducing training time by 50% (2.1 hrs vs 4.2 hrs)."

### 模式2：训练到收敛对比
**适用场景**: 你们的方法收敛更快

**测量内容**:
- 达到收敛所需的步数
- 达到收敛所需的时间
- 最终性能对比

**论文描述**:
> "When trained to convergence, our method requires 40% fewer steps (18K vs 30K) and 50% less time (1.8 hrs vs 3.6 hrs) while achieving better final performance."

### 模式3：相同时间预算对比
**适用场景**: 强调实际应用效率

**测量内容**:
- 在相同训练时间内（如2小时）
- 能训练多少步
- 达到什么性能

**论文描述**:
> "Given the same training time budget (2 hours), our method can complete 35K steps compared to baseline's 20K steps, achieving 2.1 dB higher PSNR."

## 快速开始

### 1. 测试功能
```bash
cd /root/autodl-tmp/3d-defense-migration/3d-defense
python tools/test_efficiency_tracker.py
```

### 2. 集成到 run_pipeline.py

需要修改两个文件：

**A. `script/run_pipeline.py`** (3处修改)
1. 导入 `EfficiencyTracker`
2. 在 Phase 1 和 Phase 3 前初始化并启动 tracker
3. 在末尾生成报告

**B. `training/finetuner.py`** (1处修改)
1. 在 `step_history.append()` 后调用 `tracker.record()`

详细代码见 `tools/efficiency_integration_patch.py`

### 3. 运行实验
```bash
python script/run_pipeline.py \
    --gpu 0 \
    --config configs/config.yaml \
    --trap_losses position,scale \
    --tag efficiency_test \
    --attack_epochs 5 \
    --defense_epochs 25 \
    --eval_every_steps 10
```

### 4. 查看结果
- 终端输出: 三种对比模式的结果
- `output_dir/efficiency_report.json`: 完整数据
- 可直接用于论文表格

## 论文中的使用建议

### 主对比表（放在 Results 部分）

```markdown
| Method      | PSNR↑ | LPIPS↓ | Time | FLOPs/step | Memory | Speedup |
|-------------|-------|--------|------|------------|--------|---------|
| Baseline    | 28.5  | 0.15   | 4.2h | 120 G      | 18 GB  | 1.0x    |
| Ours        | 29.8  | 0.12   | 2.1h | 65 G       | 12 GB  | 2.0x    |
| Improvement | +1.3  | -0.03  | 50%↓ | 46%↓       | 33%↓   | -       |
```

### 文字描述（放在 Abstract/Introduction）

**如果效果好且轻量**:
> "Our method achieves X% higher PSNR while reducing training time by Y% and computational cost by Z%, demonstrating superior efficiency without sacrificing quality."

**如果效果略差但很轻量**:
> "While achieving comparable performance (X dB PSNR difference), our method significantly reduces training cost by Y% in time and Z% in FLOPs, making it more practical for resource-constrained scenarios."

### 训练曲线图

建议绘制：
- 横轴: 训练时间（小时）或 FLOPs
- 纵轴: PSNR 或 LPIPS
- 两条曲线: Baseline vs Ours
- 你们的曲线应该在左上角（更快达到更好性能）

## 注意事项

1. **FLOPs 计算**: 如果不确定确切值，可以设为 `None`，只报告时间和steps
2. **多次运行**: 建议运行3次取平均值，报告标准差
3. **公平对比**: 确保两种方法使用相同的硬件、batch size、数据集
4. **收敛判断**: 默认使用最终性能的95%作为收敛阈值，可根据需要调整
5. **显存测量**: 自动使用 `torch.cuda.max_memory_allocated()`

## 下一步

1. ✅ 测试工具功能 (`python tools/test_efficiency_tracker.py`)
2. ⬜ 集成到 `run_pipeline.py` 和 `finetuner.py`
3. ⬜ 运行完整实验收集数据
4. ⬜ 生成论文表格和图表
5. ⬜ 撰写论文相关章节

## 问题排查

如果遇到问题：
1. 检查 `config._efficiency_tracker` 是否正确传递
2. 确认 `tracker.start()` 在训练前调用
3. 确认 `tracker.record()` 在每个 eval 点调用
4. 查看 `efficiency_report.json` 中的原始数据

有问题随时问我！
