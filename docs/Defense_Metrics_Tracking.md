# 防御指标追踪使用指南

## 概述

在攻击训练脚本中添加了防御指标追踪功能，可以实时监控防御陷阱在攻击过程中的变化。

## 追踪的指标

### 1. 陷阱强度指标
- **position_static**: 位置塌缩指标（越负越强，趋向 0 表示陷阱被破坏）
- **scale_static**: 尺度各向异性指标（越负越强，趋向 0 表示陷阱被破坏）
- **opacity_static**: 透明度塌缩指标（越负越强，趋向 0 表示陷阱被破坏）

### 2. 防御机制核心指标（重要！）
- **coupling_value**: 乘法耦合强度（越负越强，趋向 0 表示耦合被破坏）
  - 计算公式：`-(∏(1 - L_i) - 1)`
  - 反映不同 trap 之间的相互放大效应
- **grad_cosine_sim**: 梯度余弦相似度（反映梯度冲突状态）
  - 交替反向模式：应趋向 -1（梯度反向对齐）
  - 正交化模式：应趋向 0（梯度正交）
  - 趋向 1 表示梯度对齐（冲突被破坏）

### 3. Gaussian 统计信息
- **position_std**: 位置分布的标准差（反映 Gaussian 的空间分布）
- **scale_mean/std**: 尺度的均值和标准差（反映 Gaussian 的大小分布）
- **opacity_mean/std**: 透明度的均值和标准差（反映 Gaussian 的可见性）

### 4. Source 质量指标
- **source_psnr**: Source 数据上的 PSNR（越高越好，下降表示攻击影响了正常能力）
- **source_lpips**: Source 数据上的 LPIPS（越低越好，上升表示攻击影响了正常能力）

## 使用方法

### 方法 1: 使用测试脚本（推荐）

```bash
# 测试 position+scale 防御模型
./script/test_attack_on_defense.sh 0 output/sweep_combos_20260221_011903/position+scale/model_defense.pth position+scale

# 参数说明:
# - 第一个参数: GPU ID
# - 第二个参数: 防御模型路径
# - 第三个参数: trap 组合名称（用于标识）
```

### 方法 2: 手动运行

1. 修改 `configs/config.yaml`，将 `model.resume` 指向防御模型：
```yaml
model:
  resume: output/sweep_combos_20260221_011903/position+scale/model_defense.pth
```

2. 运行攻击训练：
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file acc_configs/gpu1.yaml \
    script/attack_train_ddp.py \
    --config configs/config.yaml
```

## 输出文件

### 1. 日志文件
- **路径**: `output/attack_*/defense_metrics_log.json`
- **内容**: 每个 epoch 的防御指标和 Gaussian 统计信息
- **格式**: JSON，包含完整的历史记录

### 2. 可视化图表
使用 `plot_defense_metrics.py` 生成图表：

```bash
python script/plot_defense_metrics.py output/attack_*/defense_metrics_log.json
```

生成的图表包含：
- 第一行：三个陷阱指标的变化趋势
- 第二行：Gaussian 统计信息的变化
- 第三行：Source 质量指标 + 陷阱综合对比

### 3. 渲染结果
- **路径**: `output/attack_*/renders/`
- **内容**: 每个 epoch 的渲染对比图（GT vs Pred）

## 解读指标

### 防御成功的标志
1. **陷阱指标保持负值**: position_static 和 scale_static 在攻击后仍然是负数（如 -3 以下）
2. **耦合保持强**: coupling_value 保持负值（如 -10 以下）
3. **梯度冲突保持**: grad_cosine_sim 保持在 -1 附近（交替反向模式）或 0 附近（正交模式）
4. **Source 质量保持**: source_psnr 没有明显下降，source_lpips 没有明显上升
5. **渲染质量差**: Target 数据的渲染结果模糊、扭曲

### 防御失败的标志
1. **陷阱指标趋向 0**: position_static 和 scale_static 从负值快速上升到接近 0
2. **耦合被破坏**: coupling_value 从负值快速上升到接近 0
3. **梯度冲突被破坏**: grad_cosine_sim 从 -1 或 0 趋向 1（梯度对齐）
4. **Gaussian 统计恢复正常**: position_std、scale_mean 等恢复到正常范围
5. **渲染质量好**: Target 数据的渲染结果清晰、正确

### 关键诊断点
- **如果 coupling_value 快速趋向 0**：说明乘法耦合被破坏，不同 trap 之间失去了相互放大效应
- **如果 grad_cosine_sim 从 -1 趋向 0 或 1**：说明梯度冲突被破坏，攻击者可以同时修复多个 trap
- **如果只有单个 trap 指标上升**：说明攻击者找到了绕过某个 trap 的方法，但其他 trap 仍然有效

## 示例输出

```
Epoch 5/10 - Loss: 0.0234, PSNR: 24.56

  计算防御指标...
  [Defense Metrics]
    position_static: -4.2341 (越负越强)
    scale_static: -5.6789 (越负越强)
    opacity_static: -2.1234 (越负越强)
    coupling_value: -12.3456 (越负越强)
    grad_cosine_sim: -0.8234 (应趋向-1或0)
  [Gaussian Stats]
    position: std=0.3456, range=[-0.987, 0.876]
    scale: mean=0.0234, std=0.0123, range=[0.001, 0.156]
    opacity: mean=0.4567, std=0.2345, range=[0.001, 0.987]

  评估 source 质量...
  [Source] PSNR=22.34, LPIPS=0.0456
```

## 注意事项

1. **计算开销**: 每个 epoch 会额外计算防御指标，增加约 10-20% 的时间
2. **样本数量**: 默认使用 10 个验证样本计算指标，可以在 `compute_defense_metrics()` 中修改 `num_samples` 参数
3. **日志文件**: 每个 epoch 都会更新日志文件，可以实时查看进度

## 快速诊断

如果防御被轻易攻破，检查以下几点：

1. **查看 epoch 0 的 baseline**:
   - position_static 和 scale_static 应该是负值（如 -5 以下）
   - 如果 baseline 就接近 0，说明防御训练没有成功嵌入陷阱

2. **查看变化趋势**:
   - 如果陷阱指标在前 1-2 个 epoch 就快速上升到 0，说明陷阱太弱
   - 如果陷阱指标缓慢上升，说明防御有一定效果但不够强

3. **对比不同 trap 组合**:
   - 使用可视化脚本对比不同组合的效果
   - 找出最有效的 trap 组合

## 下一步

根据追踪结果，可以：
1. 调整防御训练的超参数（lambda_trap、lambda_conflict 等）
2. 尝试不同的 trap 组合
3. 增加防御训练的 epochs
4. 调整目标层的选择
