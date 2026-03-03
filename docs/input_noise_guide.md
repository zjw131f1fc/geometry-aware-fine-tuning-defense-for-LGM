# 输入加噪功能使用指南

## 功能简介

输入加噪（Input Noise）是一种数据增强技术，通过在训练时对输入图像添加高斯噪声来提高模型的鲁棒性。与参数加噪（Parameter Noise）不同，输入加噪直接作用于输入数据，模拟输入扰动场景。

## 快速开始

### 1. 配置文件设置

在 `configs/config.yaml` 中添加或修改 `defense.input_noise` 配置：

```yaml
defense:
  # ... 其他配置 ...

  # 输入加噪配置
  input_noise:
    enabled: true           # 启用输入加噪
    noise_scale: 0.02       # 噪声强度（建议 0.01~0.05）
    warmup_steps: 300       # warmup步数（0=不使用warmup）
```

### 2. 运行训练

正常运行防御训练脚本即可：

```bash
python script/train_defense_with_eval.py --config configs/config.yaml
```

训练日志会显示输入加噪的配置信息：

```
DefenseTrainer 初始化
  防御方法: geotrap
  ...
  输入加噪: σ=0.02 (warmup_steps=300)
```

## 配置参数详解

### enabled
- **类型**: boolean
- **默认值**: false
- **说明**: 是否启用输入加噪功能

### noise_scale
- **类型**: float
- **默认值**: 0.24 (对应RGB约5%扰动)
- **建议范围**: 0.10 ~ 0.47
- **说明**: 噪声强度（高斯噪声的标准差）
  - 较小值（0.10）: 约2%扰动，训练更稳定
  - 中等值（0.24）: 约5%扰动，**推荐默认值**
  - 较大值（0.47）: 约10%扰动，极强鲁棒性但可能影响收敛

**重要**: 输入数据的实际范围
- 输入形状: `[B, 4, 9, H, W]`（4个视角，9通道特征）
- 前3通道: ImageNet归一化的RGB，范围约 [-2.1, 2.6]，跨度4.7
- 后6通道: Rays Plucker坐标，范围约 [-1, 1]，跨度2
- `noise_scale=0.24` 对应RGB约5%扰动 (0.24/4.7≈5%)，对Rays约12%扰动 (0.24/2≈12%)

### warmup_steps
- **类型**: int
- **默认值**: 300
- **说明**: 噪声warmup的优化器步数
  - `0`: 不使用warmup，从训练开始就使用完整噪声
  - `> 0`: 噪声从0线性增长到 `noise_scale`

## 使用场景

### 场景1: 基础输入加噪（推荐）

适用于大多数情况，提供5%的输入鲁棒性：

```yaml
defense:
  input_noise:
    enabled: true
    noise_scale: 0.24  # 5%扰动
    warmup_steps: 100
```

### 场景2: 强鲁棒性训练

需要更强的输入鲁棒性（10%扰动）：

```yaml
defense:
  input_noise:
    enabled: true
    noise_scale: 0.47  # 10%扰动
    warmup_steps: 300  # 更长的warmup避免训练不稳定
```

### 场景3: 组合参数加噪和输入加噪

同时启用两种噪声获得最强鲁棒性：

```yaml
defense:
  # 参数加噪
  robustness:
    enabled: true
    noise_scale: 0.005
    warmup_steps: 100

  # 输入加噪
  input_noise:
    enabled: true
    noise_scale: 0.02
    warmup_steps: 300
```

## 训练流程

启用输入加噪后，每个训练step包含以下前向传播：

1. **干净前向**: 使用原始输入计算 trap loss
2. **参数加噪前向**（如果启用 `robustness.enabled`）: 对模型参数加噪后前向
3. **输入加噪前向**（如果启用 `input_noise.enabled`）: 对输入图像加噪后前向

所有前向的loss会累积并用于反向传播，确保模型在多种扰动下都能保持陷阱效果。

## 监控指标

训练日志中会记录输入加噪相关的指标（带 `input_noisy_` 前缀）：

```
loss: 2.3456
scale_static: -8.234
opacity_static: -3.456
input_noisy_scale_static: -7.891    # 输入加噪后的scale trap
input_noisy_opacity_static: -3.123  # 输入加噪后的opacity trap
```

## 性能影响

- **训练时间**: 启用输入加噪会增加约 30-50% 的训练时间（额外的前向传播）
- **显存占用**: 增加约 20-30%（需要保存额外的计算图）
- **推理时间**: 无影响（推理时不使用加噪）

## 调试建议

### 问题1: 训练不稳定

**症状**: loss震荡剧烈，难以收敛

**解决方案**:
1. 减小 `noise_scale`（如从 0.05 降到 0.02）
2. 增加 `warmup_steps`（如从 100 增到 300）
3. 检查梯度裁剪设置 `training.gradient_clip`

### 问题2: 鲁棒性不足

**症状**: 防御效果不明显

**解决方案**:
1. 增大 `noise_scale`
2. 同时启用参数加噪和输入加噪
3. 增加训练步数

### 问题3: 显存不足

**症状**: CUDA out of memory

**解决方案**:
1. 减小 `defense.batch_size`
2. 增加 `defense.gradient_accumulation_steps`
3. 只启用一种加噪方式（参数或输入）

## 测试

运行测试脚本验证功能：

```bash
# 基础功能测试
python experiments/test_input_noise.py

# 完整训练测试（smoke test）
python script/train_defense_with_eval.py --config configs/config_smoke.yaml
```

## 参考文档

- [噪声Warmup详细说明](docs/noise_warmup.md)
- [防御训练完整文档](docs/Defense_Metrics_Tracking.md)
- [实验配置示例](experiments/run_ablation_coupling.sh)
