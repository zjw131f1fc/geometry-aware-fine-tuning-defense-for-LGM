# 输入加噪功能总结

## 功能概述

已成功添加输入加噪（Input Noise）功能，用于提高防御训练的鲁棒性。

## 实现内容

### 1. 核心代码修改

**文件**: `training/defense_trainer.py`

- 添加输入加噪配置读取（line 113-123）
- 实现 `_add_input_noise()` 方法（line 730-750）
- 更新 `_update_noise_scale()` 支持输入噪声warmup（line 695-710）
- 集成输入加噪到训练流程（line 845-860）

**关键特性**:
- 支持独立的噪声强度和warmup配置
- 与参数加噪并行工作
- 多前向机制：干净 + 参数加噪 + 输入加噪

### 2. 配置文件更新

**文件**: `configs/config.yaml`, `configs/config_smoke.yaml`

```yaml
defense:
  input_noise:
    enabled: false
    noise_scale: 0.24  # 对RGB约5%扰动
    warmup_steps: 300
```

### 3. 文档

- `docs/input_noise_guide.md` - 使用指南
- `docs/input_noise_technical.md` - 技术细节
- `docs/noise_warmup.md` - 更新包含输入加噪说明

### 4. 测试

- `experiments/test_input_noise.py` - 功能测试脚本
- 所有测试通过 ✓

## 技术细节

### 输入数据结构

- **形状**: `[B, 4, 9, H, W]`
- **通道0-2**: ImageNet归一化RGB，范围 [-2.1, 2.6]
- **通道3-8**: Rays Plucker坐标，范围 [-1, 1]

### 噪声强度标定

| noise_scale | RGB扰动 | Rays扰动 | 用途 |
|-------------|---------|----------|------|
| 0.10 | 2% | 5% | 轻度 |
| 0.24 | 5% | 12% | **推荐** |
| 0.47 | 10% | 24% | 强度 |

### 加噪位置

噪声加在UNet输入上（`model.forward_gaussians(input_images_noisy)`），同时作用于RGB和Rays。

## 使用方法

### 启用输入加噪

```yaml
defense:
  input_noise:
    enabled: true
    noise_scale: 0.24
    warmup_steps: 300
```

### 组合使用

```yaml
defense:
  robustness:
    enabled: true
    noise_scale: 0.005
    warmup_steps: 100

  input_noise:
    enabled: true
    noise_scale: 0.24
    warmup_steps: 300
```

## 性能影响

- **训练时间**: +30-50%（额外前向传播）
- **显存占用**: +20-30%（额外计算图）
- **推理时间**: 无影响

## 验证

运行测试：
```bash
python experiments/test_input_noise.py
```

输出示例：
```
✓ 基本加噪功能正常
✓ Warmup调度正常
✓ 组合噪声功能正常
✓ 噪声影响测试完成
```

## 下一步

1. 在实际防御训练中测试效果
2. 调整 `noise_scale` 以获得最佳鲁棒性
3. 监控训练指标（`input_noisy_*` 前缀）
4. 评估防御效果（ΔLPIPS）
