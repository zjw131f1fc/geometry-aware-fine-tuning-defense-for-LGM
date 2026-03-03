# 输入加噪功能 - 快速参考

## 一句话总结

在防御训练时对输入图像添加高斯噪声（noise_scale=0.24，约5%扰动），提高模型对输入扰动的鲁棒性。

## 快速启用

在 `configs/config.yaml` 中：

```yaml
defense:
  input_noise:
    enabled: true
    noise_scale: 0.24  # 5%扰动（推荐）
    warmup_steps: 300
```

## 关键信息

### 加噪位置
- **在哪加**: UNet输入 `[B, 4, 9, H, W]`
- **加在什么上**: RGB图像（前3通道）+ Rays坐标（后6通道）
- **不是embedding**: 直接在输入特征上加噪

### 输入范围
- **RGB通道**: ImageNet归一化后，范围 [-2.1, 2.6]，跨度4.7
- **Rays通道**: Plucker坐标，范围 [-1, 1]，跨度2
- **noise_scale=0.24**: 对RGB约5%扰动，对Rays约12%扰动

### 噪声强度对照表

| noise_scale | RGB扰动 | 说明 |
|-------------|---------|------|
| 0.10 | 2% | 轻度 |
| 0.24 | 5% | **推荐** |
| 0.47 | 10% | 强度 |

### 训练流程

每个step包含3次前向（如果都启用）：
1. 干净输入 + 干净权重 → trap loss
2. 干净输入 + 加噪权重 → trap loss（参数鲁棒性）
3. 加噪输入 + 干净权重 → trap loss（输入鲁棒性）

### 性能开销
- 训练时间: +30-50%
- 显存: +20-30%
- 推理: 无影响

## 监控指标

训练日志中查看：
```
scale_static: -8.234              # 干净输入
param_noisy_scale_static: -8.123  # 参数加噪
input_noisy_scale_static: -7.891  # 输入加噪
```

## 完整文档

- 使用指南: `docs/input_noise_guide.md`
- 技术细节: `docs/input_noise_technical.md`
- 总结: `docs/input_noise_summary.md`
